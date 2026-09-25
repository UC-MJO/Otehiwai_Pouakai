"""
Master dark frame construction for the B&C reduction pipeline.

Individual dark frames are grouped into calibration "clusters" by
readout mode, then by proximity in both time (Julian date) and exposure
time via DBSCAN clustering (see `_clustering_bc_darks`), and each cluster
is combined into a single master dark via sigma-clipped stacking (see
`build_master_dark_sigma_clip`). `make_master_darks` is the main entry
point: it finds clusters not yet represented in the master dark catalog,
builds masters for them, and updates the catalog CSV. `get_master_dark`
looks up the best-matching existing master dark for a given science
frame's observing conditions.
"""

import warnings
# Registering this filter here (as well as in suppress_warnings.py)
# Registering this filter here (as well as in suppress_warnings.py)
# gives some defense-in-depth for anything that imports dark_masters.py
# directly without going through an entry-point script that imports
# suppress_warnings.py first: the "pkg_resources is deprecated as an
# API" warning is raised the first time sklearn.cluster is imported
# (below), so the filter needs to be registered before that import for
# it to have any effect. See suppress_warnings.py's module docstring for
# the full explanation of why import order matters here.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
warnings.filterwarnings("ignore")

from astropy.io import fits
from astropy.time import Time
import pandas as pd
import numpy as np
from glob import glob
import os
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm import tqdm
import gc

from pathlib import Path
import subprocess

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, AgglomerativeClustering
import inspect

from collections import defaultdict

from .running_stats import RunningStats

import re
import logging

logger = logging.getLogger(__name__)

from . import config

# Resolved once at import time from POUAKAI_MASTER_DARK_DIR /
# POUAKAI_CAL_LIST_DIR (or this site's shared-storage defaults) -- see
# config.py. Both directories are created on first resolution.
MASTER_DARK_LOCATION = config.master_dark_dir()
CAL_LIST_LOCATION = config.cal_list_dir()


def normalize_readout(readout):
    """
    Normalize a readout-mode value into a consistent string for use in
    filenames, handling both numeric and string inputs.

    Examples
    --------
      2.0        -> '2MHz'
      '2.0MHz'   -> '2MHz'
      '500KHz'   -> '500KHz'
      500        -> '500MHz'

    Parameters
    ----------
    readout : int, float, or str
        Raw readout-mode value as read from a FITS header or dataframe.

    Returns
    -------
    str
        Normalized readout string, e.g. '2MHz'.
    """
    if isinstance(readout, (int, float)):
        return f"{int(round(readout))}MHz"

    s = str(readout).strip()
    s = s.replace(" ", "").replace("(", "").replace(")", "")

    m = re.match(r'^(\d+(?:\.\d+)?)(MHz|KHz)$', s, re.IGNORECASE)
    if m:
        value, unit = m.groups()
        value = str(int(round(float(value))))
        return f"{value}{unit}"

    # Bare numeric string with no unit -- assume MHz, matching the
    # int/float branch above.
    try:
        return f"{int(round(float(s)))}MHz"
    except ValueError:
        return s

def _make_clustering(method, **kwargs):
    """
    Construct a scikit-learn clustering estimator (DBSCAN or
    AgglomerativeClustering) in a way that works across scikit-learn
    versions.

    scikit-learn renamed the `affinity` keyword to `metric` for
    `AgglomerativeClustering` starting in version 1.2 (deprecated in 1.4,
    removed in a later release); `DBSCAN` has always used `metric`. This
    accepts the `affinity` keyword either way and passes it through under
    whichever name the installed version's constructor actually accepts,
    so calling code doesn't need to know which scikit-learn version is
    installed.

    Parameters
    ----------
    method : class
        The clustering estimator class to construct (e.g. `DBSCAN`,
        `AgglomerativeClustering`).
    **kwargs
        Constructor keyword arguments. If `affinity` is present, it is
        translated to whichever of `metric`/`affinity` the installed
        version's constructor actually accepts.

    Returns
    -------
    An instance of `method`, constructed with the translated keywords.
    """
    sig = inspect.signature(method.__init__)
    params = sig.parameters

    if 'affinity' in kwargs:
        metric_value = kwargs.pop('affinity')
        if 'metric' in params:
            kwargs['metric'] = metric_value
        elif 'affinity' in params:
            kwargs['affinity'] = metric_value
        else:
            raise TypeError(f'{method.__name__} accepts neither metric nor affinity in this sklearn version')

    return method(**kwargs)

def _filter_bc_darks(exp_tol=1, dark_delta_t=1):
    """
    Load the catalog of all recorded dark frames, restrict to usable
    (correct detector shape, not previously flagged bad) frames, and
    group them into calibration clusters.

    Parameters
    ----------
    exp_tol : float
        Passed to `_clustering_bc_darks` as `exp_tolerance`.
    dark_delta_t : float
        Passed to `_clustering_bc_darks` as `delta_t`.

    Returns
    -------
    pandas.DataFrame
        Filtered, clustered dark frame catalog with `cluster` and
        `master_name` columns added (see `_clustering_bc_darks`).
    """
    initial_df = pd.read_csv(CAL_LIST_LOCATION + 'bc_dark_image_list.csv')

    shape_mask = initial_df['shape'].values.astype(int) == 2048
    bad_mask = initial_df['telescope'].values.astype(str) != 'bad'

    chip_filtered_df = initial_df[shape_mask & bad_mask].copy()
    print(f"Number of darks after shape filtering: {len(chip_filtered_df)}")

    chip_filtered_df = _clustering_bc_darks(chip_filtered_df, delta_t=dark_delta_t, exp_tolerance=exp_tol)

    return chip_filtered_df

def _clustering_bc_darks(chip_filtered_df, delta_t=1, exp_tolerance=1.0):
    """
    Group dark frames into calibration clusters, separately for each
    readout mode, using DBSCAN in (JD, exposure time) space.

    Frames within `delta_t` of each other in Julian date AND within
    `exp_tolerance` in exposure time are grouped into the same cluster
    (DBSCAN with the Chebyshev/max-norm metric enforces both bounds
    independently, since Chebyshev distance is the max over dimensions).
    Each resulting cluster is assigned a `master_name` based on its mean
    UTC date, median exposure time, readout mode, and detector shape;
    if more than one cluster would produce the same base name on the
    same call, later ones get a `_v2`, `_v3`, ... suffix.

    Parameters
    ----------
    chip_filtered_df : pandas.DataFrame
        Dark frame catalog, already filtered to usable frames, with at
        least `readout`, `jd`, `exptime`, and `shape` columns.
    delta_t : float
        DBSCAN neighbourhood radius in Julian date (days), after scaling.
    exp_tolerance : float
        DBSCAN neighbourhood radius in exposure time (seconds), after
        scaling.

    Returns
    -------
    pandas.DataFrame
        `chip_filtered_df` with `cluster` and `master_name` columns
        added, restricted to rows that were assigned to some cluster
        (DBSCAN noise points, label -1, are dropped). Empty DataFrame if
        no clusters were found.
    """
    result = []
    cluster_offset = 0
    name_counter = defaultdict(int)

    for readout, df_ro in chip_filtered_df.groupby('readout'):
        df_ro = df_ro.sort_values('jd').reset_index(drop=True)

        X = df_ro[['jd', 'exptime']].astype(float).values

        X_scaled = np.copy(X)
        X_scaled[:, 0] /= delta_t
        X_scaled[:, 1] /= exp_tolerance

        clustering = _make_clustering(DBSCAN, eps=1.0, min_samples=3, metric="chebyshev").fit(X_scaled)

        labels = clustering.labels_
        df_ro['cluster'] = labels

        for cl in sorted(set(labels)):
            if cl == -1:
                continue

            sub = df_ro[df_ro['cluster'] == cl].copy()

            t_start = sub['jd'].min()
            t_end = sub['jd'].max()
            mean_jd = 0.5 * (t_start + t_end)

            median_exptime = sub['exptime'].median()

            jd_utc = Time(mean_jd, format='jd').utc.strftime('%Y%m%d')
            exp_str = f"{int(round(median_exptime))}s"

            readout_val = sub['readout'].iloc[0]
            readout_str = normalize_readout(readout_val)
            shape_val = sub['shape'].iloc[0]

            base_name = f"master_dark_{jd_utc}_{exp_str}_{readout_str}_{int(shape_val)}"

            name_counter[base_name] += 1
            master_name = (base_name if name_counter[base_name] == 1
                            else f"{base_name}_v{name_counter[base_name]}")

            sub['cluster'] = cluster_offset
            sub['master_name'] = master_name

            result.append(sub)
            cluster_offset += 1

    if len(result) == 0:
        return pd.DataFrame()

    return pd.concat(result, ignore_index=True)


def make_master_darks(exp_tol=1, dark_delta_t=1, num_cores=1):
    """
    Build any master dark frames not yet present in the master dark
    catalog, and update the catalog CSV.

    Compares the set of calibration clusters found in the current dark
    frame list against the master names already recorded (with a
    successfully-built file) in `bc_master_dark_list.csv`, builds masters
    only for the new ones, and appends them to the catalog.

    Parameters
    ----------
    exp_tol : float
        Passed to `_filter_bc_darks` as `exp_tol`.
    dark_delta_t : float
        Passed to `_filter_bc_darks` as `dark_delta_t`.
    num_cores : int
        If greater than 1, build masters for different clusters in
        parallel threads.
    """
    dark_list = _filter_bc_darks(exp_tol=exp_tol, dark_delta_t=dark_delta_t)
    try:
        masters = pd.read_csv(CAL_LIST_LOCATION + 'bc_master_dark_list.csv')
    except Exception:
        masters = pd.DataFrame(columns=['name', 'telescope', 'exptime', 'jd', 'date', 'readout',
                                        'filename', 'nimages', 'shape', 'median', 'note'])

    all_names = set(dark_list['master_name'].values) if len(dark_list) else set()

    # Master names that were never successfully built (too few frames in
    # the cluster) are NOT treated as "already done" -- if more frames
    # for the same cluster arrive on a later night, the cluster should be
    # retried rather than permanently skipped. Only names with a real,
    # successfully-built master file are excluded from `new` below.
    if 'filename' in masters.columns:
        successful_mask = masters['filename'] != 'IGNORED'
    else:
        successful_mask = pd.Series([], dtype=bool)
    master_names = set(masters.loc[successful_mask, 'name'].values) if len(masters) else set()

    dark_list = dark_list.reset_index(drop=True)
    dark_list.to_csv(CAL_LIST_LOCATION + 'temp_bc_dark_list_filtered.csv', index=False)

    new = all_names - master_names
    new = list(new)
    new.sort(reverse=True)
    print('Number of new master dark entries:', len(new))

    updated_dark_list = dark_list[dark_list['master_name'].isin(new)].copy()

    if len(new) > 0:
        if num_cores > 1:
            entries = Parallel(n_jobs=num_cores, backend="threading", prefer="threads")(
                delayed(dark_processing)(updated_dark_list, cluster)
                for cluster in tqdm(updated_dark_list['cluster'].unique(), desc='Processing files'))
        else:
            entries = []
            for cluster in tqdm(updated_dark_list['cluster'].unique(), desc='Processing files'):
                entries.append(dark_processing(updated_dark_list, cluster))

        for entry in entries:
            if entry is not None:
                masters = pd.concat([masters, entry], ignore_index=True)

    masters = masters.reset_index(drop=True)
    print('Done creating master darks, total masters:', len(masters))

    masters.to_csv(CAL_LIST_LOCATION + 'bc_master_dark_list.csv', index=False)

def _bc_master_darks(dark_list):
    """
    Build a single master dark from the frames in `dark_list` (all rows
    are assumed to belong to the same cluster) and write it to disk.

    Requires at least 3 frames to build a master; clusters smaller than
    that are skipped (returning None) rather than recorded, so they can
    be retried later if more frames for the same cluster arrive.

    Parameters
    ----------
    dark_list : pandas.DataFrame
        Rows (all one cluster) from the clustered dark frame catalog, as
        produced by `_clustering_bc_darks`.

    Returns
    -------
    pandas.DataFrame or None
        Single-row DataFrame describing the new master dark catalog
        entry, or None if too few frames were available or the master
        failed to build.
    """
    files = dark_list['filename'].values
    master_name = dark_list['master_name'].values[0]
    readout_mode = dark_list['readout'].values[0]

    entry = {}

    entry['name'] = master_name
    entry['telescope'] = 'B&C'
    entry['exptime'] = np.nanmean(dark_list['exptime'].astype(float))
    entry['readout'] = str(readout_mode)
    entry['jd'] = np.nanmean(dark_list['jd'].astype(float))
    entry['date'] = dark_list['date'].iloc[0]
    entry['nimages'] = len(files)
    entry['shape'] = int(dark_list['shape'].iloc[0])

    if len(files) < 3:
        logger.info(f'{master_name}: too few frames ({len(files)}) to build a master dark; skipping (will retry later if more arrive)')
        return None

    try:
        master, std, header = build_master_dark_sigma_clip(files)
    except Exception as e:
        logger.error(f'{master_name}: failed to build master dark: {e}')
        return None

    header['MASTER'] = True
    header['SIGCLIP'] = True

    phdu = fits.PrimaryHDU(master, header)
    ehdu = fits.ImageHDU(std)

    hdul = fits.HDUList([phdu, ehdu])

    save_name = (Path(MASTER_DARK_LOCATION) / master_name).with_suffix('.fits')

    hdul.writeto(save_name, overwrite=True)
    os.system(f"gzip -f {save_name}")

    entry['filename'] = str(save_name.with_suffix('.fits.gz'))
    entry['median'] = float(np.nanmedian(master))
    entry['note'] = 'good'

    return pd.DataFrame([entry])

def dark_processing(dark_list, cluster):
    """
    Build a master dark for a single cluster ID within `dark_list`.
    Thin wrapper around `_bc_master_darks` used as the per-cluster unit
    of work in `make_master_darks`'s (optionally parallel) loop.
    """
    sub_dark_list = dark_list[dark_list['cluster'] == cluster].copy()
    return _bc_master_darks(sub_dark_list)

def get_master_dark(jd, exptime, readout, exp_tol=1, date_tol=3, shape=2048):
    """
    Find the closest-in-time suitable master dark for a given science
    frame's JD, exposure time, readout mode, and detector shape.

    Parameters
    ----------
    jd : float
        Julian date of the science frame.
    exptime : float
        Exposure time of the science frame (seconds).
    readout : str or numeric
        Readout mode of the science frame (compared as a string).
    exp_tol : float
        Maximum allowed difference in exposure time between the science
        frame and a candidate master dark.
    date_tol : float
        Maximum allowed time difference (days) between the science
        frame and a candidate master dark's mean JD.
    shape : int
        Required detector shape (pixels per side).

    Returns
    -------
    (filename, t_diff) : (str, float)
        Path to the best-matching master dark and its time offset (days)
        from the science frame, or `('none', -999)` if no suitable master
        dark is found.
    """
    try:
        darks = pd.read_csv(CAL_LIST_LOCATION + 'bc_master_dark_list.csv')
    except Exception:
        return 'none', -999

    darks = darks[darks['filename'] != 'IGNORED'].copy()
    if len(darks) == 0:
        return 'none', -999

    dreadout = darks['readout'].values.astype(str)
    dexptime = darks['exptime'].values.astype(float)

    readout_ind = dreadout == str(readout)
    exp_ind = np.abs(dexptime - float(exptime)) <= exp_tol

    dshape = darks['shape'].values
    shape_ind = dshape.astype(int) == int(shape)

    mask = exp_ind & readout_ind & shape_ind
    good = darks.loc[mask]

    if len(good) == 0:
        return 'none', -999

    djd = good['jd'].values.astype(float)
    jd = float(jd)
    diff = (jd - djd).astype(float)
    min_ind = np.argmin(np.abs(diff))
    t_diff = diff[min_ind]

    if abs(t_diff) >= date_tol:
        return 'none', -999

    dark = good.iloc[min_ind]
    fname = dark['filename']
    return fname, t_diff

def build_master_dark_sigma_clip(files, sigma=5):
    """
    Combine a list of dark frames into a single master dark via
    two-pass sigma-clipped stacking.

    First pass computes a per-pixel running mean/std across all frames
    (via `RunningStats`, using Welford's algorithm for numerical
    stability). Second pass re-accumulates the stack, excluding any
    pixel in any frame that deviates from the first-pass mean by more
    than `sigma` standard deviations, giving the final, outlier-rejected
    master.

    Parameters
    ----------
    files : list of str
        Paths to the individual dark frames to combine.
    sigma : float
        Clipping threshold (in standard deviations) for the second pass.

    Returns
    -------
    final_mean, final_std : 2D ndarrays (float32)
        The master dark and its per-pixel standard deviation.
    header : astropy.io.fits.Header
        Header copied from the first input file, for use as a starting
        point for the output file's header.
    """
    with fits.open(files[0], memmap=False) as hdul:
        shape = hdul[0].data.shape
        header = hdul[0].header.copy()

    stats = RunningStats(shape)

    for f in files:
        with fits.open(f, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float32, copy=False)
            stats.add(data)

    mean = stats.mean()
    std = stats.std()

    stats2 = RunningStats(shape)
    threshold = sigma * std

    for f in files:
        with fits.open(f, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float32, copy=False)
            mask = np.abs(data - mean) < threshold
            stats2.add(data, mask)

    final_mean = stats2.mean()
    final_std = stats2.std()

    return final_mean.astype(np.float32), final_std.astype(np.float32), header