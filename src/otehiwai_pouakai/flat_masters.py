"""
Master flat frame construction for the B&C reduction pipeline.

Flat frames are grouped into calibration blocks per (readout mode,
band), using agglomerative clustering with complete linkage and a
Chebyshev metric to strictly bound both the time span and exposure-time
span of any single block (see `_clustering_bc_flats`). Each block is
matched to a suitable master dark (via `dark_masters.get_master_dark`)
and combined into a single master flat via sigma-clipped stacking of
dark-subtracted, median-normalized frames (see
`build_master_flat_sigma_clip`). `make_master_flats` is the main entry
point: it finds blocks not yet represented in the master flat catalog,
builds masters for them, and updates the catalog CSV. `get_master_flat`
looks up the best-matching existing master flat for a given science
frame's observing conditions.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from glob import glob
import os
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
from pathlib import Path
import subprocess
import re
from collections import defaultdict

from astropy.io import fits
from astropy.time import Time
from astropy.stats import SigmaClip

from sklearn.cluster import DBSCAN, AgglomerativeClustering

from .dark_masters import get_master_dark, normalize_readout, _make_clustering
from .running_stats import RunningStats

import logging

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

from . import config

# See dark_masters.py / config.py -- same POUAKAI_* environment
# variables control where these live.
MASTER_FLAT_LOCATION = config.master_flat_dir()
CAL_LIST_LOCATION = config.cal_list_dir()


def assign_time_blocks(df, delta_t):
    """
    Assign a running block ID to rows of `df` (assumed already sorted or
    to be sorted by Julian date) such that a new block starts whenever
    the gap since the CURRENT block's first frame exceeds `delta_t`.

    Not currently used by `_clustering_bc_flats` (which uses
    agglomerative clustering instead), but kept as a simpler, sequential
    alternative grouping strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a `jd` (Julian date) column.
    delta_t : float
        Maximum time span (days), measured from each block's first
        frame, before starting a new block.

    Returns
    -------
    block_ids : 1D ndarray of int
        Block ID for each row of `df` (after sorting by `jd`).
    """
    df = df.sort_values('jd').reset_index(drop=True)

    block_ids = np.zeros(len(df), dtype=int)

    block = 0
    start_idx = 0

    for i in range(1, len(df)):
        if df.loc[i, 'jd'] - df.loc[start_idx, 'jd'] > delta_t:
            block += 1
            start_idx = i

        block_ids[i] = block

    return block_ids


def _filter_bc_flats(exp_tol=1, flat_delta_t=30, dark_delta_t=1):
    """
    Load the catalog of all recorded flat frames, restrict to usable
    frames (correct detector shape, sensible count-level range, not
    previously flagged bad), and group them into calibration blocks.

    Parameters
    ----------
    exp_tol : float
        Passed to `_clustering_bc_flats` as `exp_tolerance`.
    flat_delta_t : float
        Passed to `_clustering_bc_flats` as `delta_t`.
    dark_delta_t : float
        Passed to `_clustering_bc_flats` as `dark_delta_t`, for matching
        each flat frame to a master dark.

    Returns
    -------
    pandas.DataFrame
        Filtered, clustered flat frame catalog with `cluster` and
        `master_name` columns added (see `_clustering_bc_flats`).
    """
    initial_df = pd.read_csv(CAL_LIST_LOCATION + 'bc_flat_image_list.csv')

    shape_mask = initial_df['shape'].values.astype(int) == 2048
    upper_mask = initial_df['median'].astype(float) >= 10000
    lower_mask = initial_df['median'].astype(float) <= 48000
    bad_mask = initial_df['telescope'].values.astype(str) != 'bad'

    chip_filtered_df = initial_df[shape_mask & upper_mask & lower_mask & bad_mask].copy()

    chip_filtered_df = _clustering_bc_flats(chip_filtered_df, delta_t=flat_delta_t, exp_tolerance=exp_tol, dark_delta_t=dark_delta_t)

    return chip_filtered_df


def _clustering_bc_flats(chip_filtered_df, delta_t=30, exp_tolerance=1.0, dark_delta_t=1):
    """
    Group flat frames into calibration blocks, separately for each
    (readout mode, band) pair, using agglomerative clustering with
    complete linkage and a Chebyshev metric.

    Complete linkage plus a Chebyshev (max-norm) distance means each
    resulting block has a strictly bounded full diameter in BOTH scaled
    dimensions at once: `max JD span <= delta_t` and
    `max exposure-time span <= exp_tolerance` across every pair of frames
    in the block, not just between neighbours -- a tighter guarantee than
    single/average linkage would give.

    Each candidate block is required to have at least 3 frames both
    before and after matching to a master dark (via `get_master_dark`);
    blocks failing either check are dropped. Surviving blocks are
    assigned a `master_name` based on mean UTC date, band, readout mode,
    and detector shape, with a `_v2`, `_v3`, ... suffix if a name would
    otherwise collide within this call.

    Parameters
    ----------
    chip_filtered_df : pandas.DataFrame
        Flat frame catalog, already filtered to usable frames, with at
        least `readout`, `band`, `jd`, `exptime`, and `shape` columns.
    delta_t : float
        Clustering distance threshold, in Julian date (days), after
        scaling.
    exp_tolerance : float
        Clustering distance threshold, in exposure time (seconds), after
        scaling.
    dark_delta_t : float
        Passed to `get_master_dark` as `date_tol` when matching each
        frame to a master dark.

    Returns
    -------
    pandas.DataFrame
        `chip_filtered_df` with `cluster`, `master_name`,
        `master_dark`, `master_dark_t_diff`, `t_start`, and `t_end`
        columns added, restricted to frames in blocks that survived
        filtering. Empty DataFrame if no blocks were found.
    """
    result = []
    block_offset = 0
    name_counter = defaultdict(int)

    for (readout, filt), df_ro in chip_filtered_df.groupby(['readout', 'band']):
        df_ro = df_ro.sort_values('jd').reset_index(drop=True)

        if len(df_ro) < 3:
            continue

        X = df_ro[['jd', 'exptime']].astype(float).values

        X_scaled = np.copy(X)
        X_scaled[:, 0] /= delta_t
        X_scaled[:, 1] /= exp_tolerance

        clustering = _make_clustering(
            AgglomerativeClustering, n_clusters=None, distance_threshold=1.0,
            linkage="complete", affinity="chebyshev",
        )

        labels = clustering.fit_predict(X_scaled)
        df_ro['block'] = labels

        for block_id in sorted(set(labels)):
            sub = df_ro[df_ro['block'] == block_id].copy()

            if len(sub) < 3:
                continue

            t_start = sub['jd'].min()
            t_end = sub['jd'].max()
            mean_jd = 0.5 * (t_start + t_end)

            jd_utc = Time(mean_jd, format='jd').utc.strftime('%Y%m%d')
            median_exptime = sub['exptime'].median()

            mdark = np.empty(len(sub), dtype=object)
            mdiff = np.empty(len(sub), dtype=float)

            for j in range(len(sub)):
                mdark[j], mdiff[j] = get_master_dark(
                    sub.iloc[j]['jd'],
                    sub.iloc[j]['exptime'],
                    readout,
                    exp_tol=exp_tolerance,
                    date_tol=dark_delta_t,
                    shape=int(sub.iloc[j]['shape']),
                )

            sub['master_dark'] = mdark
            sub['master_dark_t_diff'] = mdiff

            sub = sub[sub['master_dark'] != 'none'].copy()

            if len(sub) < 3:
                continue

            readout_str = normalize_readout(readout)
            shape_val = int(sub['shape'].iloc[0])

            base_name = (f"master_flat_{jd_utc}_{filt}_{readout_str}_{shape_val}")

            name_counter[base_name] += 1
            master_name = (base_name if name_counter[base_name] == 1 else f"{base_name}_v{name_counter[base_name]}")

            sub['cluster'] = block_offset
            sub['master_name'] = master_name
            sub['t_start'] = t_start
            sub['t_end'] = t_end

            result.append(sub)
            block_offset += 1

    if len(result) == 0:
        return pd.DataFrame()

    return pd.concat(result, ignore_index=True)


def make_master_flats(exp_tol=1, flat_delta_t=30, dark_delta_t=1, num_cores=1):
    """
    Build any master flat frames not yet present in the master flat
    catalog, and update the catalog CSV.

    Compares the set of calibration blocks found in the current flat
    frame list against the master names already recorded in
    `bc_master_flat_list.csv`, builds masters only for the new ones, and
    appends them to the catalog.

    Parameters
    ----------
    exp_tol : float
        Passed to `_filter_bc_flats` as `exp_tol`.
    flat_delta_t : float
        Passed to `_filter_bc_flats` as `flat_delta_t`.
    dark_delta_t : float
        Passed to `_filter_bc_flats` as `dark_delta_t`.
    num_cores : int
        If greater than 1, build masters for different blocks in
        parallel threads.
    """
    flat_list = _filter_bc_flats(exp_tol=exp_tol, flat_delta_t=flat_delta_t, dark_delta_t=dark_delta_t)
    try:
        masters = pd.read_csv(CAL_LIST_LOCATION + 'bc_master_flat_list.csv')
    except Exception:
        masters = pd.DataFrame(columns=['name', 'telescope', 'exptime', 'jd', 'date', 'band',
                                         'readout', 'filename', 'nimages', 'shape', 'median'])

    all_names = set(flat_list['master_name'].values) if len(flat_list) else set()
    master_names = set(masters['name'].values) if len(masters) else set()

    flat_list = flat_list.reset_index(drop=True)
    flat_list.to_csv(CAL_LIST_LOCATION + 'temp_bc_flat_list_filtered.csv', index=False)

    new = all_names - master_names
    new = list(new)
    new.sort(reverse=True)
    print('Number of new master flat entries:', len(new))

    updated_flat_list = flat_list[flat_list['master_name'].isin(new)].copy()

    if len(new) > 0:
        if num_cores > 1:
            entries = Parallel(n_jobs=num_cores, backend="threading", prefer="threads")(
                delayed(flat_processing)(updated_flat_list, cluster)
                for cluster in tqdm(updated_flat_list['cluster'].unique(), desc='Processing files')
            )
            for entry in entries:
                if entry is not None:
                    masters = pd.concat([masters, entry], ignore_index=True)
        else:
            for cluster in tqdm(updated_flat_list['cluster'].unique(), desc='Processing files'):
                entry = flat_processing(updated_flat_list, cluster)
                if entry is not None:
                    masters = pd.concat([masters, entry], ignore_index=True)

    masters = masters.reset_index(drop=True)
    print('Done creating master flats, total masters:', len(masters))

    masters.to_csv(CAL_LIST_LOCATION + 'bc_master_flat_list.csv', index=False)


def _bc_master_flats(flat_list):
    """
    Build a single master flat from the frames in `flat_list` (all rows
    are assumed to belong to the same cluster/block, each already
    matched to a master dark) and write it to disk.

    Requires at least 3 frames to build a master; smaller blocks are
    skipped (returning None).

    Parameters
    ----------
    flat_list : pandas.DataFrame
        Rows (all one cluster) from the clustered flat frame catalog, as
        produced by `_clustering_bc_flats`, including a `master_dark`
        column giving each frame's matched master dark file.

    Returns
    -------
    pandas.DataFrame or None
        Single-row DataFrame describing the new master flat catalog
        entry, or None if too few frames were available or the master
        failed to build.
    """
    files = flat_list['filename'].values
    dark_files = flat_list['master_dark'].values

    master_name = flat_list['master_name'].values[0]
    readout_mode = flat_list['readout'].values[0]
    band = flat_list['band'].values[0]

    entry = {}

    if len(files) < 3:
        return None

    try:
        master, std, header = build_master_flat_sigma_clip(files, dark_files, sigma=5)

        time_jd = np.nanmean(flat_list['jd'].astype(float))

        header['MASTER'] = True
        header['JDSTART'] = time_jd
        header['SIGCLIP'] = True
        header['BAND'] = band

        phdu = fits.PrimaryHDU(master, header)
        ehdu = fits.ImageHDU(std)

        hdul = fits.HDUList([phdu, ehdu])

        save_name = Path(MASTER_FLAT_LOCATION) / master_name
        save_name = save_name.with_suffix('.fits')

        hdul.writeto(save_name, overwrite=True)

        os.system(f"gzip -f {save_name}")

        entry['name'] = master_name
        entry['telescope'] = 'B&C'
        entry['exptime'] = np.nanmean(flat_list['exptime'].astype(float))
        entry['readout'] = str(readout_mode)
        entry['jd'] = time_jd
        entry['band'] = band
        entry['date'] = header.get('DATE-OBS', '')
        entry['nimages'] = len(files)
        entry['filename'] = str(save_name.with_suffix('.fits.gz'))
        entry['shape'] = master.shape[0]
        entry['median'] = float(np.nanmedian(master))

        return pd.DataFrame([entry])

    except Exception as e:
        logger.error(f'{master_name}: flat creation failed: {e}')
        return None


def flat_processing(flat_list, cluster):
    """
    Build a master flat for a single cluster ID within `flat_list`. Thin
    wrapper around `_bc_master_flats` used as the per-block unit of work
    in `make_master_flats`'s (optionally parallel) loop.
    """
    sub_flat_list = flat_list[flat_list['cluster'] == cluster].copy()
    return _bc_master_flats(sub_flat_list)


def get_master_flat(jd, readout, band, date_tol=30, shape=2048):
    """
    Find the closest-in-time suitable master flat for a given readout
    mode, band, and detector shape.

    Parameters
    ----------
    jd : float
        Julian date of the science frame.
    readout : str or numeric
        Readout mode of the science frame (compared as a string).
    band : str
        Filter/band of the science frame (compared as a string).
    date_tol : float
        Maximum allowed time difference (days) between the science
        frame and a candidate master flat's mean JD.
    shape : int
        Required detector shape (pixels per side).

    Returns
    -------
    (filename, t_diff) : (str, float)
        Path to the best-matching master flat and its time offset (days)
        from the science frame, or `('none', -999)` if no suitable master
        flat is found.

    Notes
    -----
    Deliberately does not filter on exposure time: flats are median-
    normalized before combination, so their individual exposure times
    don't need to match the science frame's.
    """
    try:
        flats = pd.read_csv(CAL_LIST_LOCATION + 'bc_master_flat_list.csv')
    except Exception:
        return 'none', -999

    if len(flats) == 0:
        return 'none', -999

    dreadout = flats['readout'].values.astype(str)
    dband = flats['band'].values.astype(str)
    dshape = flats['shape'].values.astype(int)

    readout_ind = dreadout == str(readout)
    band_ind = dband == str(band)
    shape_ind = dshape == int(shape)

    mask = readout_ind & shape_ind & band_ind
    good = flats.loc[mask]

    if len(good) == 0:
        return 'none', -999

    djd = good['jd'].values.astype(float)
    jd = float(jd)
    diff = (jd - djd).astype(float)
    min_ind = np.argmin(np.abs(diff))
    t_diff = diff[min_ind]

    if abs(t_diff) >= date_tol:
        return 'none', -999

    flat = good.iloc[min_ind]
    fname = flat['filename']
    return fname, t_diff


def build_master_flat_sigma_clip(files, dark_files, sigma=5):
    """
    Combine a list of flat frames into a single master flat via
    two-pass sigma-clipped stacking of dark-subtracted, median-
    normalized frames.

    Each flat frame is first dark-subtracted using its matched master
    dark, then divided by its own median (so frames taken under
    different illumination levels combine on a common scale). The
    combination itself follows the same two-pass sigma-clipping approach
    as `dark_masters.build_master_dark_sigma_clip`: a first pass computes
    a per-pixel running mean/std (via `RunningStats`), and a second pass
    re-accumulates the stack excluding outlier pixels beyond `sigma`
    standard deviations from the first-pass mean. The final result is
    renormalized by its own median, so the master flat itself has a
    median of 1.

    Frames whose dark-subtracted median is non-positive or non-finite
    (e.g. a bad or saturated frame) are skipped entirely.

    Parameters
    ----------
    files : list of str
        Paths to the individual flat frames to combine.
    dark_files : list of str
        Paths to each flat frame's matched master dark (same length and
        order as `files`).
    sigma : float
        Clipping threshold (in standard deviations) for the second pass.

    Returns
    -------
    final_mean, final_std : 2D ndarrays (float32)
        The (median-normalized) master flat and its per-pixel standard
        deviation.
    header : astropy.io.fits.Header
        Header copied from the first input flat file, for use as a
        starting point for the output file's header.
    """
    with fits.open(files[0], memmap=False) as hdul:
        shape = hdul[0].data.shape
        header = hdul[0].header.copy()

    stats = RunningStats(shape)

    for flat_file, dark_file in zip(files, dark_files):
        with fits.open(dark_file, memmap=False) as hdul:
            dark = hdul[0].data.astype(np.float32, copy=False)

        with fits.open(flat_file, memmap=False) as hdul:
            flat = hdul[0].data.astype(np.float32, copy=False)

        data = flat - dark
        norm = np.nanmedian(data)

        if norm <= 0 or not np.isfinite(norm):
            continue

        data /= norm
        stats.add(data)

    mean = stats.mean()
    std = stats.std()

    stats2 = RunningStats(shape)
    threshold = sigma * std

    for flat_file, dark_file in zip(files, dark_files):
        with fits.open(dark_file, memmap=False) as hdul:
            dark = hdul[0].data.astype(np.float32, copy=False)

        with fits.open(flat_file, memmap=False) as hdul:
            flat = hdul[0].data.astype(np.float32, copy=False)

        data = flat - dark
        norm = np.nanmedian(data)

        if norm <= 0 or not np.isfinite(norm):
            continue

        data /= norm

        mask = np.abs(data - mean) < threshold
        stats2.add(data, mask)

    final_mean = stats2.mean()
    final_std = stats2.std()

    final_mean /= np.nanmedian(final_mean)

    return final_mean.astype(np.float32), final_std.astype(np.float32), header