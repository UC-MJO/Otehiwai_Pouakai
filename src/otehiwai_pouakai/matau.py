"""
Small filesystem/dataframe utilities shared across the reduction
pipeline: WCS-solving cleanup, output directory setup, science-frame
file listing, and building the per-object/night/band master-name
dataframe used to group science frames for later stacking or lookup.
"""

import numpy as np
import pandas as pd

from astropy.time import Time

import os
from glob import glob
from pathlib import Path
import logging

from . import config

logger = logging.getLogger(__name__)

def _deleting_wcs(save_location):
    """
    Remove astrometry.net's intermediate working files (.xyls, .axy,
    .corr, .match, .rdls, .solved, .wcs) from the `wcs/` output
    directory, once they are no longer needed.

    Uses Python's own `glob` + `os.remove` rather than shelling out to
    `rm`, so a missing file pattern doesn't print shell noise, and any
    individual removal failure is logged rather than silently lost
    inside a single shell command.

    Parameters
    ----------
    save_location : str
        Pipeline output root; intermediate files are looked for under
        `save_location + 'wcs/'`.
    """
    wcs_dir = save_location + 'wcs/'
    extensions = ['xyls', 'axy', 'corr', 'match', 'rdls', 'solved', 'wcs']

    removed = 0
    for ext in extensions:
        for f in glob(wcs_dir + f'*.{ext}'):
            try:
                os.remove(f)
                removed += 1
            except OSError as e:
                logger.warning(f'Failed to remove {f}: {e}')

    logger.info(f'Removed {removed} astrometry.net intermediate files from {wcs_dir}')

def rename_wcs(filename_or_result):
    """
    Rename a successfully solved `*_wcs.new` file back to its standard
    name and gzip it in place.

    Parameters
    ----------
    filename_or_result : str or dict
        Either a bare filename (legacy calling convention), or the dict
        returned by `wcs_compute.wcs_astrometrynet_local` (preferred).
        When given the dict, this only attempts the rename if
        `result['success']` is True, so frames that never solved are
        skipped rather than causing a failed rename attempt.
    """
    if isinstance(filename_or_result, dict):
        if not filename_or_result.get('success'):
            return
        filename = filename_or_result.get('new_file')
        if filename is None:
            return
    else:
        filename = filename_or_result

    try:
        old_path = Path(filename)

        new_name = old_path.name.replace('_reduced', '')
        new_path = old_path.with_name(new_name).with_suffix('.fits')

        old_path.rename(new_path)
        os.system(f"gzip -f {new_path}")

    except Exception as e:
        logger.warning(f'{filename}: failed to rename/gzip solved WCS file: {e}')

def _file_creation(save_location):
    """
    Create the pipeline's standard output directory structure under
    `save_location` if it doesn't already exist: `red/`, `cal/`, `wcs/`,
    `fig/`, `zp/`, `phot_table/`.
    """
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    for path in ['red/', 'cal/', 'wcs/', 'fig/', 'zp/', 'phot_table/']:
        if not os.path.exists(save_location + path):
            os.makedirs(save_location + path)

def master_name_from_path(path, strip_suffixes=('_reduced',)):
    """
    Recover the `master_name` grouping key (see `update_df`) from an
    output filename produced at any pipeline stage.

    Output files carry the master_name as their stem throughout the
    pipeline, plus a stage-specific marker that `rename_wcs`/
    `calibrating_internal` only ever partially strip: `red/` files are
    `<master_name>_reduced.fits.gz`; `rename_wcs` strips `_reduced` but
    leaves the `_wcs` marker it introduces, so `wcs/` files are
    `<master_name>_wcs.fits.gz`; `calibrating_internal` in turn replaces
    `_wcs` with `_cal`, so `cal/` files are `<master_name>_cal.fits.gz`.
    Pass the right `strip_suffixes` for whichever directory `path` came
    from -- this lets any stage recover which requested frame a given
    output file on disk corresponds to, regardless of which
    stage-specific suffix/extension it currently has.

    Parameters
    ----------
    path : str or Path
        Path to a `red/`, `wcs/`, or `cal/` output file.
    strip_suffixes : tuple of str
        Suffixes to strip from the stem after removing known
        extensions, if present. Default handles `red/`'s `_reduced`
        marker; use `('_wcs',)` for `wcs/` files, `('_cal',)` for
        `cal/` files, or `()` if the stage's filenames need no
        stripping.

    Returns
    -------
    str
        The recovered master_name.
    """
    stem = Path(path).name
    for ext in ('.fits.gz', '.fits'):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    for suffix in strip_suffixes:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem


def filter_by_master_names(paths, master_names, strip_suffixes=('_reduced',)):
    """
    Restrict `paths` (output files found on disk, e.g. via `glob`) to
    only those whose recovered master_name (see
    `master_name_from_path`) is in `master_names`.

    This is what scopes a stage to the frames actually requested for
    the current run (via `--files`/`--glob`), instead of a stage
    silently picking up every other file that happens to already sit
    in the same `save_location` subdirectory -- which is otherwise easy
    to hit when re-running `--mode wcs`/`cal` on a single frame inside
    a `save_location` shared with other, previously-processed frames.

    Parameters
    ----------
    paths : list of str
        Candidate file paths (typically from `glob`).
    master_names : set of str or None
        master_names for the current run's requested frames. If None,
        `paths` is returned unchanged (no scoping) -- used to preserve
        old behaviour where this isn't wired up, or when a caller
        deliberately wants every file in the directory.
    strip_suffixes : tuple of str
        Passed through to `master_name_from_path`.

    Returns
    -------
    list of str
        The subset of `paths` matching `master_names`, in original order.
    """
    if master_names is None:
        return paths
    return [p for p in paths if master_name_from_path(p, strip_suffixes) in master_names]


def get_file_paths(file_path):
    """
    Glob `file_path` and return the absolute paths of matching files,
    excluding any whose filename suggests it's a dark/flat/bias frame.

    This is a secondary, filename-based safety net -- the primary
    dark/flat/bias/science classification happens via the FITS
    `IMAGETYP` header keyword in `organise_files.py`; this just catches
    anything that might slip through by an obviously-named file.

    Parameters
    ----------
    file_path : str
        Glob pattern to search.

    Returns
    -------
    file_path_list : list of str
        Matching file paths, excluding likely calibration frames.
    """
    file_path_list = []
    for filename in glob(file_path):
        lower = str(filename).lower()
        if 'dark' in lower or 'flat' in lower or 'bias' in lower:
            continue
        if os.path.isfile(filename):
            file_path_list.append(str(filename))
    return file_path_list

def update_df(files):
    """
    Build a per-frame dataframe for the given science `files`, adding a
    UTC observation date and a `master_name` grouping key
    (`<object>_<date>_<band>_<running number>`) used to identify which
    frames belong together for later stages (e.g. stacking, lookup by
    name).

    Frames are grouped and numbered per (object, UTC date, band), sorted
    by Julian date within each group, so `running_number` reflects
    observation order within that night/band for that object.

    Parameters
    ----------
    files : list of str
        Filenames (matching the `filename` column of the master science
        image list) to include in the returned dataframe.

    Returns
    -------
    updated_sci_list : pandas.DataFrame
        Rows of the master science image list matching `files`, with
        `jd_utc`, `running_number`, and `master_name` columns added.
    """
    all_sci_df = pd.read_csv(config.cal_list_dir() + 'bc_science_image_list.csv')

    df = all_sci_df.copy()

    df['jd_utc'] = Time(df['jd'].values, format='jd').utc.strftime('%Y%m%d')

    df = df.sort_values(['object', 'jd_utc', 'band', 'jd']).reset_index(drop=True)
    df['running_number'] = (df.groupby(['object', 'jd_utc', 'band']).cumcount().add(1).astype(str).str.zfill(4))
    df['master_name'] = (df['object'].astype(str) + '_' + df['jd_utc'] + '_' + df['band'].astype(str) + '_' + df['running_number'])

    updated_sci_list = df[df['filename'].isin(files)].copy()
    updated_sci_list = updated_sci_list.reset_index(drop=True)

    return updated_sci_list