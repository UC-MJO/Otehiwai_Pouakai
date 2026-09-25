"""
Cross-stage file lineage tracking, independent of folder location or
naming convention.

Why this exists
----------------
Once reduction/wcs/calibration stages can each be pointed at an
arbitrary input folder or file list (see pipeline.py's
--wcs-input-dir/--cal-input-dir/--cal-input-files), you can no longer
answer "what became of raw frame X" just by checking whether a
same-named file happens to sit in the conventional red/wcs/cal
sub-folder -- it might not be there at all, or several different runs
against different folders might have touched it. This module gives
every stage a common, filename-derived `source_id` (stripping only the
pipeline's OWN stage suffixes, not any convention the raw data has to
follow) and appends one manifest row per (source_id, stage) outcome, so
a frame's full history can be reconstructed regardless of where it was
actually processed from.

This is deliberately append-only (unlike failure_ledger.py, which keeps
only the latest outcome per (stage, file) -- see that module's
docstring) so a full run history survives repeated re-runs; use
`pivot_status` for a "latest outcome per stage" summary view.
"""

import os
import time
import logging
from pathlib import Path

import pandas as pd

from .failure_ledger import _LedgerLock

logger = logging.getLogger(__name__)

_COLUMNS = ['source_id', 'stage', 'status', 'input_path', 'output_path', 'reason', 'timestamp']

# Only the suffixes THIS pipeline appends at each stage (see
# core_reduction.py / wcs_compute.py). Deliberately does NOT try to
# strip or normalise anything about the raw filename itself -- the
# whole point is that stage independence should not require the raw
# data to follow any particular naming convention.
_STAGE_SUFFIXES = ['_reduced', '_wcs', '_cal', '_phottable', '_zpsurface']
_STAGE_EXTENSIONS = ('.fits.gz', '.fits', '.csv', '.npy')


def stable_id(path):
    """
    Derive a stage-independent identifier for `path`: strip a known
    pipeline output extension, then at most one known pipeline stage
    suffix, leaving whatever the raw filename actually was. The same
    source frame therefore maps to the same id at every stage,
    regardless of which directory it currently lives in.
    """
    name = Path(str(path)).name
    for ext in _STAGE_EXTENSIONS:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    for suffix in _STAGE_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _manifest_path(save_location):
    log_dir = os.path.join(save_location, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, 'manifest.csv')


def record_stage(save_location, stage, status, input_path=None, output_path=None,
                  reason='', source_id=None):
    """
    Append one manifest row for a single file's outcome at `stage`.

    Parameters
    ----------
    stage : str
        e.g. 'reduction', 'wcs', 'calibration'.
    status : str
        e.g. 'success', 'failed', 'skipped'.
    source_id : str or None
        If not given, derived via `stable_id` from `input_path` (or
        `output_path` if `input_path` isn't available).
    """
    path = _manifest_path(save_location)
    sid = source_id or stable_id(input_path or output_path)

    row = pd.DataFrame([{
        'source_id': sid, 'stage': stage, 'status': status,
        'input_path': str(input_path) if input_path else '',
        'output_path': str(output_path) if output_path else '',
        'reason': str(reason), 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }])

    with _LedgerLock(path):
        header_needed = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a') as f:
            row.to_csv(f, index=False, header=header_needed)

    return sid


def history(save_location, source_id):
    """Return every recorded row for a given `source_id`, oldest first."""
    df = full_manifest(save_location)
    if len(df) == 0:
        return df
    return df[df['source_id'] == source_id].sort_values('timestamp')


def full_manifest(save_location):
    path = _manifest_path(save_location)
    if not os.path.exists(path):
        return pd.DataFrame(columns=_COLUMNS)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_COLUMNS)


def pivot_status(save_location):
    """
    Wide summary: one row per source_id, one column per stage holding
    its MOST RECENT status. The quickest way to answer "which frames
    made it all the way through calibration" or "which frames died at
    wcs" for a whole run, regardless of which folders were involved.
    """
    df = full_manifest(save_location)
    if len(df) == 0:
        return df
    latest = df.sort_values('timestamp').groupby(['source_id', 'stage'], as_index=False).tail(1)
    return latest.pivot(index='source_id', columns='stage', values='status').reset_index()