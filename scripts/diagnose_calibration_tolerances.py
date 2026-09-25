"""
Diagnostic: empirically determine appropriate dark_date_tol / flat_date_tol
(and exp_tol) values from your actual archive, rather than guessing.

What this does
---------------
For every science frame matching the given glob, finds the closest-in-time
available master dark and master flat (matching on readout/shape/band as
appropriate) WITHOUT applying any date or exposure tolerance cutoff -- i.e.
this bypasses get_master_dark/get_master_flat's tolerance check entirely
and just reports "how far away is the nearest one, regardless of whether
it would currently pass."

This directly answers the tolerance question: if most failing frames have
a nearest master just outside your current tolerance (e.g. nearest dark
is 15 days away and dark_date_tol=12), widening the tolerance will
recover them. If gaps are huge (weeks to months) or scattered with no
clustering near the current tolerance, no tolerance value fixes that --
it means calibration frames are genuinely missing for that period, and
the fix is operational (take more frequent darks/flats), not a config
change.

Usage
-----
    python3 diagnose_calibration_tolerances.py

Adjust TEST_NIGHT_GLOB below, or import and call
`diagnose(glob_pattern)` directly for other nights/date ranges.
"""

import numpy as np
import pandas as pd
import logging

from otehiwai_pouakai.matau import get_file_paths
from otehiwai_pouakai import config

logger = logging.getLogger(__name__)

# Follows POUAKAI_CAL_LIST_DIR (see config.py); override that env var
# rather than editing this file for a different site.
cal_list_location = config.cal_list_dir()

# Site- and run-specific -- adjust this to the night/date range you want
# to diagnose, or call diagnose(glob_pattern) directly with your own.
TEST_NIGHT_GLOB = '/home/phys/astro8/MJArchive/octans/20250914/*.fit'


def _closest_master_no_tolerance(jd, masters_df, time_col='jd'):
    """
    Return (filename, abs_time_diff_days, exptime_diff) for the
    closest-in-time master in masters_df, with NO tolerance applied. If
    masters_df is empty, returns (None, np.inf, np.nan).
    """
    if len(masters_df) == 0:
        return None, np.inf, np.nan

    djd = masters_df[time_col].values.astype(float)
    diff = np.abs(jd - djd)
    min_ind = np.argmin(diff)
    row = masters_df.iloc[min_ind]
    return row.get('filename', None), diff[min_ind], row.get('exptime', np.nan)


def diagnose(glob_pattern=TEST_NIGHT_GLOB, science_csv=None):
    """
    Returns a pandas DataFrame, one row per science frame, with columns:
        filename, jd, exptime, readout, band,
        dark_gap_days, dark_exp_diff, dark_candidate,
        flat_gap_days, flat_candidate

    `dark_gap_days`/`flat_gap_days` are the actual time gap to the
    nearest master, regardless of whether it would pass any tolerance.
    """
    science_csv = science_csv or (cal_list_location + 'bc_science_image_list.csv')

    all_sci = pd.read_csv(science_csv)

    test_files = set(get_file_paths(glob_pattern))
    sci = all_sci[all_sci['filename'].isin(test_files)].copy()

    if len(sci) == 0:
        logger.warning(f'No science frames from {science_csv} matched {glob_pattern}')
        return pd.DataFrame()

    try:
        darks = pd.read_csv(cal_list_location + 'bc_master_dark_list.csv')
        darks = darks[darks['filename'] != 'IGNORED'].copy()
    except Exception:
        darks = pd.DataFrame(columns=['filename', 'jd', 'exptime', 'readout', 'shape'])

    try:
        flats = pd.read_csv(cal_list_location + 'bc_master_flat_list.csv')
    except Exception:
        flats = pd.DataFrame(columns=['filename', 'jd', 'band', 'readout', 'shape'])

    rows = []
    for _, sci_row in sci.iterrows():
        jd = float(sci_row['jd'])
        readout = str(sci_row['readout'])
        band = str(sci_row.get('band', ''))
        shape = int(sci_row.get('shape', 2048))

        dark_candidates = darks[(darks['readout'].astype(str) == readout) &
                                (darks['shape'].astype(int) == shape)] if len(darks) else darks
        dark_fname, dark_gap, dark_exp_diff_raw = _closest_master_no_tolerance(jd, dark_candidates)
        dark_exp_diff = (abs(dark_exp_diff_raw - float(sci_row['exptime'])) 
                         if np.isfinite(dark_exp_diff_raw) else np.nan)

        flat_candidates = flats[(flats['readout'].astype(str) == readout) & (flats['band'].astype(str) == band) & 
                                (flats['shape'].astype(int) == shape)] if len(flats) and 'band' in flats.columns else pd.DataFrame()
        flat_fname, flat_gap, _ = _closest_master_no_tolerance(jd, flat_candidates)

        rows.append({'filename': sci_row['filename'], 'jd': jd, 'exptime': sci_row['exptime'],
                     'readout': readout, 'band': band, 'dark_gap_days': dark_gap, 'dark_exp_diff': dark_exp_diff, 
                     'dark_candidate': dark_fname, 'flat_gap_days': flat_gap, 'flat_candidate': flat_fname})

    return pd.DataFrame(rows)

def summarize(df, current_dark_date_tol=12, current_flat_date_tol=45,
              current_dark_exp_tol=3):
    """
    Print a human-readable summary: how many frames are currently failing
    purely due to date tolerance (gap exists but exceeds current
    tolerance) vs. genuinely missing (no candidate at all, gap == inf).
    """
    if len(df) == 0:
        print('No data to summarize.')
        return

    n = len(df)
    print(f'=== Dark calibration gap analysis ({n} science frames) ===')
    no_dark_at_all = np.isinf(df['dark_gap_days']).sum()
    print(f'Frames with NO matching-readout/shape dark master at all: {no_dark_at_all}')

    has_dark = df[np.isfinite(df['dark_gap_days'])]
    if len(has_dark):
        within_tol = (has_dark['dark_gap_days'] <= current_dark_date_tol).sum()
        print(f'Of {len(has_dark)} frames with a candidate dark:')
        print(f'  Within current dark_date_tol={current_dark_date_tol}d: {within_tol}')
        print(f'  Gap distribution (days): '
              f'p50={has_dark["dark_gap_days"].median():.1f}  '
              f'p90={has_dark["dark_gap_days"].quantile(0.9):.1f}  '
              f'max={has_dark["dark_gap_days"].max():.1f}')
        # how many would be recovered by widening tolerance to p90
        suggested = has_dark['dark_gap_days'].quantile(0.9)
        recovered = ((has_dark['dark_gap_days'] > current_dark_date_tol) &
                     (has_dark['dark_gap_days'] <= suggested)).sum()
        print(f'  Widening dark_date_tol to {suggested:.1f}d would recover {recovered} more frames '
              f'(of those with a candidate at all)')

    print()
    print(f'=== Flat calibration gap analysis ({n} science frames) ===')
    no_flat_at_all = np.isinf(df['flat_gap_days']).sum()
    print(f'Frames with NO matching-readout/band/shape flat master at all: {no_flat_at_all}')

    has_flat = df[np.isfinite(df['flat_gap_days'])]
    if len(has_flat):
        within_tol = (has_flat['flat_gap_days'] <= current_flat_date_tol).sum()
        print(f'Of {len(has_flat)} frames with a candidate flat:')
        print(f'  Within current flat_date_tol={current_flat_date_tol}d: {within_tol}')
        print(f'  Gap distribution (days): '
              f'p50={has_flat["flat_gap_days"].median():.1f}  '
              f'p90={has_flat["flat_gap_days"].quantile(0.9):.1f}  '
              f'max={has_flat["flat_gap_days"].max():.1f}')
        suggested = has_flat['flat_gap_days'].quantile(0.9)
        recovered = ((has_flat['flat_gap_days'] > current_flat_date_tol) &
                     (has_flat['flat_gap_days'] <= suggested)).sum()
        print(f'  Widening flat_date_tol to {suggested:.1f}d would recover {recovered} more frames '
              f'(of those with a candidate at all)')

    print()
    print('Interpretation:')
    print('- If "NO matching dark/flat master at all" is large -> this is a genuine')
    print('  calibration-coverage gap (wrong readout/shape recorded, or no calibration')
    print('  frames were ever taken in that configuration). No tolerance value fixes this.')
    print('- If most frames HAVE a candidate but it sits just outside the current')
    print('  tolerance -> widening the tolerance (to roughly the suggested p90 value')
    print('  above) is a reasonable, data-driven fix.')
    print('- Be cautious widening flat_date_tol much beyond ~30-45 days regardless of')
    print('  what gaps exist -- flats track dust/illumination patterns that can drift')
    print('  faster than darks, especially in i/z band; a "recovered" frame calibrated')
    print('  against a very old flat may pass the reduction stage but still carry a')
    print('  flat-fielding systematic that the date gap alone does not capture.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    df = diagnose(TEST_NIGHT_GLOB)
    summarize(df)
    df.to_csv('calibration_gap_diagnosis.csv', index=False)
    print('\nFull per-frame breakdown written to calibration_gap_diagnosis.csv')