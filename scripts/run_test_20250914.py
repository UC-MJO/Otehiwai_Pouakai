"""
One-off test run: organise the full archive, then run the FULL pipeline
(reduction + WCS solving + calibration) on a single night, specifically to
surface failure modes end-to-end.

What this does
---------------
1. Organises ALL files under /home/phys/astro8/MJArchive/octans/ (recursing
   into every subfolder) using 30 cores. This updates
   bc_all_image_list.csv / bc_dark_image_list.csv / bc_flat_image_list.csv
   / bc_science_image_list.csv with anything not already catalogued --
   organise_fli_files() only processes files it hasn't seen before, so
   this is safe to re-run.
2. Builds any missing master darks/flats (needed for step 3 -- reduction
   looks up a master dark/flat via get_master_dark/get_master_flat, so
   without this step every frame would fail with "No Dark!!!"/"No
   Flat!!!" for the wrong reason). Already-built masters are skipped, so
   this is cheap on a re-run.
3. Runs the FULL pipeline (mode='modulo': reduction -> WCS solving ->
   calibration) restricted to science files in
   /home/phys/astro8/MJArchive/octans/20250914/.

   This is deliberately the full chain, not just reduction, because the
   goal of this run is to see where and how things fail across every
   stage -- a frame can fail reduction (no dark/flat match), WCS solving
   (no astrometric solution, timeout), or calibration (no Gaia matches,
   too few isolated stars, bad filter mapping) for different reasons, and
   each failure mode is logged with a specific reason rather than a bare
   exception. At the end, this script prints a per-stage failure summary
   pulled from the wcs/red/cal directory contents so you don't have to
   dig through the log file to see the overall picture.

Usage
-----
    python run_test_20250914.py

Adjust SAVE_LOCATION below to wherever you want the test output written.
"""

"""
One-off test run: organise the full archive, then run the FULL pipeline
(reduction + WCS solving + calibration) on a single night, specifically to
surface failure modes end-to-end.

What this does
---------------
1. Organises ALL files under /home/phys/astro8/MJArchive/octans/ (recursing
   into every subfolder) using 30 cores. This updates
   bc_all_image_list.csv / bc_dark_image_list.csv / bc_flat_image_list.csv
   / bc_science_image_list.csv with anything not already catalogued --
   organise_fli_files() only processes files it hasn't seen before
   (including files previously marked telescope='bad'), so this is safe
   to re-run.
2. Builds any missing master darks/flats (needed for step 3 -- reduction
   looks up a master dark/flat via get_master_dark/get_master_flat, so
   without this step every frame would fail with "No Dark!!!"/"No
   Flat!!!" for the wrong reason). Already-built masters are skipped, so
   this is cheap on a re-run.
3. Runs the FULL pipeline (mode='modulo': reduction -> WCS solving ->
   calibration) restricted to science files in
   /home/phys/astro8/MJArchive/octans/20250914/.

   This is deliberately the full chain, not just reduction, because the
   goal of this run is to see where and how things fail across every
   stage. Each stage now persists failures to a shared ledger
   (failure_ledger.py, stored at <save_location>/logs/failed_files.csv),
   so a RE-RUN of this script will, by default, skip any file that
   previously failed at a given stage rather than re-attempting it from
   scratch -- set RETRY_KNOWN_FAILURES=True below to force a full retry
   (e.g. after changing match_tol_px/isolation_radius_px and expecting
   previously-failing frames to now succeed).

Console output is quiet by default (ERROR+ only, plus a handful of one-time
startup messages and the final summary, which are always printed directly)
so routine per-frame INFO/WARNING messages don't interrupt the tqdm
progress bars -- the full detail, including every INFO/WARNING line, is
always in the log file.

Usage
-----
    python run_test_20250914.py

Adjust SAVE_LOCATION below to wherever you want the test output written.
"""

import logging

from otehiwai_pouakai.pipeline import Pouakai, setup_logging
from otehiwai_pouakai.matau import get_file_paths
from otehiwai_pouakai.failure_ledger import load_known_failures, load_all_failures

NUM_CORES = 30
ORGANISE_GLOB = '/home/phys/astro8/MJArchive/octans/'          # full archive, all subfolders
# TEST_NIGHT_GLOB = '/home/phys/astro8/MJArchive/octans/20250914/*.fit'  # single night to run
TEST_NIGHT_GLOB = '/home/phys/astro8/MJArchive/octans/ASTR211_2025/20250823/*.fit'
SAVE_LOCATION = '/home/users/zgl12/Pouakai_Test_20250914/'      # adjust as needed

# Calibration matching tolerances -- see calibration_saurus.py docstring.
# The previous default (match_tol_px=1.0) was tighter than typical
# combined WCS-solution + centroid positional scatter and was very likely
# the dominant cause of widespread "too few calibration stars" failures.
MATCH_TOL_PX = 2.5
ISOLATION_RADIUS_PX = 21.0

# Set True to force every stage to re-attempt files that failed in a
# previous run of this script (e.g. after changing the tolerances above).
RETRY_KNOWN_FAILURES = True


def _print_stage_summary(logger, save_location, n_science_input):
    """
    Report per-stage pass/fail counts AND a breakdown of failure reasons
    pulled from the persistent failure ledger -- this is the quick "what
    failed, where, and why" view for a deliberate failure-mode test.

    This is printed directly via `print()` (in addition to being logged
    at INFO for the file record), since by design the console logging
    level is set quiet (ERROR by default) to keep per-frame noise out of
    the tqdm progress bars -- but this final summary should always be
    visible on screen regardless of that setting, and by the time this
    runs there is no active progress bar left to clobber anyway.
    """
    from glob import glob

    red_files = glob(save_location + 'red/*.fits.gz')
    wcs_new = glob(save_location + 'wcs/*.new')
    wcs_solved = glob(save_location + 'wcs/*.fits.gz')
    cal_files = glob(save_location + 'cal/*.fits.gz')
    phot_tables = glob(save_location + 'phot_table/*.csv')

    lines = []
    lines.append('==================== STAGE SUMMARY ====================')
    lines.append(f'Input science frames (20250914):      {n_science_input}')
    lines.append(f'Reduced (red/*.fits.gz):               {len(red_files)}  '
                 f'(failed/skipped: {n_science_input - len(red_files)})')
    lines.append(f'WCS-solved (wcs/*.fits.gz):             {len(wcs_solved)}  '
                 f'(failed to solve: {len(red_files) - len(wcs_solved)})')
    if wcs_new:
        lines.append(f'WARNING: {len(wcs_new)} solved .new files were not renamed '
                      f'(unexpected -- check rename_wcs failures in the log)')
    lines.append(f'Calibrated (cal/*.fits.gz):             {len(cal_files)}  '
                 f'(failed calibration: {len(wcs_solved) - len(cal_files)})')
    lines.append(f'Photometry tables written (phot_table): {len(phot_tables)}')
    lines.append('========================================================')

    # Failure-reason breakdown, per stage, from the persistent ledger.
    df = load_all_failures(save_location)
    if len(df) == 0:
        lines.append('No failures recorded in the ledger.')
    else:
        lines.append('--- Failure reason breakdown (from failure_ledger) ---')
        for stage in sorted(df['stage'].unique()):
            stage_df = df[df['stage'] == stage]
            lines.append(f'[{stage}] {len(stage_df)} recorded failures:')
            counts = stage_df['reason'].value_counts()
            for reason, count in counts.items():
                lines.append(f'    {count:4d}x  {reason}')
    lines.append('========================================================')

    summary_text = '\n'.join(lines)
    print('\n' + summary_text)       # always visible on console, regardless of console_level
    logger.info('\n' + summary_text)  # also captured in the log file


def main():
    # console_level=ERROR keeps the terminal to progress bars only (plus
    # genuine unexpected exceptions) -- per-frame WARNINGs like "no
    # suitable master dark found" or "too few calibration stars" are
    # common enough at real-world failure volumes that even WARNING-level
    # console output was disruptive; the per-stage failure-reason
    # breakdown at the end of this script, plus the full log file, cover
    # that detail instead.
    logger, log_file = setup_logging(SAVE_LOCATION, level=logging.INFO,
                                    console_level=logging.CRITICAL)

    startup_msgs = [
        f'Logging to {log_file}',
        f'Using {NUM_CORES} cores throughout',
        f'match_tol_px={MATCH_TOL_PX}, isolation_radius_px={ISOLATION_RADIUS_PX}, '
        f'retry_known_failures={RETRY_KNOWN_FAILURES}',
    ]

    # Step 3 needs an explicit file list for just the test night.
    # get_file_paths already excludes filenames containing dark/flat/bias.
    test_files = get_file_paths(TEST_NIGHT_GLOB)
    startup_msgs.append(f'Resolved {len(test_files)} science files for 20250914 test run')

    if RETRY_KNOWN_FAILURES:
        for stage in ('reduction', 'wcs', 'calibration'):
            n = len(load_known_failures(SAVE_LOCATION, stage))
            if n:
                startup_msgs.append(f'RETRY_KNOWN_FAILURES=True: {n} previously-failed files at '
                                    f'stage "{stage}" will be re-attempted this run')

    if len(test_files) == 0:
        startup_msgs.append(f'WARNING: No files found matching {TEST_NIGHT_GLOB}; nothing to run. '
                            'Organising will still run below.')

    # These happen once before any progress bar exists, so printing them
    # directly is always safe and they're useful context to see even with
    # console_level=ERROR suppressing routine per-frame messages.
    for msg in startup_msgs:
        print(msg)
        logger.info(msg)

    # organise_files=True here recurses through the ENTIRE archive
    # (organise_fli_files uses Path(fli_dir).rglob("*.fit*") internally,
    # and fli_dir is fixed to /home/phys/astro8/MJArchive/octans/ -- the
    # ORGANISE_GLOB constant above is just documenting that, not an
    # argument that gets passed through).
    #
    # make_masters=True builds any darks/flats newly discovered by the
    # organise step above, so the reduction step below has something to
    # look up.
    #
    # mode='modulo' runs the FULL chain: reduction -> WCS solving ->
    # calibration, so failure modes at every stage are visible.
    Pouakai(files=test_files, save_location=SAVE_LOCATION, num_cores=NUM_CORES,
            organise_files=True, make_masters=True, run=True, mode='modulo',
            dark_exp_tol=3, dark_date_tol=12, dark_delta_t=12,
            flat_exp_tol=3, flat_date_tol=45, flat_delta_t=15,
            match_tol_px=MATCH_TOL_PX, isolation_radius_px=ISOLATION_RADIUS_PX, subtract_background=True)

    _print_stage_summary(logger, SAVE_LOCATION, len(test_files))
    logger.info('Test run complete.')

if __name__ == '__main__':
    main()