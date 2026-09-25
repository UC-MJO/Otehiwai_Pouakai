"""
Pouakai: end-to-end FLI/B&C reduction, astrometric, and photometric
calibration pipeline.

Usage
-----
As a cron job (processes whatever new files match a date glob, safe to
re-run -- already-processed files are skipped at each stage):

    0 14 * * * /usr/bin/python3 /path/to/otehiwai_pouakai.py --mode modulo \\
        --glob "/home/phys/astro8/MJArchive/octans/$(date +%Y%m%d)*/*.fit" \\
        --save-location /home/users/zgl12/Otehiwai_Nightly/ \\
        >> /home/users/zgl12/logs/pouakai_cron.log 2>&1

Ad hoc, on a specific file or small set of files (e.g. reprocessing a
frame after a parameter change, or running on a single frame
interactively):

    python3 otehiwai_pouakai.py --files /path/to/frame1.fit /path/to/frame2.fit \\
        --save-location /home/users/zgl12/Otehiwai_Test/ --mode modulo

Both paths go through the same `Pouakai` class and the same per-stage
skip-if-already-done logic, so there is no special "manual mode" to keep
in sync with the cron path -- the only difference is how the input file
list is constructed.
"""

from . import suppress_warnings  # noqa: F401 -- MUST be the first import: registers
                           # warning filters before anything below has a
                           # chance to import sklearn (which warns at
                           # import time, not call time -- see that
                           # module's docstring for why import order
                           # matters here).

import argparse
import logging
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from .organise_files import organise_fli_files
from .dark_masters import make_master_darks, get_master_dark
from .flat_masters import make_master_flats, get_master_flat
from .wcs_compute import wcs_astrometrynet_local
from .core_reduction import reduction_script, calibrating_internal
from .matau import get_file_paths, _deleting_wcs, rename_wcs, _file_creation, update_df
from .worker_logging import get_current_logging_config
from . import manifest as _manifest


def _resolve_stage_inputs(default_glob, input_files=None, input_dir=None, input_glob=None):
    """
    Resolve the candidate file list for a stage runner (_run_wcs /
    _run_calibration), independent of the pipeline's own save_location
    folder structure.

    Priority: explicit `input_files` list > `input_glob` pattern >
    `input_dir` (globbed for '*.fits.gz') > `default_glob` (the
    pipeline's own conventional save_location/<stage>/*.fits.gz
    folder). This lets a stage be pointed at ANY folder of
    already-prepared files (e.g. wcs-solved frames from somewhere
    else entirely) without needing to be copied into the pipeline's
    own directory layout, while still defaulting to the normal
    behaviour when nothing extra is passed.
    """
    if input_files is not None:
        return [str(f) for f in input_files]
    if input_glob is not None:
        return sorted(glob(input_glob))
    if input_dir is not None:
        return sorted(glob(str(Path(input_dir) / '*.fits.gz')))
    return sorted(glob(default_glob))

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", module="sklearn")
logging.getLogger('astroquery').setLevel(logging.WARNING)


class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler that writes via tqdm.write() instead of a raw
    stream write. tqdm.write() clears the active progress bar line,
    prints the message, then redraws the bar -- so log output and a live
    progress bar no longer visually clobber each other on the same
    terminal.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logging(save_location, level=logging.INFO, console_level=logging.ERROR):
    """
    Configure logging to both a full-detail file under
    `save_location/logs/` and a quieter console.

    Parameters
    ----------
    level : int
        Minimum level captured in the log FILE. Use logging.INFO (default)
        to keep the complete per-frame diagnostic trail needed to debug
        failures after the fact.
    console_level : int
        Minimum level shown on the CONSOLE. Defaults to ERROR -- routine
        per-frame INFO messages (e.g. "no usable RA/Dec header hint found;
        blind-solving") and per-frame WARNING messages (e.g. "no suitable
        master dark found", "too few calibration stars") are still fully
        captured in the log file, but don't print to the terminal and
        interrupt an active tqdm progress bar. With per-frame failure
        volumes that can run into the hundreds (see failure_ledger.py),
        the per-stage summary printed at the end of a run already gives
        a failure-reason breakdown, so per-frame console warnings would
        mostly be redundant noise. Only ERROR (unexpected exceptions, not
        routine skip/fail reasons) reaches the console by default. Set
        this to logging.WARNING or logging.INFO for more verbose console
        behaviour, or logging.CRITICAL for an entirely silent console
        (only tqdm bars and the final printed summary).

        NOTE: this only governs the MAIN process. The calibration stage
        runs worker code in separate OS processes (see
        Pouakai._run_calibration's docstring); those workers pick up the
        same log_file/level/console_level via worker_logging.py's
        configure_process_logging(), called from calibrating_internal.
        Without that plumbing, worker-process log calls would fall back
        to Python's unconfigured-logger stderr default (fixed at
        WARNING) and bypass this console_level setting entirely.
    """
    log_dir = Path(save_location) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"pouakai_{time.strftime('%Y%m%d_%H%M%S')}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(min(level, console_level))
    root_logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s %(levelname)-8s %(name)s: %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    return logging.getLogger('otehiwai_pouakai'), log_file


class Pouakai:
    def __init__(self, files, save_location, num_cores=1,
                 dark_exp_tol=1, dark_date_tol=3, dark_delta_t=7,
                 flat_exp_tol=3, flat_date_tol=30, flat_delta_t=15,
                 wcs_order=3, wcs_cpulimit=300, wcs_subprocess_timeout=330,
                 make_masters=True, organise_files=True,
                 run=True, mode='modulo', overwrite=True,
                 bkg_box_size=100, bkg_filter_size=3,
                 match_tol_px=2.5, isolation_radius_px=21.0,
                 max_contamination_frac=0.05, max_calibration_stars=150,
                 group_min_separation_px=None, group_min_separation_fwhm_factor=2.0,
                 max_group_size=25,
                 use_grouping=True,
                 psf_error_inflation_max_scale=8.0,
                 epsf_sampling_candidates=(3, 2),
                 assess_spatial_variation=True, subtract_background=True,
                 wcs_input_files=None, wcs_input_dir=None, wcs_input_glob=None,
                 cal_input_files=None, cal_input_dir=None, cal_input_glob=None):
        """
        wcs_input_files, wcs_input_dir, wcs_input_glob :
            Override the WCS stage's candidate frame list. By default
            (all None) it globs `save_location/red/*.fits.gz`, i.e. the
            output of THIS pipeline's own reduction stage. Pass
            `wcs_input_dir='/some/other/folder'` (or `wcs_input_files=
            [...]` / `wcs_input_glob='/path/*.fits.gz'`) to instead
            WCS-solve an arbitrary folder/subset of reduced frames that
            didn't come from this pipeline's reduction stage or don't
            follow its naming convention -- solved output still lands
            under `save_location/wcs/` as usual. Only relevant for
            mode in ('modulo', 'wcs').
        cal_input_files, cal_input_dir, cal_input_glob :
            Same idea for the calibration stage. By default globs
            `save_location/wcs/*.fits.gz`. Pass e.g.
            `cal_input_dir='/home/users/zgl12/Pouakai_Test_20250914/wcs'`
            to calibrate a specific folder of already WCS-solved frames
            directly, without them needing to sit under THIS run's
            `save_location/wcs/` or match any filename pattern --
            calibrating_internal() itself already only requires a
            readable FITS path, it does not require the '_wcs' naming
            convention (see its docstring). Only relevant for mode in
            ('modulo', 'cal'). Calibration outputs (cal/, phot_table/,
            zp/) always land under THIS run's `save_location`
            regardless of where the inputs came from.
        """

        self.logger = logging.getLogger('otehiwai_pouakai.Pouakai')

        extra_conds = {'dark_exp_tol': dark_exp_tol,
                       'dark_date_tol': dark_date_tol,
                       'dark_delta_t': dark_delta_t,
                       'flat_exp_tol': flat_exp_tol,
                       'flat_date_tol': flat_date_tol,
                       'flat_delta_t': flat_delta_t,
                       'shape': 2048}

        _file_creation(save_location)

        if organise_files:
            self._timed_stage('Organising files', self._run_organise, num_cores)

        if make_masters:
            self._timed_stage('Building master darks', make_master_darks,
                               exp_tol=extra_conds['dark_exp_tol'],
                               dark_delta_t=extra_conds['dark_delta_t'],
                               num_cores=num_cores)
            self._timed_stage('Building master flats', make_master_flats,
                               exp_tol=extra_conds['flat_exp_tol'],
                               flat_delta_t=extra_conds['flat_delta_t'],
                               dark_delta_t=extra_conds['dark_date_tol'],
                               num_cores=num_cores)

        updated_sci_list = update_df(files)
        self.logger.info(f'{len(updated_sci_list)} science frames matched for this run')

        mode = mode.lower()
        valid_modes = {'modulo', 'red', 'wcs', 'cal'}
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {sorted(valid_modes)}, got {mode!r}')

        if not run:
            self.logger.info('run=False; stopping after setup/master-building stages')
            return

        if mode in ('modulo', 'red'):
            self._run_reduction(updated_sci_list, save_location, extra_conds, num_cores,
                                 bkg_box_size, bkg_filter_size, subtract_background=subtract_background)

        if mode in ('modulo', 'wcs'):
            self._run_wcs(save_location, wcs_order, num_cores, wcs_cpulimit, wcs_subprocess_timeout,
                          input_files=wcs_input_files, input_dir=wcs_input_dir, input_glob=wcs_input_glob)

        if mode in ('modulo', 'cal'):
            self._run_calibration(save_location, num_cores, match_tol_px, isolation_radius_px,
                                   max_contamination_frac, max_calibration_stars,
                                   group_min_separation_px, group_min_separation_fwhm_factor,
                                   max_group_size, use_grouping, psf_error_inflation_max_scale,
                                   epsf_sampling_candidates, assess_spatial_variation,
                                   input_files=cal_input_files, input_dir=cal_input_dir,
                                   input_glob=cal_input_glob)

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _timed_stage(self, name, fn, *args, **kwargs):
        self.logger.info(f'--- Stage start: {name} ---')
        t0 = time.time()
        result = fn(*args, **kwargs)
        dt = time.time() - t0
        self.logger.info(f'--- Stage end: {name} ({dt:.1f}s) ---')
        return result

    def _run_organise(self, num_cores):
        organise_fli_files(num_cores=num_cores)

    def _run_reduction(self, updated_sci_list, save_location, extra_conds, num_cores,
                        bkg_box_size, bkg_filter_size, subtract_background=True):
        self.logger.info(f'--- Stage start: Reduction ({len(updated_sci_list)} frames) ---')
        t0 = time.time()

        results = Parallel(n_jobs=num_cores, backend="threading", prefer="threads")(
            delayed(reduction_script)(
                updated_sci_list, idx, save_location, extra_conds,
                bkg_box_size=bkg_box_size, bkg_filter_size=bkg_filter_size,
                subtract_background=subtract_background
            )
            for idx in tqdm(range(len(updated_sci_list)), desc='Reducing files')
        )

        dt = time.time() - t0
        self.logger.info(f'--- Stage end: Reduction ({dt:.1f}s) ---')

    def _run_wcs(self, save_location, wcs_order, num_cores, cpulimit=300, subprocess_timeout=330,
                 input_files=None, input_dir=None, input_glob=None):
        red_files = _resolve_stage_inputs(
            save_location + 'red/*.fits.gz',
            input_files=input_files, input_dir=input_dir, input_glob=input_glob,
        )
        self.logger.info(f'--- Stage start: WCS solving ({len(red_files)} candidate frames) ---')
        t0 = time.time()

        results = Parallel(n_jobs=num_cores, backend="threading", prefer="threads")(
            delayed(wcs_astrometrynet_local)(
                save_location, filename, wcs_order,
                cpulimit=cpulimit, subprocess_timeout=subprocess_timeout,
            )
            for filename in tqdm(red_files, desc='WCS Solving')
        )

        n_success = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        self.logger.info(f'WCS solved: {n_success}/{len(red_files)}')

        for r, fname in zip(results, red_files):
            ok = isinstance(r, dict) and r.get('success')
            reason = (r.get('reason') if isinstance(r, dict) else r) or ''
            new_file = r.get('new_file') if isinstance(r, dict) else None
            _manifest.record_stage(save_location, 'wcs', 'success' if ok else 'failed',
                                    input_path=fname, output_path=new_file, reason=reason)
            if not ok:
                self.logger.warning(f'WCS failed for {fname}: {reason}')

        _deleting_wcs(save_location)

        new_files = glob(save_location + 'wcs/*.new')
        Parallel(n_jobs=num_cores, backend="threading", prefer="threads")(
            delayed(rename_wcs)(filename) for filename in tqdm(new_files, desc='WCS Renaming')
        )

        dt = time.time() - t0
        self.logger.info(f'--- Stage end: WCS solving ({dt:.1f}s) ---')

    def _run_calibration(self, save_location, num_cores, match_tol_px=2.5, isolation_radius_px=21.0,
                          max_contamination_frac=0.05, max_calibration_stars=150,
                          group_min_separation_px=None, group_min_separation_fwhm_factor=2.0,
                          max_group_size=25, use_grouping=True,
                          psf_error_inflation_max_scale=8.0,
                          epsf_sampling_candidates=(3, 2),
                          assess_spatial_variation=True,
                          input_files=None, input_dir=None, input_glob=None):
        wcs_files = _resolve_stage_inputs(
            save_location + 'wcs/*.fits.gz',
            input_files=input_files, input_dir=input_dir, input_glob=input_glob,
        )
        self.logger.info(f'--- Stage start: Calibration ({len(wcs_files)} candidate frames) ---')
        t0 = time.time()

        # Deliberately uses joblib's default "loky" backend (separate OS
        # processes), NOT backend="threading" like the reduction/WCS
        # stages above. Calibration does CPU-heavy, GIL-bound work (ePSF
        # building, iterative PSF photometry) where true multi-core
        # parallelism via separate processes is actually beneficial,
        # unlike reduction (I/O- and numpy-vectorized-bound) or WCS
        # solving (delegates to an external `solve-field` subprocess
        # anyway, so threads vs. processes makes little difference there).
        #
        # This means any shared state written by worker calls here (e.g.
        # failure_ledger.py's CSV) must be safe across separate processes,
        # not just separate threads -- failure_ledger.py uses a real
        # cross-process file lock (fcntl.flock) plus atomic writes for
        # exactly this reason. Don't assume thread-only safety primitives
        # are sufficient for anything called from this stage.
        #
        # It ALSO means logging handlers configured in THIS process
        # (setup_logging, called before Pouakai(...) is constructed) do
        # not exist in those worker processes -- see worker_logging.py's
        # module docstring. Recover the current config here and pass it
        # through explicitly so worker-process log output goes to the
        # same file/console level rather than falling back to Python's
        # unconfigured-logger stderr default.
        log_file, level, console_level = get_current_logging_config()

        Parallel(n_jobs=num_cores)(
            delayed(calibrating_internal)(
                filename, save_location,
                match_tol_px=match_tol_px, isolation_radius_px=isolation_radius_px,
                max_contamination_frac=max_contamination_frac,
                max_calibration_stars=max_calibration_stars,
                group_min_separation_px=group_min_separation_px,
                group_min_separation_fwhm_factor=group_min_separation_fwhm_factor,
                max_group_size=max_group_size,
                use_grouping=use_grouping,
                psf_error_inflation_max_scale=psf_error_inflation_max_scale,
                epsf_sampling_candidates=epsf_sampling_candidates,
                assess_spatial_variation=assess_spatial_variation,
                _log_file=log_file, _log_level=level, _console_level=console_level,
            )
            for filename in tqdm(wcs_files, desc='Calibrating...')
        )

        dt = time.time() - t0
        self.logger.info(f'--- Stage end: Calibration ({dt:.1f}s) ---')


def build_arg_parser():
    p = argparse.ArgumentParser(
        description='Pouakai FLI reduction/astrometry/calibration pipeline.'
    )
    p.add_argument('--glob', type=str, default=None,
                    help='Glob pattern for raw science files (e.g. for cron, today\'s date folder). '
                         'Mutually exclusive with --files.')
    p.add_argument('--files', type=str, nargs='+', default=None,
                    help='Explicit list of raw science file paths to process. '
                         'Mutually exclusive with --glob.')
    p.add_argument('--save-location', type=str, required=True,
                    help='Root output directory (red/, wcs/, cal/, etc. will be created here).')
    p.add_argument('--num-cores', type=int, default=1)
    p.add_argument('--mode', type=str, default='modulo',
                    choices=['modulo', 'red', 'wcs', 'cal'])
    p.add_argument('--no-organise', action='store_true', help='Skip the file-organising stage.')
    p.add_argument('--wcs-input-dir', type=str, default=None,
                    help='Run the WCS stage (mode=wcs/modulo) against every *.fits.gz in this '
                         'folder instead of save-location/red/. Solved output still goes to '
                         'save-location/wcs/ as usual.')
    p.add_argument('--wcs-input-glob', type=str, default=None,
                    help='Same as --wcs-input-dir but with an explicit glob pattern, e.g. to '
                         'run WCS-solving on a small subset by name.')
    p.add_argument('--cal-input-dir', type=str, default=None,
                    help='Run the calibration stage (mode=cal/modulo) against every *.fits.gz '
                         'in this folder instead of save-location/wcs/ -- e.g. to calibrate '
                         'already WCS-solved frames from another run/location without needing '
                         'them inside THIS save-location\'s wcs/ folder or any particular '
                         'filename convention. Calibration outputs (cal/, phot_table/, zp/) '
                         'still land under save-location as usual.')
    p.add_argument('--cal-input-glob', type=str, default=None,
                    help='Same as --cal-input-dir but with an explicit glob pattern, e.g. to '
                         'calibrate a specific subset of files by name.')
    p.add_argument('--no-masters', action='store_true', help='Skip building master darks/flats.')
    p.add_argument('--dark-exp-tol', type=float, default=3)
    p.add_argument('--dark-date-tol', type=float, default=12)
    p.add_argument('--dark-delta-t', type=float, default=12)
    p.add_argument('--flat-exp-tol', type=float, default=3)
    p.add_argument('--flat-date-tol', type=float, default=45)
    p.add_argument('--flat-delta-t', type=float, default=15)
    p.add_argument('--wcs-order', type=int, default=3)
    p.add_argument('--wcs-cpulimit', type=float, default=300,
                    help='Seconds passed to solve-field\'s own --cpulimit (graceful internal '
                         'give-up). Default 300s (5 min).')
    p.add_argument('--wcs-timeout', type=float, default=330,
                    help='Hard wall-clock backstop (seconds) enforced by the pipeline itself, '
                         'in case --cpulimit alone does not bound wall-clock time. Should be '
                         'somewhat larger than --wcs-cpulimit. Default 330s.')
    p.add_argument('--bkg-box-size', type=int, default=100)
    p.add_argument('--bkg-filter-size', type=int, default=3)
    p.add_argument('--match-tol-px', type=float, default=2.5,
                    help='Max pixel distance for Gaia<->detection and PSF-fit<->detection '
                         'matching. 2.5px comfortably covers typical combined WCS+centroid '
                         'scatter; a much tighter tolerance risks rejecting genuine matches '
                         'and causing "too few calibration stars" failures.')
    p.add_argument('--isolation-radius-px', type=float, default=21.0,
                    help='Max search radius (px) for the flux-contamination estimate used to '
                         'select clean calibration stars (see calibration_saurus.py). This is '
                         'not a hard exclusion radius -- see matching_sources docstring.')
    p.add_argument('--max-contamination-frac', type=float, default=0.05,
                    help='A Gaia-matched star is usable for calibration if the estimated '
                         'fraction of its own flux contaminated by nearby neighbours is below '
                         'this. Relaxed automatically (up to a ceiling) if too few stars '
                         'survive -- see calibration_saurus.matching_sources.')
    p.add_argument('--max-calibration-stars', type=int, default=150,
                    help='Cap on how many usable stars are fed into ePSF building/PSF '
                         'photometry, keeping the brightest N. Feeding an unbounded number '
                         '(can be several hundred in a rich field) into one simultaneous '
                         'multi-star fit was found to produce unstable per-star flux '
                         'measurements. Pass 0 or a negative number to disable the cap.')
    p.add_argument('--group-min-separation-px', type=float, default=None,
                    help='SourceGrouper min_separation (px) -- stars closer than this are fit '
                         'SIMULTANEOUSLY as one joint group by IterativePSFPhotometry. Default '
                         '(unset) derives this per-frame from the frame\'s own measured FWHM '
                         '(see --group-min-separation-fwhm-factor) rather than a fixed pixel '
                         'value, so grouping tracks each frame\'s actual PSF width: a value far '
                         'larger than the actual PSF width would group stars that aren\'t '
                         'remotely blended, while one smaller than the PSF width could split '
                         'real blends apart. Pass an explicit value here only to override the '
                         'per-frame derivation.')
    p.add_argument('--group-min-separation-fwhm-factor', type=float, default=2.0,
                    help='Multiplier on measured fwhm_px used to derive '
                         '--group-min-separation-px when that is left unset. 2.0 (default) '
                         'means stars within 2 PSF-widths of each other are treated as a blend '
                         'requiring a joint fit.')
    p.add_argument('--max-group-size', type=int, default=25,
                    help='Ceiling on simultaneous PSF-fit group size, regardless of '
                         '--group-min-separation-px. SourceGrouper links stars transitively, '
                         'so a fixed min_separation alone is not safe across fields of varying '
                         'density -- min_separation is shrunk automatically (down to a 2px '
                         'floor) whenever the group it would produce exceeds this. Prevents an '
                         'astropy.modeling RecursionError from an oversized simultaneous fit '
                         'group (see psf_photometry.PSFGroupingError). Ignored if '
                         '--no-grouping is set.')
    p.add_argument('--no-grouping', dest='use_grouping', action='store_false',
                    help='Bypass simultaneous group-fitting entirely -- every source fit '
                         'independently. Eliminates the astropy.modeling recursion risk '
                         'structurally and was empirically more stable than even a small '
                         'simultaneous group for a dense test field; the pipeline\'s flux-'
                         'deblending step (using actual measured fluxes) still corrects for '
                         'residual neighbour contamination afterward. Use '
                         'calibration_diagnostics.py\'s grouping-stability sweep to compare '
                         'against simultaneous-group results for your data first.')
    p.set_defaults(use_grouping=True)
    p.add_argument('--psf-error-inflation-max-scale', type=float, default=8.0,
                    help="Sanity ceiling on psf_photometry.inflate_psf_errors's empirical "
                         "PSF-vs-aperture error inflation factor (computed from high-SNR, "
                         "isolated stars). Raised from an earlier default of 5.0 to 8.0 after "
                         "two independently-tested fields both converged on an empirical scale "
                         "around 7-7.7 even after isolation-gating -- see "
                         "calibration_saurus.cal_photom's docstring for the reasoning. "
                         "Validate against more of your own data before trusting this as final.")
    p.add_argument('--epsf-sampling', type=int, nargs='+', default=[3, 2],
                    help='ePSF oversampling candidates, most preferred first -- tries each in '
                         'order, falling back with a logged warning if a frame\'s star sample '
                         'can\'t support the preferred value. See '
                         'psf_photometry.build_epsf_adaptive.')
    p.add_argument('--no-spatial-variation-check', dest='assess_spatial_variation',
                    action='store_false',
                    help='Skip the diagnostic-only spatial zeropoint variation check (does not '
                         'affect the recorded ZP -- see calibration_saurus.cal_photom\'s '
                         'assess_spatial_variation docstring). On by default; disabling saves a '
                         'small amount of time per frame.')
    p.set_defaults(assess_spatial_variation=True)
    p.add_argument('--log-level', type=str, default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='Minimum level captured in the log FILE.')
    p.add_argument('--console-log-level', type=str, default='ERROR',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='Minimum level shown on the CONSOLE. Defaults to ERROR so routine '
                         'per-frame INFO/WARNING messages do not interrupt active tqdm progress '
                         'bars; the full detail is still always in the log file, and per-stage '
                         'failure-reason summaries are printed at the end of a run. Use CRITICAL '
                         'for a fully silent console (only tqdm bars and the final summary).')
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # --glob/--files feed the RAW-frame -> reduction stage. If this run
    # only touches the wcs or cal stage (mode='wcs'/'cal') and the
    # relevant --*-input-dir/--*-input-glob override is given instead,
    # there's nothing for --glob/--files to do, so don't require them.
    stage_input_given = (
        (args.mode == 'wcs' and (args.wcs_input_dir or args.wcs_input_glob)) or
        (args.mode == 'cal' and (args.cal_input_dir or args.cal_input_glob))
    )
    if (args.glob is None) == (args.files is None) and not stage_input_given:
        parser.error('Exactly one of --glob or --files must be provided (unless running a single '
                      'stage with --wcs-input-dir/--wcs-input-glob or --cal-input-dir/--cal-input-glob).')

    logger, log_file = setup_logging(
        args.save_location, level=getattr(logging, args.log_level),
        console_level=getattr(logging, args.console_log_level),
    )
    logger.info(f'Logging to {log_file}')

    if args.glob is not None:
        files = get_file_paths(args.glob)
        logger.info(f'Resolved {len(files)} files from glob {args.glob!r}')
    elif args.files is not None:
        files = list(args.files)
        logger.info(f'Using {len(files)} explicitly provided files')
    else:
        files = []

    if len(files) == 0 and not stage_input_given:
        logger.warning('No input files found; exiting.')
        return 0

    t0 = time.time()
    max_cal_stars = args.max_calibration_stars if args.max_calibration_stars > 0 else None
    try:
        Pouakai(
            files, args.save_location, num_cores=args.num_cores,
            dark_exp_tol=args.dark_exp_tol, dark_date_tol=args.dark_date_tol, dark_delta_t=args.dark_delta_t,
            flat_exp_tol=args.flat_exp_tol, flat_date_tol=args.flat_date_tol, flat_delta_t=args.flat_delta_t,
            wcs_order=args.wcs_order, wcs_cpulimit=args.wcs_cpulimit, wcs_subprocess_timeout=args.wcs_timeout,
            make_masters=not args.no_masters, organise_files=not args.no_organise,
            run=True, mode=args.mode,
            bkg_box_size=args.bkg_box_size, bkg_filter_size=args.bkg_filter_size,
            match_tol_px=args.match_tol_px, isolation_radius_px=args.isolation_radius_px,
            max_contamination_frac=args.max_contamination_frac,
            max_calibration_stars=max_cal_stars,
            group_min_separation_px=args.group_min_separation_px,
            group_min_separation_fwhm_factor=args.group_min_separation_fwhm_factor,
            max_group_size=args.max_group_size,
            use_grouping=args.use_grouping,
            psf_error_inflation_max_scale=args.psf_error_inflation_max_scale,
            wcs_input_dir=args.wcs_input_dir, wcs_input_glob=args.wcs_input_glob,
            cal_input_dir=args.cal_input_dir, cal_input_glob=args.cal_input_glob,
        )
    except Exception:
        logger.exception('Pipeline run failed with an unhandled exception')
        return 1

    logger.info(f'Pipeline run complete in {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())