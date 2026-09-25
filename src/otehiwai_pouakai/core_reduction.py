import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats

from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation

from .background_subtraction import determine_background, background_residual_flatness, build_source_and_nebula_masks
from .failure_ledger import record_failure, clear_failure, is_known_failure
from .provenance import build_provenance_dict
from .worker_logging import configure_process_logging
from . import manifest as _manifest

import os
import logging
from glob import glob
from pathlib import Path

from .dark_masters import get_master_dark
from .flat_masters import get_master_flat
from .calibration_saurus import cal_photom, NarrowbandFilterError
from .frame_quality import SepExtractionError
from .psf_photometry import PSFGroupingError

logger = logging.getLogger(__name__)

_STAGE_REDUCTION = 'reduction'
_STAGE_WCS = 'wcs'
_STAGE_CALIBRATION = 'calibration'


def reduction_script(updated_sci_list, i, save_location,
                      extra_conds={'dark_exp_tol': 1,
                                   'dark_date_tol': 3,
                                   'flat_exp_tol': 3,
                                   'flat_date_tol': 30,
                                   'shape': 2048},
                      bkg_box_size=100, bkg_filter_size=3,
                      skip_known_failures=True, subtract_background=True):
    """
    Reduce a single science frame: dark-subtract, flat-field, estimate and
    subtract the sky background, and write a calibrated, gzipped FITS file.

    Parameters
    ----------
    skip_known_failures : bool
        If True (default), skip this frame without re-attempting it if it
        is recorded in the failure ledger from a previous run -- e.g. a
        frame with no matching master dark is unlikely to suddenly have
        one on a re-run within the same night, so retrying it every run
        wastes time. Set False to force a retry regardless of ledger
        history (e.g. after building new masters, or after a parameter
        change you expect to fix frames that failed before).

    Returns
    -------
    None always (kept for compatibility with the Parallel(...) call
    pattern in otehiwai_pouakai.py, which discards return values).
    Failures are logged with a specific reason via `logger`, and
    recorded in the persistent failure ledger (failure_ledger.py) so a
    rerun does not re-attempt the same known-bad file from scratch.
    """
    sci_file = updated_sci_list['filename'].iloc[i]
    exptime = updated_sci_list['exptime'].iloc[i]
    jd = updated_sci_list['jd'].iloc[i]
    readout = updated_sci_list['readout'].iloc[i]
    band = updated_sci_list['band'].iloc[i]

    new_name = updated_sci_list['master_name'].iloc[i]
    new_name = new_name.replace(' ', '_')
    new_name = new_name + '_reduced'

    save_name = Path(save_location + 'red/') / new_name
    save_name = save_name.with_suffix('.fits')

    # Skip if this frame has already been reduced, so repeated (e.g.
    # nightly cron) runs only pay the reduction cost for new frames
    # rather than re-reducing the entire historical dataset each time.
    if os.path.exists(str(save_name) + '.gz'):
        return None

    if skip_known_failures:
        known_reason = is_known_failure(save_location, _STAGE_REDUCTION, sci_file)
        if known_reason is not None:
            logger.info(f'{i} {sci_file}: skipping (known failure: {known_reason})')
            return None

    dark_fname, dark_t_diff = get_master_dark(
        jd, exptime, readout,
        exp_tol=extra_conds['dark_exp_tol'],
        date_tol=extra_conds['dark_date_tol'],
        shape=extra_conds['shape'],
    )
    flat_fname, flat_t_diff = get_master_flat(
        jd, readout, band,
        date_tol=extra_conds['flat_date_tol'],
        shape=extra_conds['shape'],
    )

    if dark_fname == 'none':
        reason = 'no suitable master dark found'
        logger.warning(f'{i} {sci_file}: {reason}')
        record_failure(save_location, _STAGE_REDUCTION, sci_file, reason)
        _manifest.record_stage(save_location, _STAGE_REDUCTION, 'failed', input_path=sci_file, reason=reason)
        return None

    if flat_fname == 'none':
        reason = 'no suitable master flat found'
        logger.warning(f'{i} {sci_file}: {reason}')
        record_failure(save_location, _STAGE_REDUCTION, sci_file, reason)
        _manifest.record_stage(save_location, _STAGE_REDUCTION, 'failed', input_path=sci_file, reason=reason)
        return None

    try:
        with fits.open(sci_file, memmap=False) as hdul:
            sci_data = hdul[0].data.astype(np.float64)
            sci_header = hdul[0].header
        with fits.open(dark_fname, memmap=False) as hdul:
            dark_data = hdul[0].data.astype(np.float64)
        with fits.open(flat_fname, memmap=False) as hdul:
            flat_data = hdul[0].data.astype(np.float64)
    except Exception as e:
        reason = f'failed to read input/calibration FITS: {e}'
        logger.error(f'{i} {sci_file}: {reason}')
        record_failure(save_location, _STAGE_REDUCTION, sci_file, reason)
        return None

    with np.errstate(divide='ignore', invalid='ignore'):
        reduced_data = (sci_data - dark_data) / flat_data
    reduced_data[~np.isfinite(reduced_data)] = np.nan

    bkg_status = 'SUCCESS'
    bkg_rms_median = np.nan
    residual_flatness = np.nan

    try:
        background, background_rms = determine_background(
            reduced_data, box_size=bkg_box_size, filter_size=bkg_filter_size,
            plot=False, testing_plot=False, verbose=False,
        )
        
        if subtract_background:
            reduced_data -= background
            bkg_rms_median = float(np.nanmedian(background_rms))
        else:
            bkg_status = 'SKIPPED'
            bkg_rms_median = float(np.nanmedian(background_rms))
            residual_flatness = np.nan

        # Diagnostic: log residual sky flatness post-subtraction. This is
        # the feedback signal recommended for catching mask-threshold
        # failures on new fields without relying on visual inspection.
        _, _, source_mask = build_source_and_nebula_masks(reduced_data + background)
        residual_flatness, _ = background_residual_flatness(
            reduced_data, np.zeros_like(reduced_data), source_mask,
            box_size=bkg_box_size,
        )

    except Exception as e:
        logger.warning(f'{i} {sci_file}: background estimation failed ({e}); falling back to sigma-clipped median')
        bkg_status = 'FALLBACK_MEDIAN'
        _, median, _ = sigma_clipped_stats(reduced_data, sigma=3)
        background = np.full_like(reduced_data, median)
        reduced_data -= median

    sci_header['DATERED'] = Time.now().isot
    sci_header['REDUCED'] = (True, 'Data reduced')
    sci_header['REDSTAT'] = ('SUCCESS', 'Photometric reduction status')
    sci_header['REDPIPE'] = ('reduction_script', 'Reduction pipeline script')
    sci_header['REDBKG'] = ('determine_background', 'Background pipeline script')
    sci_header['REDBKGST'] = (bkg_status, 'Background estimation status')
    sci_header['REDSKY'] = (float(np.nanmedian(background)), 'Median background level')
    _safe_header_set(sci_header, 'REDSKRMS', bkg_rms_median, 'Median per-pixel background RMS')
    _safe_header_set(sci_header, 'REDFLAT', residual_flatness,
                      'Std of per-tile residual sky after subtraction (mag/flux units)')

    # Provenance (item 7): exact calibration inputs used for THIS frame,
    # so a systematic discovered later can be traced back to the precise
    # dark/flat masters and pipeline version involved, rather than only
    # knowing that *some* dark/flat correction happened.
    prov = build_provenance_dict(dark_filename=dark_fname, flat_filename=flat_fname)
    for key, (value, comment) in prov.items():
        sci_header[key] = (value, comment)

    phdu = fits.PrimaryHDU(data=reduced_data, header=sci_header)
    hdul = fits.HDUList([phdu])

    try:
        hdul.writeto(save_name, overwrite=True)
        os.system(f"gzip -f {save_name}")
        clear_failure(save_location, _STAGE_REDUCTION, sci_file)
        _manifest.record_stage(save_location, _STAGE_REDUCTION, 'success',
                                input_path=sci_file, output_path=str(save_name) + '.gz')
    except Exception as e:
        reason = f'failed to write output: {e}'
        logger.error(f'{i} {sci_file}: {reason}')
        record_failure(save_location, _STAGE_REDUCTION, sci_file, reason)
        _manifest.record_stage(save_location, _STAGE_REDUCTION, 'failed', input_path=sci_file, reason=reason)

    return None


def _safe_header_set(header, key, value, comment):
    """
    Set a FITS header keyword, omitting it entirely if `value` is a
    non-finite float (NaN/inf) rather than letting astropy.io.fits raise
    ValueError ("Floating point nan values are not allowed in FITS
    headers"). This matters because some calibration metrics are
    legitimately NaN on certain code paths -- e.g. aperture_correction's
    documented (1.0, np.nan) fallback when the correction itself fails --
    and a NaN anywhere in the metrics block would otherwise abort writing
    the header for an otherwise fully successful calibration.

    Omitting the keyword (rather than writing a sentinel like -999) is
    the standard FITS convention for "not available" and is unambiguous
    for any downstream reader: check `'KEY' in header` rather than having
    to know which sentinel value means missing for which specific
    keyword.
    """
    if isinstance(value, float) and not np.isfinite(value):
        return
    header[key] = (value, comment)


def calibrating_internal(filename, save_location, skip_known_failures=True,
                          match_tol_px=2.5, isolation_radius_px=21.0,
                          max_contamination_frac=0.05, max_calibration_stars=150,
                          group_min_separation_px=None, group_min_separation_fwhm_factor=2.0,
                          max_group_size=25,
                          use_grouping=True,
                          psf_error_inflation_max_scale=8.0,
                          epsf_sampling_candidates=(3, 2),
                          assess_spatial_variation=True,
                          _log_file=None, _log_level=logging.INFO,
                          _console_level=logging.ERROR):
    """
    Parameters
    ----------
    skip_known_failures : bool
        If True (default), skip this frame without re-attempting it if it
        is recorded in the failure ledger from a previous run (e.g. a
        field with consistently too few isolated calibration stars). Set
        False to force a retry, e.g. after changing match_tol_px /
        isolation_radius_px in a way that should let a previously
        failing frame succeed.
    match_tol_px : float
        Passed through to cal_photom -- see calibration_saurus.py.
    isolation_radius_px : float
        Passed through to cal_photom -- the max search radius (px) used
        for the flux-contamination estimate. See
        calibration_saurus.matching_sources's docstring.
    max_contamination_frac : float
        Passed through to cal_photom -- see calibration_saurus.py.
    max_calibration_stars : int or None
        Passed through to cal_photom -- caps the number of stars fed
        into ePSF building/PSF photometry to the brightest N, for
        numerical stability. See calibration_saurus.matching_sources.
    group_min_separation_px : float or None
        Passed through to cal_photom -- SourceGrouper's requested
        min_separation for IterativePSFPhotometry's simultaneous
        multi-star fit groups. If None (default), derived per-frame from
        the frame's own measured FWHM (see
        group_min_separation_fwhm_factor and
        calibration_saurus.cal_photom's docstring) rather than a fixed
        pixel value, so grouping tracks each frame's actual PSF width
        instead of over- or under-grouping stars that aren't (or are)
        genuinely blended. Also automatically shrunk further per-frame
        if it would still produce an oversized group -- see
        max_group_size. Ignored if use_grouping=False.
    group_min_separation_fwhm_factor : float
        Passed through to cal_photom -- multiplier on measured fwhm_px
        used to derive group_min_separation_px when that is None. Only
        used when group_min_separation_px is None.
    max_group_size : int
        Passed through to cal_photom -- ceiling on simultaneous PSF-fit
        group size, regardless of group_min_separation_px. See
        calibration_saurus.cal_photom's docstring and
        psf_photometry.PSFGroupingError for why this exists. Ignored if
        use_grouping=False.
    epsf_sampling_candidates : sequence of int
        Passed through to cal_photom -- ePSF oversampling factors to
        try, most preferred first, falling back with a logged warning
        if a frame's star sample can't support the preferred value. See
        calibration_saurus.cal_photom's docstring.
    assess_spatial_variation : bool
        Passed through to cal_photom -- diagnostic-only spatial
        zeropoint variation check (does not modify the image). See
        calibration_saurus.cal_photom's docstring.
    use_grouping : bool
        Passed through to cal_photom -- if False, bypasses simultaneous
        group-fitting entirely (every source fit independently),
        eliminating the astropy.modeling recursion risk structurally.
        See calibration_saurus.cal_photom's docstring.
    _log_file, _log_level, _console_level :
        Internal. Passed by otehiwai_pouakai.Pouakai._run_calibration so
        this worker process (a separate OS process under joblib's loky
        backend) configures logging identically to the main process --
        see worker_logging.py's module docstring for why this is
        necessary. Not meant to be set by hand when calling this
        function directly/interactively -- leave as the defaults, and
        logging falls back to Python's own defaults.
    """
    configure_process_logging(_log_file, level=_log_level, console_level=_console_level)

    if skip_known_failures:
        known_reason = is_known_failure(save_location, _STAGE_CALIBRATION, filename)
        if known_reason is not None:
            logger.info(f'{filename}: skipping (known failure: {known_reason})')
            return None

    try:
        infile = Path(filename)
        cal_outdir = Path(save_location) / 'cal'
        phot_outdir = Path(save_location) / 'phot_table'
        zp_outdir = Path(save_location) / 'zp'
        cal_outdir.mkdir(parents=True, exist_ok=True)
        phot_outdir.mkdir(parents=True, exist_ok=True)
        zp_outdir.mkdir(parents=True, exist_ok=True)

        cal_new_name = infile.name.replace('_wcs', '_cal').replace('.fits.gz', '.fits')
        phot_new_name = infile.name.replace('_wcs', '_phottable').replace('.fits.gz', '.csv')
        zp_new_name = infile.name.replace('_wcs', '_zpsurface').replace('.fits.gz', '.npy')
        cal_new_path = cal_outdir / cal_new_name
        phot_new_path = phot_outdir / phot_new_name
        zp_new_path = zp_outdir / zp_new_name

        if os.path.exists(str(cal_new_path) + '.gz'):
            return None

        cally = cal_photom(filename, match_tol_px=match_tol_px, isolation_radius_px=isolation_radius_px,
                            max_contamination_frac=max_contamination_frac,
                            max_calibration_stars=max_calibration_stars,
                            group_min_separation_px=group_min_separation_px,
                            group_min_separation_fwhm_factor=group_min_separation_fwhm_factor,
                            max_group_size=max_group_size,
                            use_grouping=use_grouping,
                            psf_error_inflation_max_scale=psf_error_inflation_max_scale,
                            epsf_sampling_candidates=epsf_sampling_candidates,
                            assess_spatial_variation=assess_spatial_variation)

        if cally.zeropoint_results is None:
            # zeropoint_results is None on several distinct early-return
            # paths inside cal_photom (frame quality fail, ePSF build
            # fail, or genuinely too few calibration stars after
            # matching) -- surface whichever specific reason actually
            # applies, rather than a generic message that would be
            # misleading for the first two cases.
            frame_q = getattr(cally, 'frame_quality', None)
            epsf_q = getattr(cally, 'epsf_quality', None)
            if frame_q is not None and frame_q.verdict == 'fail':
                reason = f'frame quality check failed: {"; ".join(frame_q.reasons)}'
            elif epsf_q is not None and epsf_q.verdict == 'fail':
                reason = f'ePSF build failed: {"; ".join(epsf_q.reasons)}'
            else:
                reason = 'no zeropoint result (too few calibration stars after matching)'
            logger.warning(f'{filename}: calibration failed ({reason})')
            record_failure(save_location, _STAGE_CALIBRATION, filename, reason)
            _manifest.record_stage(save_location, _STAGE_CALIBRATION, 'failed', input_path=filename, reason=reason)
            return None

        if cally.calibration_df is None or len(cally.calibration_df) == 0:
            reason = 'no usable calibration table'
            logger.warning(f'{filename}: calibration produced {reason}')
            record_failure(save_location, _STAGE_CALIBRATION, filename, reason)
            _manifest.record_stage(save_location, _STAGE_CALIBRATION, 'failed', input_path=filename, reason=reason)
            return None

        with fits.open(filename) as hdul:
            header = hdul[0].header.copy()

        # IMPORTANT: use cally.data, not a fresh re-read of `filename`.
        # cal_photom's __init__ may have RECAST the pixel data onto a
        # uniform ZP=25 scale (rescale=True, the default -- see
        # calibration_saurus.cal_photom's docstring) via
        # ZP_correction()/Recast_image_scale(). Re-reading the original
        # file here would discard that recast and write back the
        # un-rescaled pixels under a header that claims a uniform ZP=25,
        # so cally.data must be used to keep the pixels and header
        # consistent.
        data = cally.data

        for key in ('COMMENT', 'HISTORY'):
            if key in header:
                header.remove(key, remove_all=True)

        cally.calibration_df.to_csv(phot_new_path, index=False)

        # Always save the zeropoint surface to the zp/ folder, regardless
        # of whether it was actually applied to the image (rescale=True)
        # or only computed as a diagnostic -- either way, self.zp_surface
        # holds the exact array used for the ptp/rms/SETBKG numbers above
        # and is what load_zeropoint_surface()/interpolate_zeropoint_surface()
        # in calibration_saurus.py expect to read back later.
        zp_surface = getattr(cally, 'zp_surface', None)
        if zp_surface is not None:
            np.save(zp_new_path, zp_surface)
        else:
            logger.info(f'{filename}: no zp_surface to save (assess_spatial_variation=False '
                        f'and rescale=False, or too few spatially-usable calibration stars)')

        header['DATECAL'] = Time.now().isot

        # If rescale=True (default) actually ran, the SAVED pixel data
        # (cally.data, used above) is on a uniform ZP=25 scale -- the
        # header's ZP must reflect THAT scale, not the original
        # per-frame zp_median the calibration fit produced BEFORE the
        # recast, or ZP would no longer describe the pixels it's
        # sitting next to. The original value is preserved separately
        # as ZP_INIT for traceability/provenance. ZP_ERR is untouched
        # either way -- a uniform multiplicative rescale doesn't change
        # the RELATIVE precision of the calibration, only its absolute
        # scale.
        zp_value = cally.zeropoint_results['zp_median']
        recast_newzp = getattr(cally, 'recast_newzp', None)
        if getattr(cally, 'rescale', False) and recast_newzp is not None:
            _safe_header_set(header, 'ZP_INIT', zp_value,
                              'Original photometric zeropoint before ZP=25 recast')
            zp_value = recast_newzp

        _safe_header_set(header, 'ZP', zp_value, 'Photometric zeropoint')
        _safe_header_set(header, 'ZP_ERR', cally.zeropoint_results['zp_err'], 'Zeropoint uncertainty')
        _safe_header_set(header, 'ZP_FLOOR', cally.zeropoint_results['zp_floor'], 'Zeropoint floor')
        _safe_header_set(header, 'ZP_NEFF', cally.zeropoint_results['N_eff'], 'Calibration stars used')
        _safe_header_set(header, 'MAGLIM3', cally.maglim3, '3-sigma magnitude limit')
        _safe_header_set(header, 'MAGLIM5', cally.maglim5, '5-sigma magnitude limit')

        ap_corr, ap_corr_err = getattr(cally, 'aperture_correction', (np.nan, np.nan))
        _safe_header_set(header, 'APCORR', ap_corr, 'PSF-to-aperture flux correction factor')
        _safe_header_set(header, 'APCORERR', ap_corr_err, 'Aperture correction scatter')
        _safe_header_set(header, 'ERRSCALE', getattr(cally, 'psf_error_inflation_scale', np.nan),
                          'Empirical PSF flux-error inflation factor')
        _safe_header_set(header, 'MEDCNTAM', getattr(cally, 'median_contam_frac', np.nan),
                          'Median estimated flux-contamination fraction (deblending)')

        # Spatial zeropoint variation diagnostic (item 2, option b -- see
        # calibration_saurus.cal_photom's assess_spatial_variation
        # docstring). ZP itself (written above) remains the correct
        # single value for general use; these are QC numbers only, plus
        # a compact planar fit for the rare case a specific position
        # needs a better-than-average local zeropoint (see
        # cal_photom.get_local_zeropoint).
        _safe_header_set(header, 'ZPSURPTP', getattr(cally, 'zp_surface_ptp', np.nan),
                          'Peak-to-peak spatial ZP variation (mag)')
        _safe_header_set(header, 'ZPSURRMS', getattr(cally, 'zp_surface_rms', np.nan),
                          'RMS of spatial ZP variation (mag)')
        zp_plane = getattr(cally, 'zp_plane_coeffs', None)
        if zp_plane is not None:
            c0, cx, cy = zp_plane
            _safe_header_set(header, 'ZPPLN0', c0, 'Planar ZP fit: constant term (mag)')
            _safe_header_set(header, 'ZPPLNX', cx, 'Planar ZP fit: x slope (mag/px)')
            _safe_header_set(header, 'ZPPLNY', cy, 'Planar ZP fit: y slope (mag/px)')

        set_bkg = getattr(cally, 'set_background_value', None)
        if set_bkg is not None:
            _safe_header_set(header, 'SETBKG', set_bkg,
                              'Constant offset added after ZP=25 recast (avoids negative flux)')

        redflat_used = getattr(cally, 'redflat_systematic', None)
        header['REDFLUSD'] = (redflat_used is not None,
                               'Whether REDFLAT was propagated into photometric errors')

        # Frame-level and ePSF build quality verdicts (items 3/4) -- so a
        # calibrated frame's own header tells you whether/why it was
        # flagged, without needing to dig through the log file.
        frame_quality = getattr(cally, 'frame_quality', None)
        if frame_quality is not None:
            for key, (value, comment) in frame_quality.as_header_dict().items():
                _safe_header_set(header, key, value, comment)

        epsf_quality = getattr(cally, 'epsf_quality', None)
        if epsf_quality is not None:
            header['EPSFVRD'] = (epsf_quality.verdict, 'ePSF build quality verdict: pass/warn/fail')
            epsf_reasons = '; '.join(epsf_quality.reasons) if epsf_quality.reasons else 'none'
            header['EPSFRSN'] = (epsf_reasons[:68], 'ePSF quality reasons (truncated; see log for full)')
            _epsf_key_map = {
                'n_input_stars': 'EPNINPUT',
                'converged_informational': 'EPCONVRG',
                'n_excluded_stars': 'EPNEXCL',
                'excluded_frac': 'EPEXFRAC',
                'final_center_accuracy_px': 'EPCNTRPX',
                'median_residual_frac': 'EPRESID',
                'max_residual_frac': 'EPRESMAX',
                'oversampling_used': 'EPOVERS',
                'subpixel_coverage_frac': 'EPSUBPX',
            }
            for k, v in epsf_quality.metrics.items():
                key = _epsf_key_map.get(k)
                if key is None:
                    continue  # unmapped metric -- log only, not header (avoids truncation collisions)
                if isinstance(v, float):
                    _safe_header_set(header, key, round(v, 4), f'ePSF metric: {k}')
                else:
                    header[key] = (v, f'ePSF metric: {k}')

        # Provenance (item 7): calibration-stage half. PROVDARK/PROVFLAT
        # were already written at the reduction stage and survive into
        # this header unchanged (only COMMENT/HISTORY are stripped
        # above) -- this adds the calibration-specific inputs/parameters.
        prov = build_provenance_dict(
            sauron_state_filename=getattr(cally, 'sauron_state_filename', None),
            match_tol_px=match_tol_px, isolation_radius_px=isolation_radius_px,
            zp_floor=getattr(cally, 'zp_floor', None),
        )
        for key, (value, comment) in prov.items():
            _safe_header_set(header, key, value, comment)

        header['CALIB'] = (True, 'Data calibrated')
        header['CALSTAT'] = ('SUCCESS', 'Photometric calibration status')
        header['CALPIPE'] = ('cal_photom', 'Calibration pipeline script')

        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(cal_new_path, overwrite=True)
        os.system(f"gzip -f {cal_new_path}")
        clear_failure(save_location, _STAGE_CALIBRATION, filename)
        _manifest.record_stage(save_location, _STAGE_CALIBRATION, 'success',
                                input_path=filename, output_path=str(cal_new_path) + '.gz')
        return None

    except NarrowbandFilterError as e:
        # Expected, permanent limitation for this filter -- not a bug,
        # not transient, will never succeed on retry regardless of
        # anything that changes run to run (the limitation is that
        # calibrimbore cannot calibrate narrowband filters at all, not
        # anything about this specific frame). Logged at INFO rather
        # than WARNING/ERROR since this isn't something to investigate
        # or fix, and always recorded so it's never retried.
        reason = f'narrowband filter, not calibratable via calibrimbore: {e}'
        logger.info(f'{filename}: {reason}')
        record_failure(save_location, _STAGE_CALIBRATION, filename, reason)
        _manifest.record_stage(save_location, _STAGE_CALIBRATION, 'skipped', input_path=filename, reason=reason)
        return None

    except PSFGroupingError as e:
        # Raised when astropy.modeling's CompoundModel evaluation hits
        # Python's recursion limit because IterativePSFPhotometry tried
        # to fit too large a group of stars simultaneously (see
        # psf_photometry.PSFGroupingError's docstring for the full
        # mechanism). cal_photom already tries to prevent this
        # automatically -- shrinking group_min_separation_px down to a
        # 2px floor before ever attempting the fit (see
        # psf_photometry.pick_safe_group_separation) and retrying
        # several times with a smaller separation -- so this exception
        # means even that gave up. This is a stable characteristic of
        # this specific field at current settings (a very dense field
        # may need a smaller max_calibration_stars to avoid oversized
        # groups entirely), not a transient blip, so it's logged at
        # WARNING like the other data-quality gates above, not ERROR.
        reason = f'PSF group-fit failed (astropy.modeling recursion limit, group too large): {e}'
        logger.warning(f'{filename}: {reason}')
        record_failure(save_location, _STAGE_CALIBRATION, filename, reason)
        _manifest.record_stage(save_location, _STAGE_CALIBRATION, 'failed', input_path=filename, reason=reason)
        return None

    except RecursionError as e:
        # A bare RecursionError reaching here (rather than being caught
        # as PSFGroupingError above) means it did not originate from
        # photometry()'s own group-fit attempt -- that path always
        # raises the specific PSFGroupingError instead, once its own
        # backoff retries are exhausted -- so this is genuinely from
        # somewhere else in this frame's processing. The full traceback
        # (attached via exc_info=True below) is worth checking to
        # identify the origin.
        reason = f'recursion error (origin not photometry()\'s group-fit path -- see log for full traceback): {e}'
        logger.warning(f'{filename}: {reason}', exc_info=True)
        record_failure(save_location, _STAGE_CALIBRATION, filename, reason)
        _manifest.record_stage(save_location, _STAGE_CALIBRATION, 'failed', input_path=filename, reason=reason)
        return None

    except SepExtractionError as e:
        # A pixstack overflow even after escalating the buffer is a
        # stable, structural problem with THIS frame (almost always a
        # background-subtraction issue, per sep's own maintainers -- see
        # frame_quality.py's SepExtractionError docstring) -- unlike the
        # transient Gaia-network RuntimeError handled below, retrying
        # this frame bare will not help, so it IS recorded in the
        # failure ledger. Logged at WARNING (not ERROR) since it's a
        # data-quality issue this pipeline is designed to detect and
        # report, not an unexpected code-level failure.
        reason = f'sep extraction failed: {e}'
        logger.warning(f'{filename}: {reason}')
        record_failure(save_location, _STAGE_CALIBRATION, filename, reason)
        _manifest.record_stage(save_location, _STAGE_CALIBRATION, 'failed', input_path=filename, reason=reason)
        return None

    except RuntimeError as e:
        # gaia_query.py raises RuntimeError specifically after exhausting
        # its own retry logic on a network failure (e.g. connection
        # reset, archive timeout) -- this is an expected, occasionally-
        # occurring condition that retrying already tried to absorb, not
        # a pipeline bug. Logged at WARNING (like the other per-frame
        # failure reasons above) rather than ERROR, so it doesn't appear
        # on the console at the default ERROR-only console log level and
        # get conflated with genuine unexpected failures.
        #
        # Deliberately NOT recorded in the failure ledger: a one-off
        # network blip says nothing about whether this frame is
        # processable, unlike "no master dark" or "too few calibration
        # stars" which are genuinely stable until something external
        # changes. Recording it would cause skip_known_failures=True to
        # permanently skip a frame that would likely succeed on the very
        # next run.
        reason = f'calibration step failed (transient, not recorded -- will retry next run): {e}'
        logger.warning(f'{filename}: {reason}')
        # Deliberately NOT recorded in the failure ledger (see comment
        # above), but still logged to the manifest -- unlike the
        # ledger, the manifest is an append-only history rather than a
        # skip-list, so recording a transient hiccup here doesn't cause
        # any future run to skip the frame.
        _manifest.record_stage(save_location, _STAGE_CALIBRATION, 'transient_failed', input_path=filename, reason=reason)
        return None

    except Exception as e:
        reason = f'calibration step failed: {e}'
        logger.error(f'{filename}: {reason}')
        record_failure(save_location, _STAGE_CALIBRATION, filename, reason)
        _manifest.record_stage(save_location, _STAGE_CALIBRATION, 'failed', input_path=filename, reason=reason)
        return None