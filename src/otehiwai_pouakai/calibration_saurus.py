"""
Photometric calibration for astronomical images.

This module takes a science frame (image + WCS) and works out how to convert
raw, instrumental star brightnesses into real, catalog-calibrated magnitudes.
It does this by:

1. Detecting point sources in the frame (DAOFind).
2. Cross-matching them against Gaia and a photometric reference catalog,
   and estimating how much each star's measured flux is contaminated by
   its neighbours (rather than just throwing out anything with a close
   neighbour).
3. Building an effective PSF (ePSF) model from clean, isolated stars and
   using it to measure precise instrumental fluxes for every star,
   including ones in crowded groups.
4. Predicting each matched star's catalog magnitude with `calibrimbore`
   (`sauron`), which already applies its own colour term and extinction
   correction internally.
5. Fitting a zeropoint (instrumental mag -> catalog mag offset) from those
   matches, using an inverse-variance-weighted, iteratively sigma-clipped
   estimator so faint/noisy stars don't get the same say as bright,
   well-measured ones.
6. Optionally fitting a smooth, position-dependent zeropoint surface across
   the frame and rescaling the image onto a uniform photometric zeropoint
   (ZP=25), while still keeping the per-position zeropoint information
   available for later use (`get_local_zeropoint` / `load_zeropoint_surface`
   + `interpolate_zeropoint_surface`).

The main entry point is the `cal_photom` class -- construct it with an
image (via `file=` or `data=`/`wcs=`/`header=`) and it runs the full
detect -> match -> PSF-fit -> calibrate pipeline. See its docstring for
the available options (aperture-correction, PSF error inflation, crowding
controls, whether to rescale the saved image, etc.).

Note on narrowband filters: calibrimbore only has synthetic-photometry
models for broadband filters (decam/skymapper/lsst/ps1-family). Narrowband
emission-line filters (Halpha, OIII, SII, etc.) cannot be calibrated this
way, since their flux doesn't vary smoothly with stellar colour the way
broadband flux does -- see `NarrowbandFilterError` below.
"""

import numpy as np
import matplotlib.pyplot as plt

# Imported before calibrimbore/pysynphot below on purpose: config.py
# sets a PYSYN_CDBS default (os.environ.setdefault) as an import-time
# side effect, and pysynphot reads PYSYN_CDBS from the environment the
# moment IT is imported -- so config needs to run first for that
# default to be in place in time. See config.py's module docstring.
from . import config

from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clip, mad_std, sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy.visualization import SqrtStretch, simple_norm, ImageNormalize
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.time import Time
from astropy.table import QTable

import astroscrappy

from photutils.detection import DAOStarFinder
from photutils.aperture import (SkyCircularAperture, CircularAperture,
                                 CircularAnnulus, aperture_photometry, ApertureStats)

from scipy.ndimage import convolve, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.optimize import minimize

from calibrimbore import sauron, get_skymapper_region, get_ps1_region
from .gaia_query import get_gaia_region, gaia_cone
from .psf_photometry import (build_epsf, build_epsf_adaptive, photometry, do_aperture_photometry,
                             compute_aperture_correction, inflate_psf_errors,
                             mag_error, local_flux_contamination, PSFGroupingError)
from .frame_quality import sep_extract_sources, assess_frame_quality, mask_crowded_regions
from .provenance import build_provenance_dict

from copy import deepcopy
import warnings
import os
import logging

logger = logging.getLogger(__name__)

logging.getLogger('astroquery').setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings('ignore')

# Resolved from POUAKAI_CAL_FILES_DIR (or this site's shared-storage
# default) -- see config.py. `config` itself was imported at the top of
# this file, ahead of calibrimbore/pysynphot -- see the comment there.
cal_files_location = config.cal_files_dir()

# Recognised filter -> calibrimbore band-name mapping. Add an entry here to
# support calibrating a new filter. Anything not listed (or listed in
# _NARROWBAND_FILTERS below) raises a FilterMappingError from `_get_filter`
# instead of calibrating against the wrong band.
_FILTER_MAP = {
    'V': 'bessell_V',
    'I': 'bessell_I',
    'Blue': 'bessell_B',
    'Red': 'bessell_R',
    'g': 'sloan_g',
    'r': 'sloan_r',
    'i': 'sloan_i',
    'z': 'sloan_z',
}

# Narrowband emission-line filters this pipeline's FITS headers can carry.
# These are intentionally NOT in _FILTER_MAP: calibrimbore only ships
# synthetic photometry for broadband filters (decam/skymapper/lsst/ps1
# families), and narrowband flux is driven by emission-line strength
# rather than varying smoothly with broadband stellar colour, so
# calibrimbore's approach doesn't apply here. If you need photometric
# calibration for narrowband imaging, use a dedicated narrowband
# standard-star calibration method instead -- this pipeline will raise
# NarrowbandFilterError rather than attempt it.
_NARROWBAND_FILTERS = {'Halpha', 'OIII', 'SII', 'Methane', 'NII', 'SIII', 'Hbeta'}


class FilterMappingError(ValueError):
    """Raised when a FITS FILTER header value has no known calibration mapping."""


class NarrowbandFilterError(FilterMappingError):
    """
    Raised specifically for narrowband emission-line filters (Halpha,
    OIII, SII, etc.), which cannot be calibrated by this pipeline's
    broadband, colour-based method. Subclasses FilterMappingError, so
    existing `except FilterMappingError` handlers still catch it, while
    callers that want to treat "narrowband, not supported" differently
    from "unrecognised filter string" can catch this type specifically.
    """
    pass


def interpolate_zeropoint_surface(zp_surface, x, y, order=1):
    """
    Interpolate a fitted zeropoint surface at pixel position(s) (x, y).

    Standalone -- works on a plain 2D array, independent of any live
    `cal_photom` object. This is exactly the function you need to
    recover a position-dependent zeropoint from a SAVED surface (see
    `load_zeropoint_surface`) after the pipeline run that produced it
    has finished and the `cal_photom` instance no longer exists --
    e.g. for a calibrated frame you're revisiting later.

    Parameters
    ----------
    zp_surface : 2D ndarray
        A zeropoint surface as produced by `cal_photom.Fit_surface` /
        `ZP_correction` (same shape as the science frame it was fit to).
    x, y : float or array_like
        Pixel position(s) to interpolate at. Fractional (sub-pixel)
        values are fine -- that's the point of interpolating rather than
        indexing.
    order : int
        Spline order for `scipy.ndimage.map_coordinates` -- 1 (default)
        is bilinear, smooth and robust; use 0 for nearest-neighbour, or
        higher for a smoother (but potentially ringing, near sharp
        surface features) interpolation.

    Returns
    -------
    float or ndarray -- matches the shape of the input x/y.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    # map_coordinates indexes as (row, col) = (y, x), and clips to the
    # nearest valid pixel at the edges rather than extrapolating wildly.
    coords = np.vstack([y, x])
    values = map_coordinates(zp_surface, coords, order=order, mode='nearest')
    return float(values[0]) if values.size == 1 else values


def load_zeropoint_surface(path):
    """
    Load a zeropoint surface saved by core_reduction.calibrating_internal
    (one `.npy` file per calibrated frame, under `<save_location>/zp/`).
    Pass the result straight to `interpolate_zeropoint_surface`.
    """
    return np.load(path)


class cal_photom():
    def __init__(self, file=None, data=None, wcs=None, mask=None,
                 header=None, ax=None, threshold=10.0, run=True,
                 cal_model='ckmodel', rescale=True,
                 plot=True, floor=None, radius_override=None, use_catalogue=True,
                 band_override=None, zp_floor=0.02,
                 isolation_radius_px=21.0, match_tol_px=2.5,
                 max_contamination_frac=0.05,
                 max_calibration_stars=150,
                 group_min_separation_px=None,
                 group_min_separation_fwhm_factor=2.0,
                 max_group_size=25,
                 use_grouping=True,
                 psf_error_inflation_max_scale=8.0,
                 epsf_sampling_candidates=(3, 2),
                 assess_spatial_variation=True,
                 pixel_scale_arcsec=0.72):
        '''
        Parameters
        ----------
        rescale : bool
            If True (default), fit a smooth zeropoint surface and
            rescale `self.data` onto a uniform photometric scale of
            ZP=25 everywhere (`ZP_correction` + `Recast_image_scale`) --
            this is what actually gets written to the saved calibrated
            FITS file. The fitted surface itself is also saved
            separately (see core_reduction.calibrating_internal's
            `zp/` output) and can be interpolated at any position via
            `get_local_zeropoint` / `interpolate_zeropoint_surface`, so
            the per-position information isn't lost even though the
            saved image itself is flattened to one scale. Set False to
            skip the recast and keep `self.data` at its native
            instrumental scale (the zeropoint surface diagnostics below
            still run, just without being applied to the image).
        zp_floor : float
            Irreducible systematic floor (mag) added in quadrature to the
            formal zeropoint uncertainty.
        isolation_radius_px : float
            Max search radius (px) for the flux-contamination estimate
            (see `matching_sources`) -- NOT a hard exclusion radius
            anymore. Stars farther apart than this are never considered
            as mutual contaminants at all, since a Gaussian PSF's
            contribution is already negligible well before typical
            values of this parameter.
        max_contamination_frac : float
            A Gaia-matched star is usable for calibration if the
            estimated fraction of its own flux contaminated by nearby
            neighbours (see `matching_sources`) is below this. Relaxed
            automatically, up to a ceiling, if too few stars survive.
        max_calibration_stars : int or None
            Cap on how many usable stars are actually fed into ePSF
            building / IterativePSFPhotometry (see `matching_sources`).
            A well-chosen ~100-200 bright, clean stars gives a
            statistically excellent zeropoint -- feeding in every last
            usable star (which can be several hundred in a rich field)
            gains little additional precision but risks a much larger,
            harder-to-fit simultaneous multi-star PSF group solve, which
            empirically produced wildly unstable per-star flux
            measurements (see the aperture-correction sanity bound in
            psf_photometry.compute_aperture_correction for the symptom
            this was added to address). None disables the cap.
        pixel_scale_arcsec : float
            Plate scale, used only for converting FWHM into arcsec for
            frame-level quality-gate reporting/thresholds
            (frame_quality.assess_frame_quality). Default matches this
            instrument's 0.72"/px; update if reused with different optics.
        match_tol_px : float
            Maximum pixel distance for nearest-neighbour matching steps
            (Gaia<->detection, and PSF-fit<->detection). Default of 2.5px
            reflects typical combined WCS-solution residual (often
            ~0.3-0.7px for a good solve, worse for noisier frames) plus
            DAOFind centroid scatter (~0.1-0.3px) -- a tolerance much
            tighter than this (e.g. 1.0px, the previous default) rejects
            many genuine matches purely from positional noise rather than
            an actual mismatch, which manifests as widespread "too few
            calibration stars" failures unrelated to the underlying data
            quality. If you are still seeing many such failures with this
            default, check REDFLAT/zp_sigma trends first -- it may indicate
            a genuine WCS quality problem rather than a tolerance issue.
        epsf_sampling_candidates : sequence of int
            ePSF oversampling factors to try, in order of preference,
            via psf_photometry.build_epsf_adaptive -- e.g. (3, 2) tries
            3x oversampling first and falls back to 2x (with a logged
            warning) only if this frame's calibration star sample can't
            support it (too few stars, or their sub-pixel centroid
            offsets don't spread out enough across the finer grid --
            see build_epsf_adaptive's docstring). Set to a single-value
            tuple, e.g. (2,), to always use a fixed oversampling with no
            adaptive fallback (the previous behaviour).
        assess_spatial_variation : bool
            If True (default), fit a diagnostic-only spatial zeropoint
            surface after a successful calibration (does NOT touch
            self.data, runs independently of `rescale`) -- the "0th-
            order capture" of spatially-varying PSF/flat-field/
            vignetting response: rather than modelling PSF SHAPE
            variation directly, the per-star zeropoint residual (which
            already reflects whatever the ACTUAL local response
            delivered for that star) is fit as a smooth 2D surface, and
            its amplitude reported as `self.zp_surface_ptp` /
            `self.zp_surface_rms` (mag). A planar (tip/tilt) fit is
            also stored as `self.zp_plane_coeffs = (c0, cx, cy)` -- a
            compact, 3-number way to apply a first-order position-
            dependent correction without shipping a full grid.

            IMPORTANT -- what to actually RECORD as "the" zeropoint for
            general use: `self.zeropoint_results['zp_median']` (written
            to the ZP header key) remains the right single scalar value
            for general use, unchanged by any of this -- it's already
            the robust, sigma-clipped, position-independent central
            estimate, i.e. the 0th-order term of the surface itself.
            zp_surface_ptp/rms exist so you can SEE whether that single
            value is good enough for a given frame (small ptp relative
            to zp_err -> yes, just use ZP everywhere) or whether it's
            worth applying the planar correction for specific
            high-precision positions (large ptp -> use
            get_local_zeropoint(x, y) instead of the bare header ZP for
            those). Most frames should not need the latter.
        psf_error_inflation_max_scale : float
            Sanity ceiling passed to psf_photometry.inflate_psf_errors.
            That function empirically inflates PSF flux errors to match
            observed scatter against aperture photometry on high-SNR,
            isolated stars, since the formal PSF-fit covariance is
            typically optimistic. The default (8.0) is based on testing
            across a dense, cluster-adjacent field and a sparse general
            field; if your PSF errors are consistently getting capped at
            this ceiling, that's worth checking against
            calibration_diagnostics.py's `--full` trace, as it may mean
            this field needs a different scale.
        '''
        self.file = file
        self.data = data
        self.wcs = wcs
        self.header = header
        self.mask = mask
        self.hdu = None
        self.band = None
        self.image_floor = floor
        self.radius_override = radius_override
        self.use_catalogue = use_catalogue
        self._band_override = band_override
        self.sources = None
        self.cal_sys = None
        self.cal_model = cal_model.lower()
        self.sauron = None
        self.zps = None
        self.phot_table = None
        self.rescale = rescale
        self.zp_floor = zp_floor
        self.isolation_radius_px = isolation_radius_px
        self.match_tol_px = match_tol_px
        self.max_contamination_frac = max_contamination_frac
        self.max_calibration_stars = max_calibration_stars
        self.group_min_separation_px = group_min_separation_px
        self.group_min_separation_fwhm_factor = group_min_separation_fwhm_factor
        self.max_group_size = max_group_size
        self.use_grouping = use_grouping
        self.psf_error_inflation_max_scale = psf_error_inflation_max_scale
        self.epsf_sampling_candidates = tuple(epsf_sampling_candidates)
        self.assess_spatial_variation = assess_spatial_variation
        self.zp_surface_ptp = np.nan
        self.zp_surface_rms = np.nan
        self.zp_plane_coeffs = None
        self.zp_surface = None
        self.set_background_value = None
        self.recast_newzp = None
        self._pixel_scale_arcsec = pixel_scale_arcsec
        self.zeropoint_results = None
        self.epsf_quality = None
        self.frame_quality = None
        self.median_contam_frac = np.nan

        if run:
            self._load_image()
            self._clean_cosmics()
            self._starfinding()

            # Frame-level quality gate (item 3): assessed immediately
            # after detection, before any of the expensive downstream
            # work (Gaia query, ePSF build, iterative PSF photometry) --
            # a fail-verdict frame stops here rather than spending that
            # work on a frame already known to be degraded (cloud,
            # tracking failure, severe guiding error). Self-contained per
            # frame: no comparison against this field's own history is
            # made (this pipeline serves a multi-purpose pointed
            # observing program, not a survey with guaranteed repeat
            # visits -- see frame_quality.py's module docstring).
            self.frame_quality = assess_frame_quality(
                self.sources, self.background_rms,
                redflat_systematic=self.header.get('REDFLAT', None),
                pixel_scale_arcsec=self._pixel_scale_arcsec,
            )
            if self.frame_quality.verdict == 'fail':
                reasons = '; '.join(self.frame_quality.reasons)
                logger.warning(f'Frame quality check failed; aborting calibration. Reasons: {reasons}')
                self.zeropoint_results = None
                self.calibration_df = None
                return
            elif self.frame_quality.verdict == 'warn':
                logger.warning(f'Frame quality warning(s): {"; ".join(self.frame_quality.reasons)}')

            # This frame's actual measured PSF FWHM, computed once here and
            # reused for every aperture-photometry/aperture-correction call
            # below (falls back to 3.2px only if frame-quality metrics are
            # unavailable). Using the real per-frame FWHM keeps aperture
            # radii derived from it -- including
            # compute_aperture_correction's r_large=r_large_factor*fwhm --
            # correctly sized to this frame's actual PSF width; an
            # oversized aperture would pick up nearby stars' flux and
            # extra background noise, both of which bias the PSF-to-
            # aperture flux ratio that function computes.
            self.measured_fwhm_px = (self.frame_quality.metrics.get('fwhm_px', 3.2)
                                      if self.frame_quality is not None else 3.2)

            # REDFLAT is written by core_reduction.reduction_script as a
            # per-frame diagnostic of background-subtraction quality (std
            # of per-tile residual sky after subtraction, in flux units).
            # Propagated into the photometric error budget below via
            # do_aperture_photometry's redflat_systematic parameter --
            # see that function's docstring for why this is a distinct
            # error term from the per-annulus local sky scatter it
            # already includes. Falls back to None (no extra term) if
            # REDFLAT is absent or non-finite, e.g. for an older reduced
            # frame that predates this header key.
            self.redflat_systematic = self.header.get('REDFLAT', None)
            if self.redflat_systematic is not None and not np.isfinite(self.redflat_systematic):
                self.redflat_systematic = None

            _, _, snr, _ = do_aperture_photometry(
                self.data, self.positions, fwhm=self.measured_fwhm_px,
                redflat_systematic=self.redflat_systematic,
            )

            self.create_saturation_mask()

            snr_mask = snr > 10
            keep = snr_mask & self.sat_mask

            self.positions = self.positions[keep]
            self.sources = self.sources[keep]
            # `ref_id` is the single identity that survives every later
            # filtering/matching/re-indexing step below.
            self.sources['ref_id'] = np.arange(len(self.sources))

            self.catalogue_sources()
            self.matching_sources(max_contamination_frac=self.max_contamination_frac,
                                   max_calibration_stars=self.max_calibration_stars)

            if len(self.sources) < 5:
                logger.warning('Too few isolated, catalog-matched stars after matching_sources(); aborting calibration.')
                self.zeropoint_results = None
                self.calibration_df = None
                return

            epsf_data, epsf, epsf_quality = build_epsf_adaptive(
                self.data, self.sources, sampling_candidates=self.epsf_sampling_candidates,
            )
            self.epsf_quality = epsf_quality

            if epsf is None:
                reasons = '; '.join(epsf_quality.reasons) if epsf_quality.reasons else 'unknown reason'
                logger.warning(f'ePSF build failed; aborting calibration. Reasons: {reasons}')
                self.zeropoint_results = None
                self.calibration_df = None
                return

            if epsf_quality.verdict == 'warn':
                logger.warning(f'ePSF build quality warning(s): {"; ".join(epsf_quality.reasons)}')

            # PSF photometry on exactly the matched/isolated star positions
            # (not the full re-detected source list), so identity carries
            # through cleanly.
            matched_positions = np.column_stack(
                (self.sources['x'], self.sources['y'])
            )

            # group_min_separation_px (see __init__ docstring) controls
            # how large a simultaneous multi-star fit group can get --
            # exposed for tuning per target type rather than fixed
            # pipeline-wide (see calibration_diagnostics.py's grouping-
            # stability sweep).
            if self.group_min_separation_px is None:
                resolved_min_separation = self.measured_fwhm_px * self.group_min_separation_fwhm_factor
                logger.info(f'Deriving group_min_separation_px from measured FWHM: '
                            f'{self.measured_fwhm_px:.2f}px * {self.group_min_separation_fwhm_factor} '
                            f'= {resolved_min_separation:.2f}px')
            else:
                resolved_min_separation = self.group_min_separation_px

            result_table = photometry(
                self.data, epsf, matched_positions, self.daofind,
                progress_bar=False, max_iter=30, tol=1e-4, size=11,
                min_separation=resolved_min_separation, fwhm=self.measured_fwhm_px,
                max_group_size=self.max_group_size, use_grouping=self.use_grouping,
            )
            result_table = result_table.to_pandas()

            # IterativePSFPhotometry's output order corresponds to the
            # input init_params order for single-iteration, non-grouped
            # fits; we still re-match by position (with a tight tolerance)
            # rather than assume row order survives, since grouping/
            # iteration can reorder or drop rows.
            result_table = self._attach_ref_id(result_table, self.sources, tol=self.match_tol_px)
            result_table = result_table[np.isfinite(result_table['ref_id'])].copy()
            result_table['ref_id'] = result_table['ref_id'].astype(int)

            psf_positions = result_table[['x_fit', 'y_fit']].values

            flux_ap, err_ap, snr_ap, bkg_term_psf = do_aperture_photometry(
                self.data, psf_positions, fwhm=self.measured_fwhm_px,
                redflat_systematic=self.redflat_systematic,
            )

            result_table['flux_err'] = np.sqrt(
                bkg_term_psf**2 + result_table['flux_err'].values**2
            )
            result_table['flux_ap'] = flux_ap
            result_table['flux_ap_err'] = err_ap

            # Aperture correction from high-SNR, ISOLATED stars only (see
            # compute_aperture_correction's docstring). Restricting to
            # isolated stars avoids an oversized/mis-sized aperture
            # picking up neighbour flux or excess background noise, which
            # would otherwise produce unreliable correction factors even
            # on ostensibly "sparse" fields.
            psf_snr_tmp = result_table['flux_fit'].values / result_table['flux_err'].values
            corr, corr_err = compute_aperture_correction(
                self.data, psf_positions, result_table['flux_fit'].values,
                fwhm=self.measured_fwhm_px, redflat_systematic=self.redflat_systematic,
            )
            if np.isfinite(corr) and corr > 0:
                result_table['flux_fit_corr'] = result_table['flux_fit'] * corr
                result_table['flux_err_corr'] = result_table['flux_err'] * corr
                self.aperture_correction = (corr, corr_err)
            else:
                result_table['flux_fit_corr'] = result_table['flux_fit']
                result_table['flux_err_corr'] = result_table['flux_err']
                self.aperture_correction = (1.0, np.nan)
                logger.warning('Aperture correction failed (non-finite/non-positive); using uncorrected PSF flux.')

            # Empirically inflate PSF flux errors against the aperture
            # measurement on high-SNR, ISOLATED stars. Formal PSF-fit
            # covariance is typically optimistic, so zp_err would
            # understate the real uncertainty without this step. The same
            # isolation gate used for compute_aperture_correction above
            # applies here too: comparing PSF vs aperture flux for a star
            # with a real, uncorrected neighbour would measure neighbour
            # contamination rather than the PSF fit's actual precision.
            error_scale = inflate_psf_errors(
                result_table, psf_flux_col='flux_fit_corr',
                psf_err_col='flux_err_corr', ap_flux_col='flux_ap', min_snr=10,
                fwhm=self.measured_fwhm_px, max_scale=self.psf_error_inflation_max_scale,
            )
            self.psf_error_inflation_scale = error_scale

            # ------------------------------------------------------------
            # Flux DEBLENDING: using the now fully-measured, aperture-
            # corrected instrumental flux of EVERY fitted source in the
            # frame (not just the calibration candidates -- result_table
            # at this point still holds every star IterativePSFPhotometry
            # fit), estimate how much of each candidate's own flux is
            # actual measured contamination from real nearby sources, and
            # subtract it before computing sysmag. This is the
            # "theoretical cumulative flux vs actual cumulative flux"
            # correction: theoretical = flux_fit_corr (what got measured,
            # including any neighbour leakage); actual = flux_fit_deblend
            # (theoretical minus the modelled neighbour contribution).
            #
            # Unlike matching_sources' contamination estimate (which uses
            # Gaia G magnitude as a brightness PROXY, since PSF flux
            # doesn't exist yet at that stage), this uses REAL measured
            # instrumental fluxes in the pipeline's own photometric
            # system -- a more accurate correction now that photometry
            # has actually been done.
            fwhm_px = (self.frame_quality.metrics.get('fwhm_px', 3.5)
                       if self.frame_quality is not None else 3.5)
            all_xy = result_table[['x_fit', 'y_fit']].values
            all_flux = result_table['flux_fit_corr'].values

            contam_flux, contam_frac = local_flux_contamination(
                all_xy, all_flux, all_xy, all_flux, fwhm_px=fwhm_px,
                max_radius_px=self.isolation_radius_px,
            )
            result_table['contam_flux'] = contam_flux
            result_table['contam_frac'] = contam_frac

            deblended_flux = result_table['flux_fit_corr'].values - contam_flux
            # A correction that would drive flux non-positive means the
            # model has no reliable signal to recover -- fall back to the
            # uncorrected flux rather than propagate a nonsensical
            # negative/zero value into -2.5*log10 below.
            bad_deblend = ~np.isfinite(deblended_flux) | (deblended_flux <= 0)
            deblended_flux[bad_deblend] = result_table['flux_fit_corr'].values[bad_deblend]
            result_table['flux_fit_deblend'] = deblended_flux

            # The correction's own uncertainty (deliberately conservative:
            # 30% of the correction itself) is folded into the flux error
            # in quadrature, so a star that got a large correction also
            # gets an appropriately larger error -- it doesn't silently
            # look as precise as an uncontaminated star of the same
            # brightness.
            result_table['flux_err_corr'] = np.sqrt(
                result_table['flux_err_corr'].values**2 + (0.3 * contam_flux) ** 2
            )

            self.median_contam_frac = float(np.nanmedian(contam_frac)) if len(contam_frac) else np.nan
            n_meaningfully_contaminated = int((contam_frac > 0.01).sum())
            if n_meaningfully_contaminated:
                logger.info(f'Flux deblending: {n_meaningfully_contaminated}/{len(result_table)} '
                            f'detected sources had >1% estimated flux contamination from '
                            f'neighbours; corrected before computing sysmag (fwhm={fwhm_px:.2f}px)')
            # ------------------------------------------------------------

            result_table['snr'] = (
                result_table['flux_fit_deblend'] / result_table['flux_err_corr']
            )

            filt_result_table = result_table[(result_table['snr'] > 10)].copy()
            filt_result_table = filt_result_table.reset_index(drop=True)

            calibration_df = self._calibration_photometry(filt_result_table)

            if calibration_df is None or len(calibration_df) < 5:
                logger.warning('Too few calibration stars after matching; aborting calibration.')
                self.zeropoint_results = None
                self.calibration_df = calibration_df
                return

            calibration_df['sysmag'] = -2.5 * np.log10(calibration_df['flux_fit_deblend'])
            calibration_df['sysmag_err'] = mag_error(
                calibration_df['flux_fit_deblend'].values,
                calibration_df['flux_err_corr'].values,
                0.0,
            )

            self.calibration_df = calibration_df

            self._load_sauron()
            self.predict_mags()
            self.estimate_zeropoint(mag_limit=19.0, sigma=3.0, maxiters=5, zp_floor=self.zp_floor)

            if self.zeropoint_results is not None:
                self.magnitude_limit(snr_lim=threshold)

                self.calibration_df['ra'] = self.ra
                self.calibration_df['dec'] = self.dec
                self.calibration_df['mag'] = (
                    self.calibration_df['sysmag'].values + self.zeropoint_results['zp_median']
                )
                self.calibration_df['mag_err'] = mag_error(
                    calibration_df['flux_fit_deblend'].values,
                    calibration_df['flux_err_corr'].values,
                    self.zeropoint_results['zp_err'],
                )
                self.calibration_df['zp_err'] = self.zeropoint_results['zp_err']
                self.calibration_df['zp_sigma'] = self.zeropoint_results['zp_sigma']
                self.calibration_df['zp_median'] = self.zeropoint_results['zp_median']
                self.calibration_df['zp_Neff'] = self.zeropoint_results['N_eff']
                self.calibration_df['maglim3'] = self.maglim3
                self.calibration_df['maglim5'] = self.maglim5

                self.calibration_df = self.calibration_df[
                    np.isfinite(self.calibration_df['mag'].values)
                ].copy()
                self.calibration_df = self.calibration_df.reset_index(drop=True)

                # rescale=True (default) applies the actual recast --
                # self.data becomes uniform ZP=25 everywhere -- via the
                # fitted spatial ZP surface (ZP_correction/
                # Recast_image_scale). Run BEFORE the diagnostic below so
                # it can reuse this exact surface (self.zp_surface)
                # rather than fitting a second, separate one.
                if self.rescale:
                    self.ZP_correction()
                    self.Recast_image_scale()

                # Spatial zeropoint variation summary (ptp/rms) and a
                # compact planar fit -- see assess_spatial_zeropoint_
                # variation's and __init__'s assess_spatial_variation
                # docstrings for the full reasoning. Reuses self.zp_surface
                # from the rescale step above if it ran; otherwise fits
                # its own (diagnostic-only, self.data left untouched).
                if self.assess_spatial_variation:
                    self.assess_spatial_zeropoint_variation()

    # ------------------------------------------------------------------
    # Identity-preserving matching helpers
    # ------------------------------------------------------------------

    def _attach_ref_id(self, table, ref_sources, tol=1.0):
        """
        Match rows of `table` (with x_fit/y_fit columns) back to
        `ref_sources` (with x/y/ref_id columns) by nearest neighbour within
        `tol` pixels. Adds a `ref_id` column (NaN if unmatched).
        """
        ref_xy = np.column_stack((np.asarray(ref_sources['x']), np.asarray(ref_sources['y'])))
        ref_ids = np.asarray(ref_sources['ref_id'])

        query_xy = table[['x_fit', 'y_fit']].values
        tree = cKDTree(ref_xy)
        dist, idx = tree.query(query_xy, distance_upper_bound=tol)

        valid = np.isfinite(dist) & (idx < len(ref_xy))

        table = table.copy()
        table['ref_id'] = np.nan
        table.loc[valid, 'ref_id'] = ref_ids[idx[valid]]
        return table

    # ------------------------------------------------------------------
    # Image loading / source finding
    # ------------------------------------------------------------------

    def _load_image(self):
        if self.file is not None:
            self.hdu = fits.open(self.file)[0]
            self.header = self.hdu.header
            self.data = self.hdu.data
            self.wcs = WCS(self.header)
            self._get_filter()
        else:
            self._get_filter()

    def _get_filter(self):
        if self._band_override is not None:
            self.band = self._band_override
            return

        raw = self.header['FILTER'].strip(' ')

        if raw in _FILTER_MAP:
            self.band = _FILTER_MAP[raw]
            return

        if raw in _NARROWBAND_FILTERS:
            raise NarrowbandFilterError(
                f"Filter header value {raw!r} is a narrowband emission-line filter. "
                "calibrimbore has no bundled filter-response data or synthetic-photometry "
                "model for any narrowband filter (confirmed against its package data -- only "
                "broadband decam/skymapper/lsst/ps1 bands are supported), and narrowband flux "
                "is driven by emission-line strength rather than varying smoothly with "
                "broadband stellar colour, so this pipeline's calibrimbore-based zeropoint "
                "method is not physically applicable to this filter. Narrowband photometric "
                "calibration needs a different method (e.g. dedicated narrowband standard "
                "stars), not a calibrimbore band mapping."
            )

        raise FilterMappingError(
            f"Filter header value {raw!r} has no entry in _FILTER_MAP. "
            "Add an entry for it there before calibrating this filter."
        )

    def _starfinding(self):
        sources, positions, background_rms, crowd_frac = sep_extract_sources(
            self.data, thresh_sigma=5.0, minarea=5,
        )
        self.background_rms = background_rms
        self.crowd_frac = crowd_frac
        if crowd_frac > 0:
            logger.info(f'{self.file}: excluded {crowd_frac:.1%} of frame as crowded/core '
                        f'region before source detection')

        _, _, std = sigma_clipped_stats(self.data, sigma=3)
        self.daofind = DAOStarFinder(fwhm=4, threshold=10.0 * std)

        self.positions = positions
        self.sources = sources

    def matching_sources(self, min_isolated_target=15,
                          max_contamination_frac=0.05,
                          contamination_frac_ceiling=0.5,
                          contamination_frac_step=1.6,
                          max_calibration_stars=150):
        """
        Cross-match detected sources to Gaia positions and select a
        "clean" calibration subset using a CONTINUOUS, PSF-aware
        contamination estimate rather than a hard isolation-radius
        binary cutoff.

        How the contamination filter works
        --------------------------------------------------------------------
        Rather than applying a single fixed exclusion radius around every
        Gaia-matched star, this estimates a `contamination_frac` for each
        one (via `psf_photometry.local_flux_contamination`) -- the
        fraction of its own flux that a Gaussian PSF model (width = this
        frame's actual measured FWHM, from frame_quality's
        assess_frame_quality) predicts is leaking in from every OTHER
        matched star, using Gaia G magnitude as a relative flux proxy (no
        instrumental flux exists yet at this stage -- PSF photometry
        hasn't run). A star is kept if contamination_frac is below
        `max_contamination_frac` (default 5%). This naturally adapts to
        both the frame's actual PSF width (tighter pairs resolve cleanly
        in good seeing) and each star's brightness (a faint star near a
        bright one isn't a real contamination concern, and vice versa),
        so bright, useful calibration stars in crowded fields aren't
        discarded just for having any neighbour nearby.

        If fewer than `min_isolated_target` stars survive, the threshold
        is relaxed (multiplied by `contamination_frac_step` each round)
        up to `contamination_frac_ceiling`.

        max_calibration_stars : int or None
            After the contamination filter, if more than this many stars
            remain usable, keep only the `max_calibration_stars`
            BRIGHTEST (by Gaia G magnitude) rather than every survivor.
            Feeding an unbounded number of stars (can be several hundred
            in a rich field) into one simultaneous IterativePSFPhotometry
            group-fit solve can produce unstable per-star flux
            measurements. A few hundred -- or even ~100 -- bright, clean
            stars already give a statistically excellent zeropoint; this
            cap trades away a small amount of extra precision from
            marginal/faint additional stars for a much more numerically
            stable fit. Set to None to disable the cap.
        group_min_separation_px : float or None
            Passed to psf_photometry.photometry's SourceGrouper --
            stars closer than this (px) are fit SIMULTANEOUSLY as one
            joint group. If None (default), derived automatically per
            frame as `fwhm_px * group_min_separation_fwhm_factor` using
            this frame's actual measured PSF FWHM, rather than a fixed
            pixel value. Anchoring to the frame's own resolution element
            (a small multiple of FWHM, matching the physical scale at
            which PSFs actually overlap) keeps grouping accurate across
            frames with different seeing: a separation far larger than
            the actual PSF width would needlessly group stars that
            aren't blended in reality, while one smaller than the PSF
            width could split a genuinely blended pair into two
            independent, poorly-constrained fits. Pass an explicit
            numeric value here only to override this per-frame
            derivation.
        group_min_separation_fwhm_factor : float
            Multiplier applied to this frame's measured fwhm_px to
            derive group_min_separation_px when that is None. 2.0
            (default) means stars within 2 PSF-widths of each other are
            treated as a blend requiring a joint fit -- consistent with
            typical crowded-field photometry practice (e.g. DAOPHOT-style
            grouping radii). Only used when group_min_separation_px is
            None.
        max_group_size : int
            Target ceiling on how many stars IterativePSFPhotometry will
            ever fit SIMULTANEOUSLY in one group, regardless of
            group_min_separation_px. SourceGrouper links stars
            TRANSITIVELY (A near B near C near D... all end up in one
            group even if A and D are far apart), so in a dense field a
            single group can grow very large very quickly.
            group_min_separation_px is automatically shrunk (down to a
            2px floor) whenever the group it would produce exceeds this
            ceiling -- see psf_photometry.pick_safe_group_separation.
            This is a no-op for typical sparse fields.
        use_grouping : bool
            If False, bypasses simultaneous group-fitting entirely
            (every source fit independently), which can be more stable
            per-star than even a small simultaneous group for a
            genuinely dense field. This pipeline's flux-deblending step
            (using actual measured fluxes, after photometry) already
            corrects for residual neighbour contamination, so this is a
            reasonable choice for dense fields rather than a loss of
            deblending capability. See
            calibration_diagnostics.py's grouping-stability sweep to
            compare against simultaneous-group results for your data
            before choosing a default per target type.
        """
        n_detected = len(self.sources)

        gx = self.gaia_sources["x"].to_numpy()
        gy = self.gaia_sources["y"].to_numpy()
        has_mag = 'phot_g_mean_mag' in self.gaia_sources.columns
        gmag = (self.gaia_sources["phot_g_mean_mag"].to_numpy() if has_mag
                else np.full(len(gx), np.nan))
        gaia_xy = np.column_stack((gx, gy))
        n_gaia_candidates = len(gaia_xy)

        sx = np.array(self.sources["x"])
        sy = np.array(self.sources["y"])
        src_xy = np.column_stack((sx, sy))

        tree = cKDTree(src_xy)
        dist, idx = tree.query(gaia_xy, distance_upper_bound=self.match_tol_px)

        valid = idx < len(src_xy)
        gaia_idx = np.where(valid)[0]
        src_idx = idx[valid]
        dist = dist[valid]

        order = np.argsort(dist)
        gaia_idx = gaia_idx[order]
        src_idx = src_idx[order]

        _, unique = np.unique(src_idx, return_index=True)
        gaia_idx = gaia_idx[unique]
        src_idx = src_idx[unique]
        n_matched = len(gaia_idx)

        matched_xy = gaia_xy[gaia_idx]
        matched_mag = gmag[gaia_idx]

        n_no_mag = int((~np.isfinite(matched_mag)).sum())
        if has_mag and n_no_mag:
            logger.info(f'matching_sources: {n_no_mag}/{n_matched} matched stars have no Gaia '
                        f'G magnitude; treated as median-brightness for contamination purposes')
        elif not has_mag:
            logger.info('matching_sources: no phot_g_mean_mag column available -- contamination '
                        'estimate falls back to treating all matched stars as equal brightness')

        finite_mag = matched_mag[np.isfinite(matched_mag)]
        fill_mag = float(np.median(finite_mag)) if finite_mag.size else 18.0
        matched_mag_filled = np.where(np.isfinite(matched_mag), matched_mag, fill_mag)
        matched_flux_proxy = 10.0 ** (-0.4 * matched_mag_filled)

        fwhm_px = 3.5
        if self.frame_quality is not None and 'fwhm_px' in self.frame_quality.metrics:
            fwhm_px = self.frame_quality.metrics['fwhm_px']

        contam_flux, contamination_frac = local_flux_contamination(
            matched_xy, matched_flux_proxy, matched_xy, matched_flux_proxy,
            fwhm_px=fwhm_px, max_radius_px=self.isolation_radius_px,
        )

        thresh = max_contamination_frac
        isolated_mask = np.zeros(n_matched, dtype=bool)
        while True:
            isolated_mask = contamination_frac < thresh
            n_isolated = int(isolated_mask.sum())

            logger.info(f'matching_sources: contamination_frac<{thresh:.3f} -> '
                        f'{n_isolated}/{n_matched} matched stars usable (fwhm={fwhm_px:.2f}px)')

            if n_isolated >= min_isolated_target or thresh >= contamination_frac_ceiling:
                break
            thresh = min(thresh * contamination_frac_step, contamination_frac_ceiling)

        self.contamination_frac_used = thresh
        if thresh != max_contamination_frac:
            logger.info(f'matching_sources: relaxed max_contamination_frac from '
                        f'{max_contamination_frac:.3f} to {thresh:.3f} to reach '
                        f'{int(isolated_mask.sum())} usable stars')

        final_gaia_idx = gaia_idx[isolated_mask]
        final_src_idx = src_idx[isolated_mask]
        final_mag = matched_mag_filled[isolated_mask]

        if max_calibration_stars is not None and len(final_gaia_idx) > max_calibration_stars:
            # Brightest first (ascending Gaia G mag) -- see docstring for
            # why capping matters more than maximizing raw star count.
            bright_order = np.argsort(final_mag)
            keep_bright = bright_order[:max_calibration_stars]
            logger.info(f'matching_sources: {len(final_gaia_idx)} usable stars exceeds '
                        f'max_calibration_stars={max_calibration_stars}; keeping the '
                        f'{max_calibration_stars} brightest')
            final_gaia_idx = final_gaia_idx[keep_bright]
            final_src_idx = final_src_idx[keep_bright]

        logger.info(f'matching_sources summary: {n_detected} detected -> '
                    f'{n_gaia_candidates} Gaia candidates in field -> {n_matched} matched -> '
                    f'{len(final_gaia_idx)} usable (contamination_frac<{thresh:.3f}'
                    f'{", brightness-capped" if max_calibration_stars is not None else ""})')

        gaia_clean = self.gaia_sources.iloc[final_gaia_idx].reset_index(drop=True)
        sources_clean = self.sources[final_src_idx]

        self.gaia_sources = gaia_clean
        self.sources = sources_clean
        self.positions = np.column_stack(
            (np.asarray(self.sources['x']), np.asarray(self.sources['y']))
        )

    def create_saturation_mask(self):
        sat_mask = []
        for i in range(len(self.positions)):
            x = self.positions[:, 0][i]
            y = self.positions[:, 1][i]
            sat_mask.append(not self.is_saturated_star(x, y, box=11, sat_level=45000, frac=0.10))

        sat_mask = np.array(sat_mask, dtype=bool)
        self.sat_mask = sat_mask

    def catalogue_sources(self, radius_arcmin=None):
        """
        Parameters
        ----------
        radius_arcmin : float or None
            Gaia cone-search radius. If None (default), computed from the
            actual image footprint (center-to-corner distance) plus a 10%
            margin, so the corners of the frame are always covered.
        """
        if radius_arcmin is None:
            ny, nx = self.data.shape
            pix_scale_deg = np.sqrt(np.abs(np.linalg.det(self.wcs.pixel_scale_matrix)))
            half_diag_px = 0.5 * np.hypot(nx, ny)
            radius_arcmin = (half_diag_px * pix_scale_deg * 60.0) * 1.1

        ra0, dec0 = self.wcs.all_pix2world(self.data.shape[1] // 2, self.data.shape[0] // 2, 0)

        cat = gaia_cone(ra0, dec0, radius_arcmin)
        tab = deepcopy(cat)

        gaia_epoch = Time(2016.0, format='jyear', scale='tt')
        image_time = Time(self.header['JD'], format='jd', scale='utc')

        # Missing pm -> pm=0 rather than dropping the star -- Gaia's
        # pmra/pmdec are NaN for a real fraction of sources (fainter
        # stars, and disproportionately so in crowded fields where
        # Gaia's own astrometric solution quality degrades), and
        # SkyCoord.apply_space_motion silently propagates a NaN pm into
        # a NaN sky position, which the pixel-bounds filter below then
        # drops with no record of why. Treating missing pm as pm=0 is a
        # much smaller error than dropping the star: over the ~9-10yr
        # baseline from the Gaia DR3 epoch to a typical observation,
        # even a large 5 mas/yr proper motion is only ~0.05" of drift --
        # negligible next to match_tol_px (2.5px = ~1.8" at 0.72"/px).
        pmra = np.nan_to_num(tab['pmra'].values, nan=0.0)
        pmdec = np.nan_to_num(tab['pmdec'].values, nan=0.0)
        n_missing_pm = int((~np.isfinite(tab['pmra'].values)).sum())
        if n_missing_pm:
            logger.info(f'catalogue_sources: {n_missing_pm}/{len(tab)} Gaia sources had no '
                        f'proper motion; treating as pm=0 rather than dropping them')

        c = SkyCoord(
            ra=tab['ra'].values * u.deg, dec=tab['dec'].values * u.deg,
            pm_ra_cosdec=pmra * u.mas / u.yr,
            pm_dec=pmdec * u.mas / u.yr,
            obstime=gaia_epoch, frame='icrs',
        )
        c_img = c.apply_space_motion(new_obstime=image_time)

        tab['ra'] = c_img.ra.deg
        tab['dec'] = c_img.dec.deg

        x, y = self.wcs.all_world2pix(tab['ra'].values, tab['dec'].values, 0)
        tab['x'] = x
        tab['y'] = y

        finite_xy = np.isfinite(x) & np.isfinite(y)
        n_nonfinite = int((~finite_xy).sum())
        if n_nonfinite:
            logger.warning(f'catalogue_sources: {n_nonfinite} Gaia sources had a non-finite '
                            f'projected pixel position after space-motion correction; dropped')

        ind = (finite_xy &
            (x > 30) & (x < self.data.shape[1] - 30) &
            (y > 30) & (y < self.data.shape[0] - 30))

        tab = tab.iloc[ind].reset_index(drop=True)

        self.cat = tab
        # 'phot_g_mean_mag' is kept (not just x/y) specifically so
        # matching_sources() can compute the flux-contamination estimate.
        keep_cols = ['x', 'y']
        if 'phot_g_mean_mag' in tab.columns:
            keep_cols.append('phot_g_mean_mag')
        self.gaia_sources = tab[keep_cols]

    def _load_sauron(self):
        ra, dec = self.wcs.all_pix2world(self.calibration_df['x_fit'], self.calibration_df['y_fit'], 0)
        self.cal_sys = 'skymapper' if (dec < -25).any() else 'ps1'
        fname = '{filt}_{sys}_{model}.npy'.format(filt=self.band, sys=self.cal_sys, model=self.cal_model)
        self.sauron_state_filename = fname
        self.sauron = sauron(load_state=cal_files_location + fname)

    def predict_mags(self):
        """
        Predict each calibration star's composite magnitude via
        calibrimbore's `sauron.estimate_mag`.

        Note on calibrimbore's internal colour correction: `estimate_mag`
        already applies a fitted cubic colour-correction term (against
        PS1 g-r) and Fitzpatrick99 extinction internally, using the
        `cubic_coeff`/`gr_lims` baked into the loaded `.npy` state file
        (set when the sauron object was originally built/trained per
        filter, via `cubic_corr=True`). This is why `cal_photom` does NOT
        also fit a colour term at the per-frame zeropoint stage -- doing
        so would either duplicate or fight calibrimbore's own correction,
        using far fewer stars per frame than calibrimbore's original
        training set. Stars whose PS1 g-r falls outside `gr_lims` are
        already returned as NaN by `estimate_mag` and are correctly
        dropped by the `np.isfinite` filter in `estimate_zeropoint`.

        `estimate_mag` always returns a single array (verified directly
        against calibrimbore's source) -- it never returns a per-star
        catalog magnitude uncertainty, so `pred_mag_err` always stays
        None and the zeropoint weighting in `estimate_zeropoint` falls
        back to instrumental error alone. This is accurately documented
        here rather than implying calibrimbore provides catalog errors
        that it does not.
        """
        ra, dec = self.wcs.all_pix2world(self.calibration_df['x_fit'].values, self.calibration_df['y_fit'].values, 0)
        self.ra = ra
        self.dec = dec
        self.pred_mag_err = None
        if self.sauron is not None:
            self.pred_mag = self.sauron.estimate_mag(ra=ra, dec=dec, close=True)
        self.sauron = None

    # ------------------------------------------------------------------
    # Calibration set construction (identity-preserving)
    # ------------------------------------------------------------------

    def _calibration_photometry(self, filt_result_table):
        """
        Build the calibration table directly from `ref_id`, which is
        propagated from the isolated/Gaia-matched detection set all the
        way through PSF photometry. This guarantees that the star used
        to predict a catalog magnitude is the same star whose
        instrumental flux was measured, with no re-matching by position
        against a separate, unfiltered position list.
        """
        if 'ref_id' not in filt_result_table.columns:
            return None

        calibration_df = filt_result_table.drop_duplicates(
            subset='ref_id', keep='first'
        ).reset_index(drop=True)

        return calibration_df

    # ------------------------------------------------------------------
    # Magnitude limit
    # ------------------------------------------------------------------

    def magnitude_limit(self, snr_lim=10, mag_limit=19, snr_max=400):
        """
        Estimate magnitude limit from the SNR-magnitude relation.
        """
        sysmag = self.calibration_df['sysmag'].values
        snr = self.calibration_df['snr'].values
        zps = self.zps

        base = (self._zp_mask & np.isfinite(sysmag) & np.isfinite(snr) &
                np.isfinite(zps))

        if np.nansum(base) < 5:
            raise RuntimeError("Too few calibration stars after ZP masking.")

        mag = sysmag[base] + zps[base]
        snr = snr[base]

        mask = ((mag < mag_limit) & (snr > snr_lim) & (snr < snr_max) & (snr > 1))

        if np.nansum(mask) < 5:
            raise RuntimeError("Too few stars to estimate magnitude limit.")

        mag = mag[mask]
        snr = snr[mask]

        p0 = [-1.0, np.nanmedian(mag)]

        result = minimize(self._maglim_objective, p0, args=(snr, mag))
        snr_model = result.x

        resid = mag - self._fitted_line(snr, snr_model)
        good = ~sigma_clip(resid, sigma=3).mask

        result2 = minimize(self._maglim_objective, snr_model, args=(snr[good], mag[good]))
        self.snr_model = result2.x

        self.maglim5 = self.fitted_line(5)
        self.maglim3 = self.fitted_line(3)

    def _maglim_objective(self, var, snr, mag):
        """Pure objective function -- no side effects on self.snr_model."""
        mod_mag = self._fitted_line(snr, var)
        diff = (mag - mod_mag) ** 2
        return np.nansum(diff)

    @staticmethod
    def _fitted_line(sn, params):
        return params[1] + params[0] * np.log10(sn)

    def fitted_line(self, sn):
        return self._fitted_line(sn, self.snr_model)

    # ------------------------------------------------------------------
    # Zeropoint surface (spatially varying ZP / image rescale)
    # ------------------------------------------------------------------

    def assess_spatial_zeropoint_variation(self, smoother=100):
        """
        Summarise how much the zeropoint varies spatially across the
        frame -- see __init__'s assess_spatial_variation docstring for
        the full reasoning. If `rescale=True` already ran
        (ZP_correction/Recast_image_scale), this REUSES that exact
        surface (self.zp_surface) rather than fitting a second one;
        otherwise it fits its own diagnostic-only surface (self.data is
        never touched by this method itself).

        Sets/returns
        ------------
        self.zp_surface_ptp : float (mag)
            Peak-to-peak range of the surface. The headline "how much
            does the zeropoint vary across this frame" number.
        self.zp_surface_rms : float (mag)
            Standard deviation of the surface -- less sensitive to a
            single noisy corner than ptp.
        self.zp_plane_coeffs : (c0, cx, cy) or None
            Least-squares planar fit to the per-star zeropoints:
            zp(x, y) ~= c0 + cx*x + cy*y. A compact (3-number), always-
            computable alternative to the full surface grid.

        Returns (ptp, rms), or (nan, nan) if there's no successful
        zeropoint result yet, or too few spatially-usable calibration
        stars to fit anything meaningful.
        """
        if self.zeropoint_results is None or self.calibration_df is None:
            return np.nan, np.nan

        if self.zp_surface is not None:
            estimate = self.zp_surface
        else:
            try:
                estimate, _ = self.Fit_surface(mask=None, smoother=smoother)
            except Exception as e:
                logger.warning(f'assess_spatial_zeropoint_variation: Fit_surface failed ({e}); skipping')
                return np.nan, np.nan

        valid = np.isfinite(estimate)
        if valid.sum() < 4:
            return np.nan, np.nan

        ptp = float(np.nanmax(estimate[valid]) - np.nanmin(estimate[valid]))
        rms = float(np.nanstd(estimate[valid]))
        self.zp_surface_ptp = ptp
        self.zp_surface_rms = rms

        # Compact planar fit, straight from the per-star zeropoints
        # (self.zps) -- cheap and always computable given >=4 non-
        # collinear stars, unlike the full griddata surface which needs
        # a denser spatial spread to be trustworthy.
        ind = np.isfinite(self.zps)
        if ind.sum() >= 4:
            x = self.calibration_df['x_fit'].values[ind]
            y = self.calibration_df['y_fit'].values[ind]
            z = self.zps[ind]
            design = np.column_stack([np.ones_like(x), x, y])
            coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
            self.zp_plane_coeffs = tuple(float(c) for c in coeffs)

        zp_err = self.zeropoint_results['zp_err']
        logger.info(f'Spatial zeropoint variation: ptp={ptp:.4f} mag, rms={rms:.4f} mag '
                    f'(zp_err={zp_err:.4f} mag for reference -- ptp << zp_err means the single '
                    f'ZP header value is fine everywhere on this frame)')

        return ptp, rms

    def get_local_zeropoint(self, x, y, use_plane=False):
        """
        Position-dependent zeropoint estimate, for the (uncommon) case
        where zp_surface_ptp/rms indicated the single header ZP value
        isn't good enough for a specific high-precision measurement.
        Most photometry should just use the header ZP -- see __init__'s
        assess_spatial_variation docstring.

        Parameters
        ----------
        x, y : float or array_like
            Pixel position(s).
        use_plane : bool
            If False (default), smoothly interpolates the full gridded
            surface (self.zp_surface) via `interpolate_zeropoint_surface`
            -- the same surface that's saved to the zp/ output folder
            and (if rescale=True) actually applied to the saved image.
            If True, uses the compact planar fit (self.zp_plane_coeffs)
            instead -- coarser, but always available even when no
            gridded surface exists (rescale=False and Fit_surface was
            never run).

        Returns
        -------
        float or ndarray -- local zeropoint estimate(s), or the plain
        `zeropoint_results['zp_median']` (with a warning) if no spatial
        fit is available at all.
        """
        zp_global = self.zeropoint_results['zp_median'] if self.zeropoint_results else np.nan

        if not use_plane:
            if self.zp_surface is not None:
                return interpolate_zeropoint_surface(self.zp_surface, x, y)
            logger.warning('get_local_zeropoint: no gridded zp_surface available '
                            '(neither rescale=True nor assess_spatial_zeropoint_variation has '
                            'produced one); falling back to the planar fit if available')

        if self.zp_plane_coeffs is None:
            logger.warning('get_local_zeropoint: no planar fit available either '
                            '(assess_spatial_zeropoint_variation has not run); '
                            'returning the global zp_median instead')
            return zp_global
        c0, cx, cy = self.zp_plane_coeffs
        return c0 + cx * np.asarray(x) + cy * np.asarray(y)

    def Fit_surface(self, mask=None, smoother=100):
        ind = np.isfinite(self.zps)
        if mask is not None:
            ind = ind & mask

        x_data = (self.calibration_df['x_fit'].values[ind] + 0.5).astype(int)
        y_data = (self.calibration_df['y_fit'].values[ind] + 0.5).astype(int)
        z_data = self.zps[ind]
        zpimage = np.full_like(self.data, np.nan, dtype=float)
        zpimage[y_data, x_data] = z_data

        x = np.arange(0, zpimage.shape[1])
        y = np.arange(0, zpimage.shape[0])
        arr = np.ma.masked_invalid(zpimage)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~arr.mask]
        y1 = yy[~arr.mask]
        newarr = arr[~arr.mask]

        if x1.size < 4:
            # Not enough points for a meaningful surface; return a flat
            # surface at the median ZP rather than letting griddata fail.
            flat = np.full_like(zpimage, np.nanmedian(z_data) if z_data.size else np.nan)
            bitmask = np.full_like(zpimage, 128 | 4, dtype=int)
            return flat, bitmask

        estimate = griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
        bitmask = np.zeros_like(zpimage, dtype=int)
        bitmask[np.isnan(estimate)] = 128 | 4
        nearest = griddata((x1, y1), newarr.ravel(), (xx, yy), method='nearest')
        estimate[np.isnan(estimate)] = nearest[np.isnan(estimate)]
        estimate = gaussian_filter(estimate, smoother)
        return estimate, bitmask

    def ZP_correction(self, sigma=2):
        """Correct the zeropoint for residual spatial (e.g. background/
        flat-field) variation, using the existing self.zps array."""
        tmp, _ = self.Fit_surface(mask=None, smoother=200)
        x_data = (self.calibration_df['x_fit'].values + 0.5).astype(int)
        y_data = (self.calibration_df['y_fit'].values + 0.5).astype(int)
        diff = (self.zps - tmp[y_data.astype(int), x_data.astype(int)])
        cut = ~sigma_clip(diff, sigma=sigma).mask
        estimate, bitmask = self.Fit_surface(mask=cut, smoother=30)
        self.zp_surface = estimate
        self.zp_surface_bitmask = bitmask

    def Recast_image_scale(self, newzp=25, set_background=500):
        """
        Recast the image onto a uniform zeropoint scale (ZP=newzp
        everywhere via the fitted zp_surface), then add a constant
        `set_background` offset.

        Why the added offset
        ---------------------
        The recast multiplies (data - floor_val) by a per-pixel scale
        factor derived from the zeropoint surface, then adds floor_val
        back. Background-dominated (near-zero, or slightly negative
        after sky subtraction) pixels can end up negative after that
        scaling -- which then breaks anything downstream expecting
        positive flux (e.g. -2.5*log10(flux), or a simple SNR ratio).
        Adding a fixed positive offset AFTER the rescale shifts the
        whole frame comfortably positive without changing any pixel's
        flux RELATIVE to any other (a constant additive shift, not a
        multiplicative one) -- relative photometry between any two
        pixels/stars in the frame is completely unaffected; this is
        purely a "make raw pixel values usable directly" convenience.
        The value used is recorded (self.set_background_value, written
        to the FITS header as SETBKG by core_reduction.py) specifically
        so it can be subtracted back out by anyone who needs the true,
        offset-free recast flux scale.

        Parameters
        ----------
        newzp : float
            Target uniform zeropoint (mag) for every pixel after the
            recast.
        set_background : float
            Constant flux offset added after rescaling.
        """
        if self.image_floor is not None:
            floor_val = np.nanmedian(self.image_floor)
        else:
            floor_val = np.nanmedian(self.data)

        new_image = ((self.data - floor_val) * 10 ** ((self.zp_surface - newzp) / -2.5)) + floor_val
        new_image = new_image + set_background

        self.data = new_image
        self.set_background_value = set_background
        self.recast_newzp = newzp

    # ------------------------------------------------------------------
    # Saturation
    # ------------------------------------------------------------------

    def is_saturated_star(self, x, y, box=11, sat_level=45000, frac=0.10):
        cutout = Cutout2D(self.data, position=(x, y), size=(box, box), mode='partial', fill_value=np.nan).data

        if cutout is None:
            return True

        npix = np.isfinite(cutout).sum()
        if npix == 0:
            return True

        sat_frac = np.nansum(cutout > sat_level) / npix
        return sat_frac >= frac

    # ------------------------------------------------------------------
    # Zeropoint estimation (weighted)
    # ------------------------------------------------------------------

    def estimate_zeropoint(self, mag_limit=19.0, sigma=3.0, maxiters=5, zp_floor=0.02):
        """
        Robust, inverse-variance-weighted photometric zeropoint estimation.

        Per-star zeropoint:  zp_i = pred_mag_i - sysmag_i
        Per-star weight:     w_i  = 1 / (sysmag_err_i^2 + pred_mag_err_i^2)

        If catalog magnitude errors are unavailable (calibrimbore's
        `estimate_mag` not returning them), falls back to weighting by
        instrumental magnitude error alone -- still a strict improvement
        over unweighted, since it stops faint/noisy stars contributing
        equally to bright/precise ones.

        Returns
        -------
        results : dict with zp_median, zp_sigma, zp_err, N_eff, zp_floor.
        (Field name `zp_median` is kept for backward compatibility even
        though the estimator is now weighted-mean-based, not a simple
        median; downstream code/headers referencing `zp_median` need no
        changes.)
        """
        zps_og = self.pred_mag - self.calibration_df['sysmag'].values
        zps_og = np.asarray(zps_og, dtype=float)

        sysmag_err = self.calibration_df['sysmag_err'].values
        if self.pred_mag_err is not None:
            pred_err = np.asarray(self.pred_mag_err, dtype=float)
        else:
            pred_err = np.zeros_like(zps_og)

        zp_errs_og = np.sqrt(sysmag_err ** 2 + pred_err ** 2)
        # Guard against zero/non-finite per-star errors swamping the
        # weighted mean with infinite weight.
        zp_errs_og = np.where(
            np.isfinite(zp_errs_og) & (zp_errs_og > 1e-6), zp_errs_og, np.nan
        )

        self.pred_mag = np.asarray(self.pred_mag, dtype=float)
        good = ((self.pred_mag < mag_limit) & np.isfinite(self.pred_mag) &
                np.isfinite(zps_og) & np.isfinite(zp_errs_og))

        self.zps = np.full_like(zps_og, np.nan, dtype=float)
        self.zps[good] = zps_og[good]
        self._zp_mask = good

        if np.nansum(good) < 5:
            logger.warning("Too few calibration stars after magnitude cut.")
            self.zeropoint_results = None
            return

        zp, zp_sigma, n_eff, keep_mask = self._weighted_clipped_zeropoint(
            zps_og[good], zp_errs_og[good], sigma=sigma, maxiters=maxiters,
        )

        if n_eff < 5:
            logger.warning("Too few calibration stars after clipping.")
            self.zeropoint_results = None
            return

        zp_err = np.sqrt((zp_sigma / np.sqrt(n_eff)) ** 2 + zp_floor ** 2)

        results = {
            "zp_median": zp, "zp_sigma": zp_sigma, "zp_err": zp_err,
            "N_eff": n_eff, "zp_floor": zp_floor,
        }
        self.zeropoint_results = results

    # ------------------------------------------------------------------
    # Cosmic ray cleaning
    # ------------------------------------------------------------------

    def _clean_cosmics(self):
        mask, clean = astroscrappy.detect_cosmics(
            self.data, sigclip=4.5, sigfrac=0.3, objlim=5.0, cleantype='medmask'
        )
        self.data = clean

    @staticmethod
    def _weighted_clipped_zeropoint(zps, zp_errs, sigma=3.0, maxiters=5):
        """
        Inverse-variance-weighted mean with iterative sigma clipping on
        the residual (using mad_std for the clip threshold, robust to the
        outliers being clipped). Falls back to equal weighting if all
        errors are degenerate.
        """
        zps = np.asarray(zps, dtype=float)
        zp_errs = np.asarray(zp_errs, dtype=float)

        mask = np.isfinite(zps) & np.isfinite(zp_errs) & (zp_errs > 0)
        zps = zps[mask]
        zp_errs = zp_errs[mask]

        if zps.size == 0:
            return np.nan, np.nan, 0, mask

        weights = 1.0 / zp_errs ** 2

        zp = zps[0]
        for _ in range(maxiters):
            zp = np.sum(weights * zps) / np.sum(weights)
            resid = zps - zp
            std = mad_std(resid) if resid.size > 1 else 0.0

            if std == 0 or not np.isfinite(std):
                break

            keep = np.abs(resid) < sigma * std
            if keep.all():
                break

            zps = zps[keep]
            zp_errs = zp_errs[keep]
            weights = 1.0 / zp_errs ** 2

        resid = zps - zp
        zp_sigma = mad_std(resid) if resid.size > 1 else 0.0
        n_eff = zps.size

        return zp, zp_sigma, n_eff, mask