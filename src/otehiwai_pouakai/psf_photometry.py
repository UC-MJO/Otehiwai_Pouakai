import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from astropy.nddata import NDData
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.stats import sigma_clipped_stats, sigma_clip, SigmaClip, mad_std
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import Cutout2D

from photutils.background import Background2D
from photutils.psf import extract_stars, EPSFStars, EPSFBuilder, EPSFModel, PSFPhotometry, SourceGrouper, IterativePSFPhotometry
from photutils.background import MMMBackground, LocalBackground
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
from photutils.aperture import ApertureStats

from photutils.profiles import RadialProfile

from photutils.detection import DAOStarFinder

logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class EPSFQualityReport:
    """
    Container for ePSF build quality, analogous in spirit to
    frame_quality.FrameQualityReport but specific to the ePSF build step.
    `verdict` is 'pass', 'warn', or 'fail'.
    """

    def __init__(self):
        self.verdict = 'pass'
        self.reasons = []
        self.metrics = {}

    def add_metric(self, name, value):
        self.metrics[name] = value

    def flag(self, severity, reason):
        self.reasons.append(f'[{severity.upper()}] {reason}')
        if severity == 'fail':
            self.verdict = 'fail'
        elif severity == 'warn' and self.verdict != 'fail':
            self.verdict = 'warn'

    def __repr__(self):
        return f'EPSFQualityReport(verdict={self.verdict!r}, reasons={self.reasons}, metrics={self.metrics})'


def _per_star_residual_fractions(fitted_stars, epsf):
    """
    Per-star sum(|data - fitted ePSF model|) / sum(star flux) -- a
    scale-independent goodness-of-fit measure. Verified against synthetic
    test data: a consistent, well-behaved star sample gives a median
    around ~0.1, while a sample contaminated with a handful of
    badly-blended/elongated outliers nearly doubles the median and shows
    a clear high-residual tail for exactly the contaminating stars.

    Stars the builder itself excluded from fitting (`_excluded_from_fit`)
    are skipped here, since normal iteration over `EPSFStars` does not
    filter them out -- they remain in the sequence with the flag set.
    Including their (likely poor) residuals in this median would bias
    the metric upward in a way that doesn't reflect the quality of the
    star sample the ePSF model was actually built from.
    """
    fracs = []
    for star in fitted_stars:
        if getattr(star, '_excluded_from_fit', False):
            continue
        try:
            resid = star.compute_residual_image(epsf)
            star_flux = np.nansum(star.data)
            frac = np.nansum(np.abs(resid)) / star_flux if star_flux > 0 else np.nan
        except Exception:
            frac = np.nan
        fracs.append(frac)
    return np.array(fracs)


def build_epsf(data, stars_tbl, sampling=3, size=11,
               max_residual_frac_warn=0.135,
               min_converged_stars_frac_warn=0.7,
               max_center_accuracy_warn_px=0.5):
    """
    Build an ePSF model and assess its build quality using the real
    diagnostics EPSFBuilder provides: `converged`, `final_center_accuracy`,
    `n_excluded_stars`, and a per-star residual fraction computed via
    each `EPSFStar`'s `compute_residual_image`.

    Threshold calibration note (read before changing these numbers):
    `max_residual_frac_warn` was set from a 30-seed-per-scenario
    synthetic sweep (clean star samples vs. samples with a known fraction
    of deliberately mis-shaped/elongated "contaminating" stars), measuring
    the actual false-positive rate against clean data and true-positive
    rate against contaminated data at several candidate thresholds:

        threshold = clean_mean + 2*clean_std (~0.136 in the sweep):
            ~3% false-positive rate on clean frames
            ~37% true-positive rate for 25%-contaminated samples
        threshold = clean_mean + 3*clean_std (~0.160 in the sweep):
            ~0% false-positive rate on clean frames
            ~7% true-positive rate for 25%-contaminated samples

    Conclusion from this sweep: the per-star residual fraction is a real
    but SOFT signal for moderate contamination on a single frame's star
    sample (tens of stars) -- it reliably flags severe cases but misses
    most moderate ones even at a threshold tuned for a low false-positive
    rate. Given that, this metric:
      (a) is WARN-only, never the sole basis for a hard 'fail' verdict
          (a noisy soft signal shouldn't unilaterally discard a frame),
      (b) uses the ~2-std threshold (0.135) rather than the stricter
          3-std one, accepting a small false-positive rate in exchange
          for catching more real contamination, since the consequence of
          a false 'warn' is just a logged note, not data loss.

    Caveat on this calibration: the sweep above used synthetic scenarios
    where EPSFBuilder's own `_excluded_from_fit` count was 0 throughout
    (no mock scenario happened to trigger a builder-side exclusion).
    `_per_star_residual_fractions` correctly skips excluded stars (since
    they're known-bad by construction and would bias the metric upward
    for reasons unrelated to the quality of the stars actually used), but
    that exclusion-skipping logic post-dates this calibration sweep, so
    the exact threshold value's validity on real frames with non-zero
    exclusion counts hasn't been independently re-verified. Re-run the
    sweep (or validate empirically against real frames) if you want full
    confidence in 0.135 specifically, rather than just the overall
    direction/soundness of the approach.

    The hard-fail signal for a genuinely broken ePSF build instead comes
    from `n_excluded_stars`/`min_converged_stars_frac_warn` below and the
    degenerate-data-range check -- both reflect information the builder
    itself derived from actually fitting every star, not an external
    proxy metric. Re-run this sweep against REAL frames from your
    instrument once you have a baseline of known-good data; synthetic
    Gaussian sources are a reasonable starting point but will not
    perfectly reflect your real PSF/noise/contamination characteristics.

    Returns
    -------
    epsf_data : 2D ndarray or None
        Normalized ePSF data array, or None if the build failed outright
        (exception, or fail-level quality verdict).
    epsf : ImagePSF or None
        The ePSF model object, or None on failure.
    quality : EPSFQualityReport
        Always returned (even on failure), so the caller can log/record
        WHY a build failed or was degraded, not just that it was.
    """
    quality = EPSFQualityReport()

    nddata = NDData(data=data)
    stars = extract_stars(nddata, stars_tbl, size=size)

    all_stars = [star for star in stars]
    n_input_stars = len(all_stars)
    quality.add_metric('n_input_stars', n_input_stars)

    if n_input_stars < 3:
        quality.flag('fail', f'only {n_input_stars} stars available for ePSF build (< 3)')
        return None, None, quality

    try:
        all_stars_combined = EPSFStars(all_stars)

        epsf_builder = EPSFBuilder(oversampling=sampling, maxiters=20,
                                    progress_bar=False, smoothing_kernel='quartic',
                                    recentering_maxiters=10)
        raw_result = epsf_builder(all_stars_combined)

        # EPSFBuilder.__call__'s return type differs across photutils
        # versions: photutils <= ~1.13 returns a plain (epsf,
        # fitted_stars) tuple with no further diagnostics attached;
        # photutils >= ~2.0 returns a single EPSFBuildResult object
        # exposing .epsf/.fitted_stars/.converged/.n_excluded_stars/
        # .final_center_accuracy. Detected at runtime (isinstance check)
        # rather than by sniffing photutils.__version__, since that's
        # robust to any future interface change in either direction.
        if isinstance(raw_result, tuple):
            epsf, fitted_stars = raw_result
            n_excluded_stars = sum(1 for s in fitted_stars.all_stars if getattr(s, '_excluded_from_fit', False))
            # Neither a 'converged' flag nor a center-accuracy figure is
            # available at all from this older interface -- recorded as
            # None/NaN rather than guessed, and the warn-level checks
            # that depend on them are skipped below (skip, not a false
            # 'pass': a missing metric should not silently count as
            # "fine").
            final_center_accuracy = np.nan
        else:
            epsf, fitted_stars = raw_result.epsf, raw_result.fitted_stars
            n_excluded_stars = int(raw_result.n_excluded_stars)
            final_center_accuracy = float(raw_result.final_center_accuracy)
            quality.add_metric('converged_informational', bool(raw_result.converged))

        quality.add_metric('n_excluded_stars', n_excluded_stars)
        if np.isfinite(final_center_accuracy):
            quality.add_metric('final_center_accuracy_px', final_center_accuracy)

        n_kept = n_input_stars - n_excluded_stars
        excluded_frac = n_excluded_stars / n_input_stars if n_input_stars else 1.0
        quality.add_metric('excluded_frac', float(excluded_frac))

        if n_kept < 3:
            quality.flag('fail', f'only {n_kept} stars remained after exclusion (< 3)')
            return None, None, quality

        if excluded_frac > 0.5:
            # A SEVERE exclusion fraction (the majority of input stars
            # rejected by EPSFBuilder's own internal fitting process, not
            # an external proxy) is treated as a hard fail -- this is the
            # builder telling us, from having actually tried to fit every
            # star, that most of the input sample didn't fit the model it
            # converged on. Distinct from the softer warn-level threshold
            # just below, which catches more moderate exclusion rates.
            quality.flag('fail', f'{n_excluded_stars}/{n_input_stars} stars excluded during '
                                 f'build ({excluded_frac:.0%}) -- majority of input sample rejected '
                                 f'by the builder itself')
            return None, None, quality

        if (1.0 - excluded_frac) < min_converged_stars_frac_warn:
            quality.flag('warn', f'{n_excluded_stars}/{n_input_stars} stars excluded during '
                                 f'build ({excluded_frac:.0%}) -- input sample may be contaminated '
                                 f'(blends, cosmic rays, mismatched detections)')

        # `converged` is recorded as an informational metric (when
        # available -- see the version-detection note above) but NOT
        # used to drive the warn/fail verdict on its own: tested
        # empirically against this builder's settings (oversampling=2-3,
        # recentering_maxiters=10) up to maxiters=200 (beyond which
        # photutils itself warns the value is unusually large), and
        # final_center_accuracy oscillates at the sub-0.05px level rather
        # than settling to exactly the strict convergence criterion --
        # `converged=False` fired on every test case regardless of actual
        # fit quality, making it useless as a discriminator if treated as
        # a warn condition. `final_center_accuracy` itself (a real
        # displacement in pixels, not a binary flag) is the part worth
        # thresholding, and it is simply skipped (not assumed fine) on
        # the older photutils interface where it isn't available at all.
        if np.isfinite(final_center_accuracy) and final_center_accuracy > max_center_accuracy_warn_px:
            quality.flag('warn', f'final center accuracy {final_center_accuracy:.3f}px '
                                 f'exceeds {max_center_accuracy_warn_px}px')

        # Per-star residual fraction: a real but SOFT signal per the
        # 30-seed calibration sweep documented in this function's
        # docstring (controlled false-positive rate of ~3% on clean data
        # corresponds to only ~37% true-positive rate for 25%
        # contamination). WARN-only, deliberately never a fail condition
        # on its own -- the hard-fail signals above (excluded_frac,
        # n_kept) come from the builder's own internal fitting decisions,
        # which are more trustworthy than this external proxy metric.
        resid_fracs = _per_star_residual_fractions(fitted_stars, epsf)
        finite_resid = resid_fracs[np.isfinite(resid_fracs)]
        if len(finite_resid) > 0:
            median_resid = float(np.median(finite_resid))
            quality.add_metric('median_residual_frac', median_resid)
            quality.add_metric('max_residual_frac', float(np.max(finite_resid)))

            if median_resid > max_residual_frac_warn:
                quality.flag('warn', f'median per-star residual fraction {median_resid:.3f} exceeds '
                                     f'warn threshold {max_residual_frac_warn} (soft signal -- see '
                                     f'build_epsf docstring for the calibration this threshold is '
                                     f'based on, and its known sensitivity limits)')

        epsf_data = epsf.data
        epsf_data = epsf_data / np.nansum(epsf_data)

        if abs(np.nanmin(epsf_data)) >= abs(np.nanmax(epsf_data)):
            quality.flag('fail', 'ePSF data range inverted (abs(min) >= abs(max)) -- degenerate model')
            return None, None, quality

        if quality.verdict == 'fail':
            return None, None, quality

        return epsf_data, epsf, quality

    except Exception as e:
        quality.flag('fail', f'ePSF build raised an exception: {e}')
        return None, None, quality


def _subpixel_coverage_fraction(stars_tbl, oversampling, x_col='x', y_col='y'):
    """
    Estimate how well star centroids sample the sub-pixel grid an ePSF
    build at this oversampling factor needs to be well-constrained.

    EPSFBuilder builds the ePSF on a grid `oversampling` times finer
    than the native pixel grid, using each star's own sub-pixel centroid
    offset to place its contribution at the right sub-pixel position --
    so a HIGHER oversampling doesn't just need more stars, it needs
    stars whose sub-pixel centroid offsets are well SPREAD OUT across
    the oversampling x oversampling grid of possible offsets. A star
    sample that happens to cluster at similar sub-pixel offsets
    (plausible with a small number of stars) leaves some sub-pixel
    cells with no real constraint at all, no matter how many total
    stars are used.

    This bins each star's fractional (sub-pixel) x/y position into an
    oversampling x oversampling grid and returns the fraction of cells
    containing at least one star -- a cheap, purely positional check
    computable BEFORE running the (expensive) EPSFBuilder, so an
    oversampling factor unlikely to be well-constrained can be skipped
    rather than discovered only after a slow, degraded build.

    Returns
    -------
    frac_covered : float
    n_stars : int
    """
    colnames = stars_tbl.colnames if hasattr(stars_tbl, 'colnames') else stars_tbl.columns
    if x_col not in colnames or y_col not in colnames:
        return 0.0, 0

    x = np.asarray(stars_tbl[x_col])
    y = np.asarray(stars_tbl[y_col])
    if len(x) == 0:
        return 0.0, 0

    frac_x = np.mod(x, 1.0)
    frac_y = np.mod(y, 1.0)

    bin_x = np.clip((frac_x * oversampling).astype(int), 0, oversampling - 1)
    bin_y = np.clip((frac_y * oversampling).astype(int), 0, oversampling - 1)

    occupied = set(zip(bin_x.tolist(), bin_y.tolist()))
    frac_covered = len(occupied) / (oversampling * oversampling)

    return frac_covered, len(x)


def build_epsf_adaptive(data, stars_tbl, sampling_candidates=(3, 2), size=11,
                         min_frac_bins_covered=0.5, **build_epsf_kwargs):
    """
    Attempt an ePSF build at progressively lower oversampling factors,
    accepting the first that gives an adequately-constrained,
    non-failing build -- rather than committing to a single fixed
    oversampling regardless of whether this particular frame's star
    sample can actually support it.

    Why this exists
    ----------------
    Higher oversampling (3x, 4x, ...) resolves finer PSF structure, but
    needs a correspondingly better-sampled sub-pixel distribution of
    star centroids to be well-constrained (see
    `_subpixel_coverage_fraction`'s docstring) -- a frame with fewer
    calibration stars, or stars whose sub-pixel offsets happen to
    cluster, may simply not support 3x oversampling even if 2x is fine.
    Always building at a fixed high oversampling risks a silently
    degraded ePSF on exactly those frames; always building at a fixed
    low oversampling leaves resolution on the table on frames that
    could have supported better. This tries the preferred value first
    and only falls back -- with a logged warning explaining why -- when
    the preferred value genuinely doesn't look supportable for THIS
    frame's star sample.

    Parameters
    ----------
    sampling_candidates : sequence of int
        Oversampling factors to try, in order of PREFERENCE (most
        desired first). Default (3, 2): prefer 3x oversampling, fall
        back to 2x. Add more entries (e.g. (4, 3, 2)) for additional
        fallback levels.
    min_frac_bins_covered : float
        Minimum fraction of the oversampling x oversampling sub-pixel
        grid that must contain at least one star's centroid for that
        oversampling to be attempted at all (see
        `_subpixel_coverage_fraction`). A candidate failing this check
        is skipped WITHOUT running EPSFBuilder (cheap pre-check) unless
        it's the last remaining candidate, in which case it's attempted
        anyway (better to try and possibly fail than have no attempt at
        all).
    **build_epsf_kwargs
        Passed through to `build_epsf` for every candidate (e.g.
        max_residual_frac_warn).

    Returns
    -------
    epsf_data, epsf, quality : same as `build_epsf`. `quality.metrics`
        additionally includes 'oversampling_used' and
        'subpixel_coverage_frac'; if a fallback actually happened, a
        'warn'-level entry is added to `quality.reasons` explaining it.
    """
    if not sampling_candidates:
        raise ValueError('sampling_candidates must be non-empty')

    preferred = sampling_candidates[0]
    last_quality = None

    for i, sampling in enumerate(sampling_candidates):
        is_last = (i == len(sampling_candidates) - 1)

        frac_covered, n_stars = _subpixel_coverage_fraction(stars_tbl, sampling)
        if frac_covered < min_frac_bins_covered and not is_last:
            logger.warning(
                f'build_epsf_adaptive: skipping {sampling}x oversampling -- only '
                f'{frac_covered:.0%} of its {sampling}x{sampling} sub-pixel grid is covered by '
                f'{n_stars} stars\' centroid offsets (need >= {min_frac_bins_covered:.0%}); '
                f'falling back to a lower oversampling rather than risking a poorly-constrained '
                f'ePSF'
            )
            continue

        epsf_data, epsf, quality = build_epsf(data, stars_tbl, sampling=sampling,
                                               size=size, **build_epsf_kwargs)
        quality.add_metric('oversampling_used', sampling)
        quality.add_metric('subpixel_coverage_frac', round(frac_covered, 3))

        if quality.verdict != 'fail':
            if sampling != preferred:
                quality.flag('warn', f'used {sampling}x oversampling instead of preferred '
                                     f'{preferred}x (insufficient sub-pixel coverage or a failed '
                                     f'build at the higher value -- see subpixel_coverage_frac '
                                     f'and earlier reasons)')
            return epsf_data, epsf, quality

        logger.warning(f'build_epsf_adaptive: {sampling}x oversampling build failed '
                        f'({"; ".join(quality.reasons)}); '
                        f'{"trying a lower oversampling" if not is_last else "no lower fallback left"}')
        last_quality = quality

    return None, None, last_quality


class PSFGroupingError(RuntimeError):
    """
    Raised when IterativePSFPhotometry's simultaneous multi-star group
    fit cannot be evaluated even after backing off SourceGrouper's
    min_separation all the way down to its floor.

    Background: astropy.modeling represents a simultaneous fit for N
    grouped stars as a binary tree of nested CompoundModel objects, one
    level per pairwise '+' combination. Evaluating that tree recurses
    through several Python stack frames per level. SourceGrouper links
    stars TRANSITIVELY (star A near B near C near D... all end up in one
    group even if A and D are far apart), so even a modest
    min_separation can produce one very large connected group in a field
    with real stellar density, eventually exceeding Python's recursion
    limit. `pick_safe_group_separation` (below) addresses this by
    capping group SIZE directly, rather than relying on a single fixed
    min_separation to be safe for every field density.
    """
    pass


def _max_group_size(x, y, min_separation):
    """
    Run SourceGrouper standalone (no photometry) purely to inspect how
    large the biggest resulting group would be for a given
    min_separation, accounting for its transitive linking.
    """
    grouper = SourceGrouper(min_separation=min_separation)
    group_ids = np.asarray(grouper(x, y))
    if group_ids.size == 0:
        return 0
    counts = np.bincount(group_ids)
    return int(counts.max())


def pick_safe_group_separation(positions, requested_min_separation=13,
                                max_group_size=25, min_separation_floor=2.0,
                                shrink_factor=0.7):
    """
    Adaptively shrink SourceGrouper's min_separation (starting from
    `requested_min_separation`) until the largest resulting group is at
    most `max_group_size`, or `min_separation_floor` is reached.

    See PSFGroupingError's docstring for why a single fixed
    min_separation isn't safe across fields of varying density, and why
    capping group SIZE (not just distance) is what actually prevents the
    astropy.modeling recursion crash.

    Returns
    -------
    min_separation : float
        The safe value to use (equals `requested_min_separation` if it
        was already safe -- this is a no-op for typical sparse fields).
    max_group_size_achieved : int
        The largest group size at the returned min_separation.
    """
    positions = np.asarray(positions)
    x, y = positions[:, 0], positions[:, 1]

    min_separation = requested_min_separation
    largest = _max_group_size(x, y, min_separation)

    while largest > max_group_size and min_separation > min_separation_floor:
        min_separation = max(min_separation * shrink_factor, min_separation_floor)
        largest = _max_group_size(x, y, min_separation)

    return min_separation, largest


def photometry(data_bkg, epsf, positions, daofind, progress_bar=False,
               max_iter=30, tol=1e-4, size=15, min_separation=13, fwhm=3.2,
               auto_limit_group_size=True, max_group_size=25,
               min_separation_floor=2.0, backoff_attempts=4, use_grouping=True):
    """
    Iterative PSF-based centroiding and photometry using
    IterativePSFPhotometry.

    Parameters
    ----------
    data_bkg : 2D ndarray
        Background-subtracted image.
    epsf : EPSFModel
        Normalized ePSF model.
    positions : array_like
        Initial (x, y) positions of sources.
    daofind : callable
        Star finder used by IterativePSFPhotometry for any additional
        iterations beyond the first.
    progress_bar : bool
        Show progress bar with tqdm.
    max_iter : int
        Maximum number of centroid refinement iterations.
    tol : float
        Convergence threshold in pixels (not directly used by
        IterativePSFPhotometry, kept for interface compatibility).
    size : int
        Fit-shape box size (pixels).
    min_separation : float
        Requested `SourceGrouper` min_separation (pixels). Only used
        when `use_grouping=True`. Stars closer than this are fit
        SIMULTANEOUSLY as one joint group. See PSFGroupingError's
        docstring for why this alone isn't safe across fields of
        varying density -- `auto_limit_group_size` below is the static
        (pre-fit) mitigation; `use_grouping=False` is the more robust
        one for genuinely dense fields, see below.
    fwhm : float
        Assumed PSF FWHM (pixels), used for IterativePSFPhotometry's
        internal initial-flux aperture guess (`aperture_radius=1.4*fwhm`).
        Pass the frame's actual measured FWHM rather than a fixed
        assumption for a better-matched initial guess.
    auto_limit_group_size : bool
        If True (default) and `use_grouping=True`, checks the largest
        group `min_separation` would produce on the INITIAL positions
        (accounting for transitive chaining) via
        `pick_safe_group_separation`, and shrinks it (down to
        `min_separation_floor`) if it exceeds `max_group_size`. NOTE:
        this check only sees the initial positions -- IterativePSFPhoto-
        metry's own `finder` re-detects additional sources every
        iteration, which can enlarge groups DYNAMICALLY beyond what this
        static pre-check predicted (confirmed empirically: a field can
        still hit PSFGroupingError at a separation this check judged
        safe). If that keeps happening for a given field even after
        backoff, `use_grouping=False` is the more robust fix.
    max_group_size : int
        Target ceiling on simultaneous-fit group size when
        `auto_limit_group_size` is True.
    min_separation_floor : float
        Floor below which min_separation is not shrunk further, even if
        groups are still large at that point.
    backoff_attempts : int
        If a RecursionError STILL occurs during the actual fit (e.g. the
        floor was reached before groups got small enough, or a group
        grew dynamically via re-detection), retry up to this many times
        with min_separation halved each time before giving up and
        raising PSFGroupingError. Set to 0 (with
        auto_limit_group_size=False) to get a single raw attempt at
        exactly the requested min_separation, letting a RecursionError
        propagate immediately -- useful for diagnostics that want to see
        exactly where a given field's grouping breaks down.
    use_grouping : bool
        If False, bypasses SourceGrouper/simultaneous group-fitting
        entirely (`grouper=None` -- every source is fit independently).
        This structurally eliminates the astropy.modeling CompoundModel
        recursion risk (there is no joint model to build), and in an
        empirical comparison on a genuinely dense field, independent
        fits may also be substantially MORE stable per-star than even a
        small simultaneous group, since group-size ceilings only bound
        the crash risk, not the degeneracy/instability a joint fit of
        several blended stars can still have. The tradeoff: a truly
        blended pair is fit as if each star were isolated (no explicit
        joint deblending) -- but this pipeline's separate flux-
        deblending step (see calibration_saurus.cal_photom, using
        `psf_photometry.local_flux_contamination` on the ACTUAL measured
        fluxes) already corrects for residual neighbour contamination
        after the fact, so this is a reasonable default to prefer for
        dense fields rather than a regression. All grouping-related
        parameters above are ignored when this is False.

    Returns
    -------
    result_tab : astropy.table.Table
        PSF photometry results.
    """
    positions = np.asarray(positions)

    nddata = NDData(data_bkg)
    fitter = LevMarLSQFitter()
    bkg_estimator = LocalBackground(bkg_estimator=MMMBackground(), inner_radius=9, outer_radius=13)

    if not use_grouping:
        psf_photometry_obj = IterativePSFPhotometry(
            finder=daofind, grouper=None, localbkg_estimator=bkg_estimator,
            psf_model=epsf, fitter=fitter, fit_shape=(size, size), aperture_radius=1.4 * fwhm,
            maxiters=max_iter, progress_bar=progress_bar,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return psf_photometry_obj(nddata.data)

    if auto_limit_group_size:
        safe_sep, largest = pick_safe_group_separation(
            positions, requested_min_separation=min_separation,
            max_group_size=max_group_size, min_separation_floor=min_separation_floor,
        )
        if safe_sep != min_separation:
            logger.info(f'photometry: shrank SourceGrouper min_separation from '
                        f'{min_separation:.2f}px to {safe_sep:.2f}px to keep the largest '
                        f'simultaneous fit group at {largest} stars (target <= {max_group_size})')
        min_separation = safe_sep

    attempt_separation = min_separation
    last_exc = None
    for attempt in range(backoff_attempts + 1):
        group_maker = SourceGrouper(min_separation=attempt_separation)
        psf_photometry_obj = IterativePSFPhotometry(
            finder=daofind, grouper=group_maker, localbkg_estimator=bkg_estimator,
            psf_model=epsf, fitter=fitter, fit_shape=(size, size), aperture_radius=1.4 * fwhm,
            maxiters=max_iter, progress_bar=progress_bar,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result_tab = psf_photometry_obj(nddata.data)
            return result_tab
        except RecursionError as e:
            last_exc = e
            if backoff_attempts == 0:
                raise  # diagnostic mode: propagate immediately, no retry
            logger.warning(f'photometry: RecursionError during simultaneous group fit at '
                            f'min_separation={attempt_separation:.2f}px (attempt {attempt + 1}/'
                            f'{backoff_attempts + 1}) -- a group grew too large (possibly via '
                            f'the finder re-detecting sources across iterations, beyond what '
                            f'the initial static check saw); retrying with a smaller '
                            f'min_separation')
            if attempt_separation <= min_separation_floor:
                break
            attempt_separation = max(attempt_separation * 0.5, min_separation_floor)

    raise PSFGroupingError(
        f'IterativePSFPhotometry hit a RecursionError even after backing off '
        f'min_separation down to {attempt_separation:.2f}px (floor={min_separation_floor}px). '
        f'A group of stars is still too large for astropy.modeling\'s CompoundModel to '
        f'evaluate -- this field is likely too densely packed for simultaneous group fitting '
        f'at any reasonable separation; try use_grouping=False instead (fits every source '
        f'independently, no joint model, no recursion risk).'
    ) from last_exc


def fwhm_compute(epsf):
    data = epsf.data
    oversamp = epsf.oversampling

    center = epsf.origin

    max_radius = data.shape[1] // 2
    radii = np.linspace(0, max_radius + 1, 31)

    rprof = RadialProfile(data, xycen=center, radii=radii)

    r = rprof.radius
    profile = rprof.profile

    half_max = profile.max() / 2
    r_half = np.interp(half_max, profile[::-1], r[::-1])

    fwhm_image_pixels = (2 * r_half) / oversamp
    return np.mean(fwhm_image_pixels)


def do_aperture_photometry(data, positions, fwhm=2.5, gain=None, redflat_systematic=None):
    """
    Simple circular-aperture photometry with annulus-based background
    subtraction.

    Parameters
    ----------
    redflat_systematic : float or None
        Frame-wide background-residual-flatness diagnostic (the REDFLAT
        FITS header value, in the same flux units as `data`), computed
        once per frame by `background_subtraction.background_residual_flatness`
        and written by `core_reduction.reduction_script`. If provided,
        an additional variance term `N_ap * redflat_systematic**2` is
        added to the background variance budget.

        This is a distinct error source from `bkg_std` (the per-annulus
        local sky scatter already included below): `bkg_std` measures
        how noisy the sky is in the immediate vicinity of this star,
        while REDFLAT measures how much the background MODEL itself is
        systematically off, correlated across the whole frame (e.g. an
        imperfectly-fit gradient, or a KDE-rescued tile at a nebula edge
        that's slightly over/under-subtracted). A star sitting in a
        region with low local sky scatter can still be measured against
        a background level that's systematically wrong by REDFLAT --
        local annulus statistics alone can't see that. Omit (None,
        default) to recover the previous behaviour exactly.

    Returns
    -------
    flux, err, snr, bkg_err : ndarray
        Background-subtracted flux, total flux error, SNR, and the
        background-only contribution to the error (useful for combining
        with PSF-fit flux errors elsewhere, since PSF fits typically
        don't include the local sky uncertainty term).
    """
    if len(positions) == 0:
        empty = np.array([])
        return empty, empty, empty, empty

    r_ap = fwhm * np.sqrt(2)
    ap = CircularAperture(positions, r=r_ap)
    ann = CircularAnnulus(positions, r_in=4 * fwhm, r_out=6 * fwhm)

    ap_stats = ApertureStats(data, ann)
    bkg = ap_stats.median
    bkg_std = ap_stats.std

    phot = aperture_photometry(data, ap)
    flux = phot['aperture_sum'] - bkg * ap.area

    N_ap = ap.area
    N_ann = ann.area

    var_bkg = (N_ap * bkg_std**2 + (N_ap**2 / N_ann) * bkg_std**2)

    if redflat_systematic is not None and np.isfinite(redflat_systematic):
        var_bkg = var_bkg + N_ap * redflat_systematic**2

    if gain is not None:
        var_src = np.maximum(flux, 0) / gain
    else:
        var_src = 0.0

    var_tot = var_src + var_bkg
    err = np.sqrt(var_tot)

    with np.errstate(invalid='ignore', divide='ignore'):
        snr = flux / err

    return flux, err, snr, np.sqrt(var_bkg)


def compute_aperture_correction(data, positions, psf_flux, fwhm, gain=None,
                                 r_large_factor=3.0, snr_min=50,
                                 redflat_systematic=None,
                                 corr_min=0.5, corr_max=5.0,
                                 max_contamination_frac=0.05):
    """
    Compute PSF-to-total (aperture) correction from high-SNR, isolated
    stars only.

    Only stars above `snr_min` (using the same aperture-based SNR
    estimate as `do_aperture_photometry`, for consistency) contribute to
    the flux ratio used to derive the correction, so noisy, low-quality
    measurements don't degrade its precision.

    "Isolated" means two things here, both important for getting a
    reliable correction:
      (a) `r_large_factor` (default 3.0) sets how large the "total flux"
          aperture is, as a multiple of `fwhm`. This is generous enough
          to capture the great majority of a normal PSF's flux for a
          well-sampled ePSF, without being so large it routinely reaches
          a neighbouring star. Pass the frame's actual measured FWHM
          (not an assumed value) so this aperture is correctly sized for
          the real PSF width -- an oversized aperture relative to the
          true PSF will tend to pick up nearby stars' flux even for
          otherwise bright, high-SNR stars, since high SNR alone doesn't
          mean a star is isolated.
      (b) Any candidate star whose OWN measured flux is significantly
          contaminated by a neighbour is explicitly excluded, via the
          same `local_flux_contamination` estimate used elsewhere in
          this pipeline (calibration_saurus.matching_sources,
          cal_photom's flux-deblending step), so "isolated" means the
          same thing here as everywhere else in this codebase.

    Sanity bound (corr_min/corr_max): a genuine PSF-to-aperture light
    loss correction should be a modest factor, typically within a few
    tens of percent of 1.0. A computed value far outside that range,
    especially combined with a corr_err comparable to corr itself (large
    star-to-star scatter rather than a clean single systematic factor),
    indicates something upstream is unreliable rather than a real
    "PSF misses the light" effect. Rather than applying an implausible
    correction (which would then get compounded further by
    inflate_psf_errors and can wipe out every star's SNR downstream),
    this rejects it and signals failure to the caller exactly as the
    existing "too few high-SNR stars" case already does, so the caller's
    existing fallback (uncorrected PSF flux) kicks in.

    Parameters
    ----------
    redflat_systematic : float or None
        Same role as in `do_aperture_photometry` -- included here only
        for consistency in the SNR gate used to select which stars are
        "high SNR enough" for the correction; the correction factor and
        its scatter (the actual return values) don't depend on this term.
    corr_min, corr_max : float
        Physically plausible bounds on the correction factor. A computed
        value outside this range is rejected (returns NaN, NaN) rather
        than applied.
    max_contamination_frac : float
        A candidate star is excluded from the correction-fitting sample
        if `local_flux_contamination` estimates more than this fraction
        of its OWN flux comes from a neighbour (using the SAME psf_flux
        values passed in as the flux reference, and `r_large` as the
        search radius, since that's the scale at which a neighbour could
        actually bias this specific measurement).

    Returns
    -------
    flux_corr_factor : float
        Multiply PSF fluxes by this factor.
    flux_corr_err : float
        Scatter in correction (mad_std of the per-star ratio).
    """
    r_large = r_large_factor * fwhm
    ap_large = CircularAperture(positions, r=r_large)
    ann = CircularAnnulus(positions, r_in=5 * fwhm, r_out=7 * fwhm)

    ap_stats = ApertureStats(data, ann)
    bkg = ap_stats.median
    bkg_std = ap_stats.std

    phot = aperture_photometry(data, ap_large)
    flux_large = phot['aperture_sum'] - bkg * ap_large.area

    N_ap = ap_large.area
    N_ann = ann.area
    var_bkg = (N_ap * bkg_std**2 + (N_ap**2 / N_ann) * bkg_std**2)
    if redflat_systematic is not None and np.isfinite(redflat_systematic):
        var_bkg = var_bkg + N_ap * redflat_systematic**2
    with np.errstate(invalid='ignore', divide='ignore'):
        ap_snr = flux_large / np.sqrt(var_bkg)

    positions_arr = np.asarray(positions)
    psf_flux_arr = np.asarray(psf_flux, dtype=float)
    finite_flux = np.isfinite(psf_flux_arr) & (psf_flux_arr > 0)
    if finite_flux.any():
        _, contam_frac = local_flux_contamination(
            positions_arr, psf_flux_arr, positions_arr, psf_flux_arr,
            fwhm_px=fwhm, max_radius_px=r_large,
        )
    else:
        contam_frac = np.full(len(positions_arr), np.nan)
    isolated = np.isfinite(contam_frac) & (contam_frac < max_contamination_frac)

    good = (
        np.isfinite(flux_large) & np.isfinite(psf_flux) & (psf_flux > 0) &
        np.isfinite(ap_snr) & (ap_snr > snr_min) & isolated
    )

    n_excluded_for_contamination = int((~isolated).sum())
    if n_excluded_for_contamination:
        logger.debug(f'compute_aperture_correction: excluded {n_excluded_for_contamination}/'
                     f'{len(positions_arr)} candidates with >{max_contamination_frac:.0%} '
                     f'estimated neighbour flux contamination before computing the correction')

    if np.nansum(good) < 5:
        # Not enough high-SNR, isolated stars to trust a correction;
        # signal failure to the caller rather than silently using a
        # noisy/biased value.
        return np.nan, np.nan

    ratio = flux_large[good] / psf_flux[good]

    ratio_clip = sigma_clip(ratio, sigma=3)
    corr = np.nanmedian(ratio_clip)
    corr_err = mad_std(ratio_clip)

    if not (np.isfinite(corr) and corr_min <= corr <= corr_max):
        logger.warning(
            f'compute_aperture_correction: computed correction {corr:.3g} '
            f'(corr_err={corr_err:.3g}, n={int(np.nansum(good))} stars) is outside the '
            f'physically plausible range [{corr_min}, {corr_max}] -- rejecting rather than '
            f'applying it; falling back to uncorrected PSF flux. A large corr_err relative '
            f'to corr (as here) points to unstable/degenerate group fits rather than a '
            f'genuine flux-scale issue -- see this function\'s docstring.'
        )
        return np.nan, np.nan

    return corr, corr_err


def inflate_psf_errors(result_table, psf_flux_col='flux_fit_corr', psf_err_col='flux_err_corr',
                       ap_flux_col='flux_ap', min_snr=10, max_scale=5.0,
                       x_col='x_fit', y_col='y_fit', fwhm=None,
                       max_contamination_frac=0.05):
    """
    Empirically inflate PSF flux uncertainties so that
    (PSF - AP) / sigma has unit variance.

    Parameters
    ----------
    result_table : pandas.DataFrame
        Output table from PSF + aperture photometry. Modified in place
        (psf_err_col is rescaled).
    psf_flux_col : str
        Column name for PSF flux (after aperture correction).
    psf_err_col : str
        Column name for PSF flux error.
    ap_flux_col : str
        Column name for aperture flux.
    min_snr : float
        Minimum PSF SNR for stars used in inflation estimate.
    max_scale : float
        Sanity ceiling on the empirical inflation factor. A genuine
        underestimate of formal PSF-fit errors is typically modest (a
        factor of a few at most); a MUCH larger empirical scale means
        `psf_flux_col` and `ap_flux_col` disagree systematically for a
        reason OTHER than underestimated noise -- e.g. an aperture
        correction that itself was wrong (see
        `compute_aperture_correction`'s own sanity bound) or unstable
        group-PSF fits. Applying an unbounded scale in that situation
        inflates every star's error so much that essentially nothing
        survives a downstream SNR cut, turning a data-quality problem
        into a total, silent loss of an otherwise-usable calibration set.
        Capped at `max_scale` (with a warning) rather than applied in
        full.
    x_col, y_col : str
        Column names for fitted position, used only for the isolation
        gate below. Ignored (isolation gate skipped, previous behaviour)
        if `fwhm` is None or either column is absent.
    fwhm : float or None
        This frame's measured PSF FWHM (pixels). If provided (along with
        x_col/y_col), an isolation gate is applied -- see
        `max_contamination_frac` -- for the same reason
        `compute_aperture_correction` gained one: comparing PSF flux
        against aperture flux for a star with a real, uncorrected-for
        neighbour is comparing two DIFFERENT things (one includes the
        neighbour's light leaking in, the other may or may not,
        depending on aperture size), which inflates the apparent
        "error" from a source that has nothing to do with the PSF fit's
        actual precision. Leaving this None reproduces the previous
        (contamination-blind) behaviour.
    max_contamination_frac : float
        A star is excluded from the inflation-estimate sample if
        `local_flux_contamination` estimates more than this fraction of
        its own flux comes from a neighbour. Only used if `fwhm` is
        provided.

    Returns
    -------
    scale : float
        Multiplicative inflation factor applied to PSF errors (capped at
        `max_scale`).
    """
    if psf_flux_col not in result_table or ap_flux_col not in result_table:
        return 1.0

    resid = result_table[psf_flux_col] - result_table[ap_flux_col]
    err = result_table[psf_err_col]

    with np.errstate(invalid='ignore', divide='ignore'):
        snr = result_table[psf_flux_col] / err

    good = (np.isfinite(resid) & np.isfinite(err) & (err > 0) & (snr > min_snr))

    if fwhm is not None and x_col in result_table and y_col in result_table:
        xy = result_table[[x_col, y_col]].values
        flux_ref = result_table[psf_flux_col].values.astype(float)
        finite_flux = np.isfinite(flux_ref) & (flux_ref > 0)
        if finite_flux.any():
            _, contam_frac = local_flux_contamination(
                xy, flux_ref, xy, flux_ref, fwhm_px=fwhm,
            )
            isolated = np.isfinite(contam_frac) & (contam_frac < max_contamination_frac)
            good = good & isolated

    if np.nansum(good) < 10:
        return 1.0

    scale = np.nanstd(resid[good] / err[good])

    if not np.isfinite(scale) or scale <= 0:
        return 1.0

    if scale > max_scale:
        logger.warning(
            f'inflate_psf_errors: empirical scale {scale:.2f} exceeds sanity ceiling '
            f'{max_scale} (n={int(np.nansum(good))} high-SNR{" isolated" if fwhm is not None else ""} '
            f'stars) -- this usually means {psf_flux_col} and {ap_flux_col} disagree '
            f'systematically (e.g. a bad aperture correction, or unstable group-PSF fits) '
            f'rather than genuinely underestimated formal errors. Capping at {max_scale} '
            f'instead of applying the full factor, which would otherwise inflate every star\'s '
            f'error enough to fail the downstream SNR cut.'
        )
        scale = max_scale

    result_table.loc[:, psf_err_col] = result_table[psf_err_col] * scale

    return scale


def mag_error(f, f_error, zp_error):
    """
    Propagate flux + zeropoint error into magnitude error.

    Returns NaN (rather than silently producing NaN/inf via division)
    for any star with f <= 0, since a non-positive flux has no
    meaningful magnitude error.
    """
    f = np.asarray(f, dtype=float)
    f_error = np.asarray(f_error, dtype=float)

    with np.errstate(invalid='ignore', divide='ignore'):
        delta_f_term = 2.5 / np.log(10) * (f_error / f)

    delta_f_term = np.where(f > 0, delta_f_term, np.nan)
    delta_zp_term = zp_error

    return np.sqrt(delta_f_term**2 + delta_zp_term**2)


def local_flux_contamination(target_xy, target_flux, all_xy, all_flux, fwhm_px,
                              max_radius_px=None, max_radius_factor=5.0,
                              exclude_self=True, self_tol_px=1e-3):
    """
    Estimate, for each target star, how much of its own flux is actually
    leaking in from OTHER nearby stars, using a symmetric Gaussian PSF
    model -- a continuous, PSF-aware and brightness-aware alternative to
    a hard fixed-radius isolation cutoff.

    Model
    -----
    For target star i at position p_i with flux f_i, and every other
    star j at position p_j with flux f_j within `max_radius_px` of p_i:

        contamination_flux_i = sum_j  f_j * exp(-0.5 * (r_ij / sigma)^2)

    where sigma = fwhm_px / 2.3548 (Gaussian FWHM<->sigma conversion) and
    r_ij is the pixel separation. This approximates the fraction of star
    j's total flux whose PSF wings overlap star i's own core/aperture --
    a rough but physically-motivated stand-in for a full PSF-overlap
    integral, good enough to rank/threshold contamination without
    requiring the actual (elliptical, non-Gaussian-in-the-wings) ePSF
    model, which isn't built yet at the point this is first used
    (matching_sources runs before ePSF construction).

    `contamination_frac_i = contamination_flux_i / target_flux_i` is the
    more directly useful quantity -- e.g. 0.05 means "5% of this star's
    measured/proxy flux is estimated to come from neighbours."

    Parameters
    ----------
    target_xy : (N, 2) array
        Positions to compute contamination FOR.
    target_flux : (N,) array
        Flux (or relative flux proxy, e.g. from Gaia mag) of each target,
        used only as the denominator for contamination_frac.
    all_xy, all_flux : (M, 2), (M,) arrays
        The full population of stars to consider as possible
        contaminants (may be the same arrays as target_xy/target_flux,
        or a larger superset -- e.g. every detected+fitted source in the
        frame vs. just the calibration candidates).
    fwhm_px : float
        This frame's actual measured PSF FWHM (pixels) -- use a real
        per-frame value (e.g. frame_quality's fwhm_px metric), not a
        fixed assumption, since a sharper frame genuinely tolerates
        tighter pairs than a poorly-focused one.
    max_radius_px : float or None
        Hard cutoff -- stars farther apart than this are never
        considered, regardless of brightness (avoids unnecessary cost
        and matches the intuition that a sufficiently distant
        neighbour's contribution is negligible anyway). If None,
        computed as `max_radius_factor * fwhm_px`.
    exclude_self : bool
        Exclude a star from being counted as its own contaminant, when
        `all_xy`/`all_flux` is the same population as `target_xy`/
        `target_flux` (identified by near-zero separation, not index,
        since the two arrays may be ordered differently).

    Returns
    -------
    contamination_flux : (N,) ndarray
    contamination_frac : (N,) ndarray
        contamination_flux / target_flux (NaN where target_flux <= 0).
    """
    from scipy.spatial import cKDTree

    target_xy = np.asarray(target_xy, dtype=float)
    target_flux = np.asarray(target_flux, dtype=float)
    all_xy = np.asarray(all_xy, dtype=float)
    all_flux = np.asarray(all_flux, dtype=float)

    if max_radius_px is None:
        max_radius_px = max_radius_factor * fwhm_px

    sigma = fwhm_px / 2.3548200450309493
    contamination_flux = np.zeros(len(target_xy))

    if len(all_xy) == 0 or len(target_xy) == 0 or sigma <= 0:
        with np.errstate(invalid='ignore', divide='ignore'):
            contamination_frac = np.where(target_flux > 0, 0.0, np.nan)
        return contamination_flux, contamination_frac

    tree = cKDTree(all_xy)

    for i in range(len(target_xy)):
        pos = target_xy[i]
        neighbours = tree.query_ball_point(pos, r=max_radius_px)
        if not neighbours:
            continue

        total = 0.0
        for j in neighbours:
            r = np.hypot(all_xy[j, 0] - pos[0], all_xy[j, 1] - pos[1])
            if exclude_self and r < self_tol_px:
                continue
            total += all_flux[j] * np.exp(-0.5 * (r / sigma) ** 2)

        contamination_flux[i] = total

    with np.errstate(invalid='ignore', divide='ignore'):
        contamination_frac = np.where(
            target_flux > 0, contamination_flux / target_flux, np.nan
        )

    return contamination_flux, contamination_frac