"""
----------------------------------------------------
This pipeline is multi-purpose: a night might be 40 different targets
under varying conditions, or a handful of targets all night, with no
guarantee of revisiting the same field. That rules out reference-tracking
approaches that assume repeated visits to compare against (e.g. "this
star's historical photometric scatter on this field"). Every check here
is therefore self-contained per frame: it judges THIS frame's PSF shape,
ellipticity, source density, and background quality against fixed,
physically-motivated thresholds, not against this field's own history.
"""

import numpy as np
import logging
import threading

from skimage.morphology import binary_dilation, disk

import sep
from astropy.stats import sigma_clipped_stats, mad_std

logger = logging.getLogger(__name__)

# sep.set_extract_pixstack is a GLOBAL, process-wide C-level setting (not
# per-call), per sep's own documentation. The lock below protects against
# multiple threads within the same process calling extraction functions
# concurrently and stepping on each other's global setting (the
# calibration stage itself runs under joblib's loky backend -- separate
# processes, each with independent C-level global state -- so this
# specifically matters if source extraction is ever also called from a
# threaded stage). The default pixel-buffer size is raised well above
# sep's own built-in default of 300000: a 2048x2048 frame has ~4.2M
# pixels total, and a dense or nebula-heavy field with a poorly-
# subtracted background can easily flag a large fraction of them as
# "above threshold" before any quality gate gets a chance to reject the
# frame. This is a documented common cause of a pixel-buffer overflow
# (see github.com/kbarbary/sep issues #33/#43): either genuinely high
# source/nebula density, or an upstream background-subtraction problem
# flagging far too many pixels. The buffer size is escalated further on
# retry (see `_extract_with_pixstack_retry`) rather than set unboundedly
# high from the start, since an unbounded buffer on a genuinely broken
# background would just turn a fast, clear failure into a slow,
# memory-heavy one.
_PIXSTACK_LOCK = threading.Lock()
_DEFAULT_PIXSTACK = 1_000_000
_MAX_PIXSTACK = 4_000_000
_DEFAULT_SUB_OBJECT_LIMIT = 1000        # sep's own built-in default
_ESCALATED_SUB_OBJECT_LIMIT = 100_000   # generous headroom -- safe to set this high
                                         # because mask_crowded_regions has already
                                         # excluded genuine cluster-core-like regions
                                         # before extraction ever reaches this point;
                                         # what's left needing this escalated limit is
                                         # moderately dense (not core-crowded), where
                                         # the cost of deblending fully is small.

class SepExtractionError(RuntimeError):
    """
    Raised when `sep.extract` still overflows its internal pixel buffer
    even at the maximum escalated pixstack size (see
    `_extract_with_pixstack_retry`).

    Subclasses `RuntimeError` (so existing broad `except RuntimeError`/
    `except Exception` handlers still catch it), but is a distinct type
    from `gaia_query.py`'s plain `RuntimeError` (raised after exhausting
    Gaia archive retries): that one is a transient, likely-to-resolve-
    on-retry network issue and is deliberately not recorded in the
    failure ledger, whereas a pixstack overflow here is a stable,
    structural problem with this specific frame (almost always a
    background-subtraction issue) that will not fix itself on a bare
    retry and SHOULD be recorded. `core_reduction.py`'s
    `calibrating_internal` distinguishes the two by type.
    """
    pass

def mask_crowded_regions(data, thresh_sigma=5.0, box_size=64, area_frac_thresh=0.4,
                          dilation_px=10, mask=None):
    """
    Identify and mask regions too crowded/blended for deblending to
    usefully resolve (e.g. a globular cluster core), WITHOUT attempting
    to deblend them.

    Rationale
    ---------
    sep's sub-object limit exists because deblending a genuinely merged
    blob (hundreds of overlapping profiles fused into one connected
    region) is both expensive and not physically meaningful in a single
    pass. For zeropoint calibration, isolated stars OUTSIDE the crowded
    region are exactly as useful as ones from a sparser field -- there's
    no reason a crowded core should fail the WHOLE frame's extraction.
    Rather than chasing an ever-larger sub-object limit (slow, and
    eventually just produces thousands of poorly-deblended "child"
    sources from the core that isolation filtering would reject anyway),
    this does a CHEAP, non-deblended threshold pass, tiles it, and flags
    any tile where above-threshold pixels dominate the tile (i.e. it's a
    merged blob, not resolvable point sources) -- then excludes those
    tiles from detection entirely, before the real deblended pass runs.

    Parameters
    ----------
    data : 2D ndarray
        Background-subtracted frame.
    thresh_sigma : float
        Threshold (units of global background RMS) for the cheap
        pre-pass. Match this to the main extraction's own threshold.
    box_size : int
        Tile size (pixels) for the coarse crowding assessment.
    area_frac_thresh : float
        A tile is flagged "crowded" if more than this fraction of its
        pixels sit above threshold. 0.4 is deliberately generous -- an
        ordinary, even fairly dense, star field essentially never covers
        40% of a 64x64 tile with source pixels; a merged cluster core
        does.
    dilation_px : int
        Approximate extra margin (pixels) grown around flagged tiles, so
        the mask also covers the blend's wings/boundary, not just its
        brightest pixels.
    mask : 2D bool ndarray or None
        Existing mask (e.g. non-finite data, bad pixels) to combine with.

    Returns
    -------
    crowd_mask : 2D bool ndarray
        True = excluded from detection (crowded/core region).
    crowd_frac : float
        Fraction of the frame flagged as crowded (0.0 if none) -- log or
        record this per frame so a reduced source count is traceable to
        "core excluded" rather than looking like reduced sensitivity.
    """
    finite = np.isfinite(data)
    exclude = ~finite if mask is None else (~finite | mask)

    bkg = sep.Background(np.ascontiguousarray(data.astype(np.float32)),
                          mask=np.ascontiguousarray(exclude))
    thresh = bkg.globalback + thresh_sigma * bkg.globalrms
    above = finite & (data > thresh)

    ny, nx = data.shape
    ny_tiles = int(np.ceil(ny / box_size))
    nx_tiles = int(np.ceil(nx / box_size))

    crowd_mask = np.zeros_like(above, dtype=bool)
    n_crowded_tiles = 0

    for j in range(ny_tiles):
        y0, y1 = j * box_size, min((j + 1) * box_size, ny)
        for i in range(nx_tiles):
            x0, x1 = i * box_size, min((i + 1) * box_size, nx)
            tile_above = above[y0:y1, x0:x1]
            frac = tile_above.mean() if tile_above.size else 0.0
            if frac > area_frac_thresh:
                crowd_mask[y0:y1, x0:x1] = True
                n_crowded_tiles += 1

    if crowd_mask.any() and dilation_px > 0:
        # cheap approximation of disk(dilation_px) via repeated small
        # dilations, so this stays fast even for a large flagged region
        crowd_mask = binary_dilation(crowd_mask, disk(max(1, dilation_px // 5)))
        for _ in range(4):
            crowd_mask = binary_dilation(crowd_mask, disk(max(1, dilation_px // 5)))

    crowd_frac = float(crowd_mask.mean())
    if n_crowded_tiles:
        logger.info(f'mask_crowded_regions: flagged {n_crowded_tiles}/{ny_tiles*nx_tiles} '
                    f'tiles as crowded/merged-blob ({crowd_frac:.1%} of frame area)')

    return crowd_mask, crowd_frac


def _extract_with_pixstack_retry(data, thresh_sigma, rms, mask, minarea,
                                  deblend_nthresh, deblend_cont, clean):
    """
    Run `sep.extract`, escalating sep's global pixel-buffer and
    sub-object limits on the corresponding overflow errors, rather than
    raising immediately.

    Two independent, separately-recognised sep errors are handled here:

      - 'pixel buffer full': sep's internal per-pixel working buffer.
      - 'sub-object'/deblend overflow: too many child objects produced
        while deblending one connected region.

    By the time this runs, `sep_extract_sources` has already excluded
    genuinely crowded/merged-blob regions via `mask_crowded_regions`, so
    reaching the sub-object escalation here should be rare and should
    represent moderate (not core-level) crowding. Any other exception
    from `sep.extract` is re-raised immediately without retry.

    Parameters
    ----------
    data : 2D ndarray (float32, contiguous)
        Background-subtracted frame.
    thresh_sigma : float
        Detection threshold, in units of `rms`.
    rms : float
        Global background RMS, used with `thresh_sigma` for detection.
    mask : 2D bool ndarray
        Pixels to exclude from detection.
    minarea : int
        Minimum number of pixels for a detection.
    deblend_nthresh, deblend_cont :
        Passed through to `sep.extract`.
    clean : bool
        Passed through to `sep.extract`.

    Returns
    -------
    objects : structured ndarray
        As returned by `sep.extract`.

    Raises
    ------
    SepExtractionError
        If extraction still fails with a pixel-buffer or sub-object
        overflow at the maximum escalated settings.
    """
    attempts = [
        (_DEFAULT_PIXSTACK, _DEFAULT_SUB_OBJECT_LIMIT),
        (_MAX_PIXSTACK, _DEFAULT_SUB_OBJECT_LIMIT),
        (_MAX_PIXSTACK, _ESCALATED_SUB_OBJECT_LIMIT),
    ]

    last_exc = None
    for attempt_idx, (pixstack, sub_object_limit) in enumerate(attempts):
        is_last = attempt_idx == len(attempts) - 1
        with _PIXSTACK_LOCK:
            sep.set_extract_pixstack(pixstack)
            sep.set_sub_object_limit(sub_object_limit)
            try:
                return sep.extract(
                    data, thresh_sigma, err=rms, mask=mask,
                    minarea=minarea, deblend_nthresh=deblend_nthresh,
                    deblend_cont=deblend_cont, clean=clean,
                )
            except Exception as e:
                last_exc = e
                msg = str(e)
                if 'pixel buffer full' in msg:
                    logger.warning(f'sep.extract pixel buffer full at pixstack={pixstack}; '
                                    f'{"retrying with a larger buffer" if not is_last else "giving up"}')
                    continue
                if 'sub-object' in msg.lower() or 'deblend' in msg.lower():
                    logger.warning(f'sep.extract sub-object limit reached at '
                                    f'limit={sub_object_limit} (pixstack={pixstack}); '
                                    f'{"retrying with a higher limit" if not is_last else "giving up"}')
                    continue
                raise  # a different sep error entirely -- don't mask it, re-raise immediately

    raise SepExtractionError(
        f'sep.extract: exhausted pixstack/sub-object escalation (final attempt: '
        f'pixstack={_MAX_PIXSTACK}, sub_object_limit={_ESCALATED_SUB_OBJECT_LIMIT}). '
        f'This should be rare given crowded/core-like regions are already excluded '
        f'before extraction (see mask_crowded_regions) -- if it still fires, this frame '
        f'likely has genuine wide-area crowding (not a compact core) or a background-'
        f'subtraction problem; check REDFLAT/REDBKGST before assuming this is '
        f'unfixable crowding.'
    ) from last_exc


def sep_extract_sources(data, thresh_sigma=3.0, minarea=5,
                         deblend_nthresh=32, deblend_cont=0.005,
                         mask=None, clean=True,
                         mask_crowded=True, crowd_box_size=64,
                         crowd_area_frac_thresh=0.4, crowd_dilation_px=10):
    """
    Detect sources in a background-subtracted frame via `sep.extract`,
    optionally excluding crowded/merged-blob regions first, and return
    both the raw detections and their pixel positions.

    Pipeline: crowded-region masking (if enabled) -> global background
    RMS estimate on the (crowd-)masked frame -> source extraction (with
    automatic pixstack/sub-object-limit escalation on overflow, via
    `_extract_with_pixstack_retry`).

    Parameters
    ----------
    data : 2D ndarray
        Background-subtracted frame.
    thresh_sigma : float
        Detection threshold, in units of the frame's global background
        RMS.
    minarea : int
        Minimum number of pixels for a detection.
    deblend_nthresh, deblend_cont : int, float
        Deblending parameters passed through to `sep.extract`.
    mask : 2D bool ndarray or None
        Existing mask (e.g. non-finite data, bad pixels) to exclude from
        detection, combined with the crowded-region mask if
        `mask_crowded` is True.
    clean : bool
        Passed through to `sep.extract` (removes likely spurious
        detections around bright sources).
    mask_crowded : bool
        If True (default), pre-exclude crowded/merged-blob regions (e.g.
        a globular cluster core) from detection entirely via
        `mask_crowded_regions`, before running the real deblended
        extraction. This is what actually prevents sep's sub-object
        overflow for cluster fields, rather than just raising the limit
        -- and it means isolated stars OUTSIDE the core are still
        detected and usable for zeropoint calibration.
    crowd_box_size, crowd_area_frac_thresh, crowd_dilation_px :
        Passed through to `mask_crowded_regions` (as `box_size`,
        `area_frac_thresh`, `dilation_px`).

    Returns
    -------
    sources : astropy.table.Table
        One row per detected source, with SEP's standard fields (`x`,
        `y`, `a`, `b`, `theta`, `flux`, `peak`, `flag`, ...) plus
        `xcentroid`/`ycentroid` aliases for `x`/`y`. Empty (zero-row)
        table with the same columns if nothing was detected.
    positions : 2D ndarray, shape (n_sources, 2)
        `(x, y)` pixel positions of each detected source; shape
        `(0, 2)` if nothing was detected.
    background_rms : float
        Global background RMS used for detection.
    crowd_frac : float
        Fraction of the frame excluded as crowded (0.0 if
        `mask_crowded=False` or nothing was flagged). Worth logging or
        writing to the header per frame so a reduced source count is
        traceable to "core excluded", not mistaken for lost sensitivity.
    """
    data = np.ascontiguousarray(data.astype(np.float32))
    base_mask = np.zeros(data.shape, dtype=bool) if mask is None else np.ascontiguousarray(mask)

    crowd_frac = 0.0
    combined_mask = base_mask
    if mask_crowded:
        crowd_mask, crowd_frac = mask_crowded_regions(
            data, thresh_sigma=thresh_sigma, box_size=crowd_box_size,
            area_frac_thresh=crowd_area_frac_thresh, dilation_px=crowd_dilation_px,
            mask=base_mask,
        )
        combined_mask = base_mask | crowd_mask

    bkg = sep.Background(data, mask=combined_mask)
    rms = bkg.globalrms

    objects = _extract_with_pixstack_retry(
        data, thresh_sigma, rms, combined_mask, minarea,
        deblend_nthresh, deblend_cont, clean,
    )

    from astropy.table import Table
    sources = Table(objects) if len(objects) else Table(
        names=['x', 'y', 'a', 'b', 'theta', 'flux', 'peak', 'flag'],
        dtype=[float, float, float, float, float, float, float, int],
    )
    if len(sources):
        sources['xcentroid'] = sources['x']
        sources['ycentroid'] = sources['y']

    positions = np.column_stack([sources['x'], sources['y']]) if len(sources) else np.empty((0, 2))

    return sources, positions, float(rms), crowd_frac

def fwhm_from_shape(sources, conversion=2.3548):
    """
    Per-source FWHM estimate from SEP's second-moment ellipse parameters,
    via the geometric mean of the semi-major/minor axes (a, b) converted
    from Gaussian sigma to FWHM. Verified against synthetic data to
    recover the true FWHM to ~2.5% for round sources.

    Parameters
    ----------
    sources : astropy.table.Table or structured array
        Must have `a` and `b` columns (as returned by `sep_extract_sources`).
    conversion : float
        Gaussian sigma-to-FWHM conversion factor
        (2*sqrt(2*ln(2)) ~= 2.3548).

    Returns
    -------
    1D ndarray
        Per-source FWHM estimate, in pixels. Empty array if `sources` is
        empty.
    """
    if len(sources) == 0:
        return np.array([])
    sigma_equiv = np.sqrt(sources['a'] * sources['b'])
    return sigma_equiv * conversion


def ellipticity_from_shape(sources):
    """
    Per-source ellipticity (1 - b/a): 0 for a perfectly round source,
    approaching 1 for a highly elongated one (e.g. a trailed source from
    guiding error or wind shake).

    Parameters
    ----------
    sources : astropy.table.Table or structured array
        Must have `a` and `b` columns (as returned by `sep_extract_sources`).

    Returns
    -------
    1D ndarray
        Per-source ellipticity. Empty array if `sources` is empty.
    """
    if len(sources) == 0:
        return np.array([])
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1.0 - (sources['b'] / sources['a'])


# ----------------------------------------------------------------------
# Frame-level quality report
# ----------------------------------------------------------------------

class FrameQualityReport:
    """
    Container for a frame's quality verdict and the metrics behind it.

    `verdict` is one of 'pass', 'warn', 'fail'. `reasons` lists every
    threshold that triggered a warn/fail, so the verdict is always
    traceable back to specific numbers, not just a label. `metrics` holds
    the raw measured values (e.g. `fwhm_px`, `ellipticity_median`)
    regardless of whether they triggered a flag.
    """

    def __init__(self):
        self.verdict = 'pass'
        self.reasons = []
        self.metrics = {}

    def add_metric(self, name, value):
        """Record a metric value under `name`, for later inclusion in
        the report (and, via `as_header_dict`, in the FITS header)."""
        self.metrics[name] = value

    def flag(self, severity, reason):
        """
        Record a warn/fail condition and escalate `self.verdict`
        accordingly.

        Parameters
        ----------
        severity : str
            Either 'warn' or 'fail'.
        reason : str
            Human-readable description of what triggered the flag,
            appended to `self.reasons`.

        Notes
        -----
        Escalates but never downgrades `self.verdict`: a 'fail' recorded
        earlier in the same report is not un-failed by a later
        'warn'-only check.
        """
        self.reasons.append(f'[{severity.upper()}] {reason}')
        if severity == 'fail':
            self.verdict = 'fail'
        elif severity == 'warn' and self.verdict != 'fail':
            self.verdict = 'warn'

    # Explicit metric-name -> FITS-key mapping, rather than generic
    # truncation to 8 chars (which risks collisions, e.g. 'fwhm_px' and
    # 'fwhm_arcsec' both truncating toward similar keys).
    _HEADER_KEY_MAP = {
        'n_sources': 'QNSRC',
        'fwhm_px': 'QFWHMPX',
        'fwhm_arcsec': 'QFWHMAS',
        'fwhm_scatter_frac': 'QFWHMSC',
        'ellipticity_median': 'QELLIP',
        'background_rms': 'QBKGRMS',
        'redflat': 'QREDFLT',
    }

    def as_header_dict(self):
        """
        Build a FITS-header-friendly summary of this report: short keys,
        values truncated to the FITS 68-character value limit where
        needed, and all flag reasons joined into a single string.

        Any metric not covered by `_HEADER_KEY_MAP` gets a generated key
        (`Q` + up to 6 alphanumeric characters from its name, with a
        numeric suffix if needed to avoid colliding with another
        generated key), so an unexpected/custom metric name still gets a
        safe, unique header key rather than risking a silent truncation
        collision.

        Returns
        -------
        dict of {FITS_KEY: (value, comment)}, ready to merge into an
        astropy.io.fits.Header.
        """
        d = {
            'QUALVRD': (self.verdict, 'Frame quality verdict: pass/warn/fail'),
        }
        used_keys = set(d.keys())
        for k, v in self.metrics.items():
            key = self._HEADER_KEY_MAP.get(k)
            if key is None:
                base = ''.join(c for c in k.upper() if c.isalnum())[:6]
                key = f'Q{base}'
                suffix = 0
                while key in used_keys:
                    suffix += 1
                    key = f'Q{base}{suffix}'[:8]
            used_keys.add(key)
            if isinstance(v, float):
                d[key] = (round(v, 4), f'Quality metric: {k}')
            else:
                d[key] = (v, f'Quality metric: {k}')
        reasons_str = '; '.join(self.reasons) if self.reasons else 'none'
        d['QUALRSN'] = (reasons_str[:68], 'Quality flag reasons (truncated; see log for full)')
        return d

    def __repr__(self):
        return f'FrameQualityReport(verdict={self.verdict!r}, reasons={self.reasons}, metrics={self.metrics})'


def assess_frame_quality(
    sources, background_rms, redflat_systematic=None,
    n_sources_min_warn=10, n_sources_min_fail=3,
    fwhm_min_px=1.0, fwhm_max_warn_px=12.0, fwhm_max_fail_px=20.0,
    fwhm_scatter_max_warn=0.5,
    ellipticity_max_warn=0.15, ellipticity_max_fail=0.35,
    redflat_max_warn=None, redflat_max_fail=None,
    pixel_scale_arcsec=0.72,
):
    """
    Apply fixed, physically-motivated thresholds to a frame's detected
    sources and derived metrics, returning a `FrameQualityReport`.

    This is deliberately self-contained per frame (see module docstring
    for why) -- no comparison against this field's own history is made
    or assumed available.

    Parameters
    ----------
    sources : structured array from sep_extract_sources
    background_rms : float
        From sep_extract_sources's third return value.
    redflat_systematic : float or None
        REDFLAT diagnostic from background_subtraction.py /
        core_reduction.py, if available.
    n_sources_min_warn / n_sources_min_fail : int
        Below these counts, the frame likely has a transparency problem
        (cloud), a tracking failure, or points at an empty/wrong field --
        not enough detections to even assess PSF shape reliably.
    fwhm_min_px : float
        Below this, "sources" are likely hot pixels/cosmic ray remnants
        rather than real PSF-shaped detections (sanity floor, not a
        seeing judgement).
    fwhm_max_warn_px / fwhm_max_fail_px : float
        Above these, seeing is poor enough to warrant flagging. At
        pixel_scale_arcsec=0.72, the defaults (12px / 20px) correspond to
        roughly 8.6" / 14.4" FWHM -- adjust for your site's typical
        seeing distribution; these are deliberately generous defaults
        meant to catch genuinely bad frames (focus drift, major guiding
        failure), not to enforce excellent seeing.
    fwhm_scatter_max_warn : float
        Max allowed (robust) scatter in per-source FWHM, in units of the
        median FWHM. High scatter with a high median often indicates
        partial cloud or spatially-variable focus, not uniform-but-poor
        seeing.
    ellipticity_max_warn / ellipticity_max_fail : float
        Median per-source ellipticity thresholds. Elevated, frame-wide
        ellipticity indicates guiding error or wind shake (a real,
        physical effect, not measurement noise) -- verified against
        synthetic trailed sources to correctly recover a strong
        ellipticity signal.
    redflat_max_warn / redflat_max_fail : float or None
        Optional thresholds on the background-residual-flatness
        diagnostic (REDFLAT, flux units) -- left as None by default since
        a sensible absolute threshold depends on this instrument's flux
        scale/gain; set these once you have a baseline of REDFLAT values
        from genuinely good frames to compare against.

    Returns
    -------
    report : FrameQualityReport
    """
    report = FrameQualityReport()

    n_sources = len(sources)
    report.add_metric('n_sources', n_sources)

    if n_sources < n_sources_min_fail:
        report.flag('fail', f'only {n_sources} sources detected (< {n_sources_min_fail}); '
                             f'likely cloud, tracking failure, or empty/wrong field')
        return report
    elif n_sources < n_sources_min_warn:
        report.flag('warn', f'only {n_sources} sources detected (< {n_sources_min_warn})')

    fwhm = fwhm_from_shape(sources)
    ellip = ellipticity_from_shape(sources)

    good_shape = np.isfinite(fwhm) & np.isfinite(ellip) & (fwhm > fwhm_min_px)
    if good_shape.sum() < max(n_sources_min_fail, 3):
        report.flag('fail', f'fewer than {max(n_sources_min_fail, 3)} sources had usable shape '
                             f'measurements after filtering tiny/degenerate detections')
        return report

    fwhm_good = fwhm[good_shape]
    ellip_good = ellip[good_shape]

    _, fwhm_median, fwhm_std = sigma_clipped_stats(fwhm_good, sigma=3, maxiters=5, stdfunc=mad_std)
    ellip_median = float(np.median(ellip_good))

    report.add_metric('fwhm_px', float(fwhm_median))
    report.add_metric('fwhm_arcsec', float(fwhm_median * pixel_scale_arcsec))
    report.add_metric('fwhm_scatter_frac', float(fwhm_std / fwhm_median) if fwhm_median > 0 else np.nan)
    report.add_metric('ellipticity_median', ellip_median)
    report.add_metric('background_rms', float(background_rms))
    if redflat_systematic is not None and np.isfinite(redflat_systematic):
        report.add_metric('redflat', float(redflat_systematic))

    if fwhm_median > fwhm_max_fail_px:
        report.flag('fail', f'median FWHM {fwhm_median:.2f}px ({fwhm_median*pixel_scale_arcsec:.2f}") '
                             f'exceeds fail threshold {fwhm_max_fail_px}px')
    elif fwhm_median > fwhm_max_warn_px:
        report.flag('warn', f'median FWHM {fwhm_median:.2f}px ({fwhm_median*pixel_scale_arcsec:.2f}") '
                            f'exceeds warn threshold {fwhm_max_warn_px}px')

    if fwhm_median > 0 and (fwhm_std / fwhm_median) > fwhm_scatter_max_warn:
        report.flag('warn', f'FWHM scatter {fwhm_std/fwhm_median:.2f} (frac of median) exceeds '
                            f'{fwhm_scatter_max_warn} -- possible partial cloud or variable focus')

    if ellip_median > ellipticity_max_fail:
        report.flag('fail', f'median ellipticity {ellip_median:.3f} exceeds fail threshold '
                             f'{ellipticity_max_fail} -- likely severe guiding/tracking error')
    elif ellip_median > ellipticity_max_warn:
        report.flag('warn', f'median ellipticity {ellip_median:.3f} exceeds warn threshold '
                            f'{ellipticity_max_warn} -- possible guiding error or wind shake')

    if redflat_max_fail is not None and redflat_systematic is not None and np.isfinite(redflat_systematic):
        if redflat_systematic > redflat_max_fail:
            report.flag('fail', f'REDFLAT {redflat_systematic:.2f} exceeds fail threshold {redflat_max_fail}')
        elif redflat_max_warn is not None and redflat_systematic > redflat_max_warn:
            report.flag('warn', f'REDFLAT {redflat_systematic:.2f} exceeds warn threshold {redflat_max_warn}')

    return report