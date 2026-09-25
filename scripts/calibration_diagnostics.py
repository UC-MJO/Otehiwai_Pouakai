"""
Standalone calibration diagnostics.

Run every stage of cal_photom's calibration chain independently on a
single WCS-solved FITS frame, printing survivor counts at each stage and
writing a full Gaia-catalog CSV (position, magnitude, matched flag) so
you can see exactly where stars are gained or lost -- without rerunning
the whole nightly pipeline or digging through the log file.

Usage
-----
    python calibration_diagnostics.py /path/to/frame_wcs.fits.gz
    python calibration_diagnostics.py /path/to/frame_wcs.fits.gz --full
    python calibration_diagnostics.py /path/to/frame_wcs.fits.gz \
        --snr-cuts 3 5 10 15 20 \
        --contamination-fracs 0.01 0.05 0.1 0.2 0.3 0.5 \
        --outdir ./diag_out

What this reports
------------------
1. Frame header basics (filter, exptime, jd, shape, REDFLAT/REDSKY/etc).
2. Source detection (sep, same settings as the pipeline) -- n sources,
   crowd_frac (fraction of frame excluded as a crowded/core blob).
3. Frame-level quality gate verdict + metrics (fwhm, ellipticity, etc.)
4. SNR-cut sensitivity sweep: how many detected sources survive at each
   candidate SNR threshold, using the pipeline's own aperture photometry.
5. Gaia catalog: full CSV dump (ra, dec, x, y, phot_g_mean_mag, matched),
   plus counts (candidates in cone -> after pm/bounds filter -> matched
   to a detection).
6. Contamination-fraction sensitivity sweep: how many matched stars
   survive at each candidate max_contamination_frac threshold -- BUT
   note this is a fast standalone re-implementation using Gaia magnitude
   as a flux proxy, run independently of the real pipeline. If this
   sweep shows plenty of usable stars but the real pipeline still fails,
   the problem is downstream of matching_sources -- use --full (below)
   to find out exactly where.
7. (--full only) Manually walks through EVERY remaining stage of
   cal_photom's __init__ one at a time -- catalogue_sources,
   matching_sources, ePSF build, IterativePSFPhotometry, ref_id
   re-matching, flux deblending, the final SNR cut, calibrimbore's
   predict_mags, and estimate_zeropoint -- printing the survivor count
   and any quality-gate verdict after each one, so a "too few
   calibration stars" failure can be pinned to a specific stage instead
   of treating cal_photom as a black box.
"""

from otehiwai_pouakai import suppress_warnings  # noqa: F401 -- must be first import, see that module's docstring

import argparse
import logging
import os

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import mad_std

from scipy.spatial import cKDTree

from otehiwai_pouakai.frame_quality import sep_extract_sources, assess_frame_quality
from otehiwai_pouakai.psf_photometry import (do_aperture_photometry, local_flux_contamination,
                             build_epsf, photometry, compute_aperture_correction,
                             inflate_psf_errors, mag_error, PSFGroupingError)
from otehiwai_pouakai.gaia_query import gaia_cone
from otehiwai_pouakai.calibration_saurus import cal_photom

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('calibration_diagnostics')

def _load_frame(filepath):
    with fits.open(filepath, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()
    wcs = WCS(header)
    return data, header, wcs

def report_header(header):
    print('=== Frame header ===')
    for key in ['OBJECT', 'FILTER', 'EXPTIME', 'JD', 'READOUTM', 'NAXIS1', 'NAXIS2',
                'REDSKY', 'REDSKRMS', 'REDFLAT', 'REDBKGST']:
        print(f'  {key}: {header.get(key)}')
    print()

def report_detection(data):
    print('=== Source detection (sep, pipeline settings: thresh_sigma=5.0, minarea=5) ===')
    sources, positions, background_rms, crowd_frac = sep_extract_sources(
        data, thresh_sigma=5.0, minarea=5)
    print(f'  n_sources detected: {len(sources)}')
    print(f'  background_rms: {background_rms:.3f}')
    print(f'  crowd_frac excluded (crowded/core regions): {crowd_frac:.1%}')
    print()
    return sources, positions, background_rms

def report_frame_quality(sources, background_rms, header):
    print('=== Frame quality gate ===')
    fq = assess_frame_quality(sources, background_rms, redflat_systematic=header.get('REDFLAT', None), pixel_scale_arcsec=0.72)
    print(f'  verdict: {fq.verdict}')
    for k, v in fq.metrics.items():
        print(f'  metric  {k}: {v}')
    if fq.reasons:
        print('  reasons:')
        for r in fq.reasons:
            print(f'    {r}')
    print()
    return fq

def report_snr_sweep(data, positions, redflat, snr_cuts):
    print('=== SNR-cut sensitivity sweep (aperture photometry, fwhm=3.2) ===')
    _, _, snr, _ = do_aperture_photometry(data, positions, fwhm=3.2, redflat_systematic=redflat)
    for cut in snr_cuts:
        n = int(np.nansum(snr > cut))
        print(f'  snr > {cut:>5}: {n} sources survive')
    print()
    return snr

def report_gaia(header, wcs, data, sources, snr, snr_cut, match_tol_px, outdir, stem):
    print('=== Gaia catalog ===')
    ny, nx = data.shape
    pix_scale_deg = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))
    half_diag_px = 0.5 * np.hypot(nx, ny)
    radius_arcmin = (half_diag_px * pix_scale_deg * 60.0) * 1.1

    ra0, dec0 = wcs.all_pix2world(nx // 2, ny // 2, 0)
    cat = gaia_cone(ra0, dec0, radius_arcmin)
    print(f'  radius_arcmin used: {radius_arcmin:.2f}')
    print(f'  n_gaia_candidates in cone (before pm-propagation/bounds filtering): {len(cat)}')

    gaia_epoch = Time(2016.0, format='jyear', scale='tt')
    image_time = Time(header['JD'], format='jd', scale='utc')

    pmra = np.nan_to_num(cat['pmra'].values, nan=0.0)
    pmdec = np.nan_to_num(cat['pmdec'].values, nan=0.0)
    c = SkyCoord(ra=cat['ra'].values * u.deg, dec=cat['dec'].values * u.deg,
                 pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                 obstime=gaia_epoch, frame='icrs')
    c_img = c.apply_space_motion(new_obstime=image_time)
    cat = cat.copy()
    cat['ra'] = c_img.ra.deg
    cat['dec'] = c_img.dec.deg
    x, y = wcs.all_world2pix(cat['ra'].values, cat['dec'].values, 0)
    cat['x'] = x
    cat['y'] = y

    finite_xy = np.isfinite(x) & np.isfinite(y)
    ind = finite_xy & (x > 30) & (x < nx - 30) & (y > 30) & (y < ny - 30)
    cat = cat[ind].reset_index(drop=True)
    print(f'  n_gaia_candidates after pm-propagation + frame-bounds filter: {len(cat)}')

    keep = np.asarray(snr) > snr_cut
    src_xy = np.column_stack((np.asarray(sources['x'])[keep], np.asarray(sources['y'])[keep]))

    matched = np.zeros(len(cat), dtype=bool)
    if len(src_xy) and len(cat):
        tree = cKDTree(src_xy)
        gxy = cat[['x', 'y']].values
        dist, idx = tree.query(gxy, distance_upper_bound=match_tol_px)
        matched = idx < len(src_xy)

    cat['matched'] = matched
    print(f'  n matched to a detected source (snr>{snr_cut}, tol={match_tol_px}px): {matched.sum()}')

    outpath = os.path.join(outdir, f'{stem}_gaia_catalog.csv')
    cat.to_csv(outpath, index=False)
    print(f'  full Gaia catalog written to: {outpath}')
    print()
    return cat

def report_contamination_sweep(cat, fwhm_px, contamination_fracs, min_isolated_target=15,
                                max_radius_px=21.0):
    print('=== Contamination-fraction sensitivity sweep (matched Gaia stars) ===')
    print('  NOTE: this is a fast, STANDALONE re-implementation using Gaia magnitude as a')
    print('  flux proxy, independent of the real pipeline. If it shows plenty of usable')
    print('  stars but the real pipeline still reports "too few calibration stars", the')
    print('  problem is downstream of matching_sources -- rerun with --full to find out')
    print('  exactly where.')
    matched_cat = cat[cat['matched']].reset_index(drop=True)
    n_matched = len(matched_cat)
    if n_matched == 0:
        print('  no matched stars -- nothing to sweep')
        print()
        return

    xy = matched_cat[['x', 'y']].values
    has_mag = 'phot_g_mean_mag' in matched_cat.columns
    mag = matched_cat['phot_g_mean_mag'].values if has_mag else np.full(n_matched, np.nan)
    finite_mag = mag[np.isfinite(mag)]
    fill_mag = float(np.median(finite_mag)) if finite_mag.size else 18.0
    mag_filled = np.where(np.isfinite(mag), mag, fill_mag)
    flux_proxy = 10.0 ** (-0.4 * mag_filled)

    _, contam_frac = local_flux_contamination(xy, flux_proxy, xy, flux_proxy, fwhm_px=fwhm_px, max_radius_px=max_radius_px)

    print(f'  n_matched: {n_matched}   fwhm_px used: {fwhm_px:.2f}   max_radius_px: {max_radius_px}')
    for frac in contamination_fracs:
        n = int((contam_frac < frac).sum())
        flag = '  <-- below min_isolated_target' if n < min_isolated_target else ''
        print(f'  contamination_frac < {frac:>5}: {n} stars usable{flag}')
    print()

def report_full_calibration_verbose(filepath, match_tol_px=2.5, isolation_radius_px=21.0,
                                     max_contamination_frac=0.05, max_calibration_stars=150,
                                     max_group_size=25, use_grouping=True,
                                     group_min_separation_px=None,
                                     group_min_separation_fwhm_factor=2.0):
    """
    Manually walk through EVERY stage of cal_photom's __init__ (rather
    than calling cal_photom() as a black box), printing the survivor
    count and any quality-gate verdict after each one.

    This is the tool to reach for when the fast standalone sweeps above
    (report_snr_sweep, report_contamination_sweep) suggest plenty of
    usable stars, but the real pipeline still reports "too few
    calibration stars after matching" -- it pinpoints exactly which
    stage disagrees, rather than leaving cal_photom as an opaque
    pass/fail box.

    Mirrors cal_photom.__init__ line-for-line (using the SAME instance
    methods/helper functions the real pipeline calls, not a
    reimplementation)
    """

    print('=== Full calibration trace (manual stage-by-stage) ===')

    cally = cal_photom(filepath, run=False, match_tol_px=match_tol_px,
                       isolation_radius_px=isolation_radius_px,
                       max_contamination_frac=max_contamination_frac,
                       max_calibration_stars=max_calibration_stars,
                       max_group_size=max_group_size, use_grouping=use_grouping,
                       group_min_separation_px=group_min_separation_px,
                       group_min_separation_fwhm_factor=group_min_separation_fwhm_factor)

    cally._load_image()
    cally._clean_cosmics()
    cally._starfinding()
    print(f'  after _starfinding(): {len(cally.sources)} sources '
          f'(crowd_frac excluded: {cally.crowd_frac:.1%})')

    cally.frame_quality = assess_frame_quality(cally.sources, cally.background_rms, 
                                               redflat_systematic=cally.header.get('REDFLAT', None),
                                               pixel_scale_arcsec=cally._pixel_scale_arcsec)
    print(f'  frame_quality verdict: {cally.frame_quality.verdict}')
    if cally.frame_quality.reasons:
        for r in cally.frame_quality.reasons:
            print(f'    {r}')
    if cally.frame_quality.verdict == 'fail':
        print('  STOP: frame quality FAIL -- the real pipeline aborts calibration here.')
        print()
        return cally

    cally.redflat_systematic = cally.header.get('REDFLAT', None)
    if cally.redflat_systematic is not None and not np.isfinite(cally.redflat_systematic):
        cally.redflat_systematic = None

    _, _, snr, _ = do_aperture_photometry(cally.data, cally.positions, fwhm=3.2,
                                          redflat_systematic=cally.redflat_systematic)
    cally.create_saturation_mask()
    snr_mask = snr > 10
    keep = snr_mask & cally.sat_mask
    print(f'  after early snr>10 + saturation-mask cut: {int(keep.sum())}/{len(keep)} sources '
          f'({int((~cally.sat_mask).sum())} flagged saturated)')

    cally.positions = cally.positions[keep]
    cally.sources = cally.sources[keep]
    cally.sources['ref_id'] = np.arange(len(cally.sources))

    cally.catalogue_sources()
    print(f'  Gaia candidates after catalogue_sources(): {len(cally.gaia_sources)}')

    cally.matching_sources(max_contamination_frac=cally.max_contamination_frac,
                           max_calibration_stars=cally.max_calibration_stars)
    print(f'  after matching_sources(): {len(cally.sources)} usable calibration candidates '
          f'(contamination_frac_used={getattr(cally, "contamination_frac_used", float("nan")):.3f})')

    if len(cally.sources) < 5:
        print('  STOP: <5 stars after matching_sources() -- the real pipeline aborts here '
              '("Too few isolated, catalog-matched stars").')
        print()
        return cally

    epsf_data, epsf, epsf_quality = build_epsf(cally.data, cally.sources, sampling=2)
    cally.epsf_quality = epsf_quality
    print(f'  ePSF build verdict: {epsf_quality.verdict}   metrics: {epsf_quality.metrics}')
    if epsf_quality.reasons:
        for r in epsf_quality.reasons:
            print(f'    {r}')
    if epsf is None:
        print('  STOP: ePSF build FAILED -- the real pipeline aborts calibration here.')
        print()
        return cally

    matched_positions = np.column_stack((cally.sources['x'], cally.sources['y']))
    fwhm_px = cally.frame_quality.metrics.get('fwhm_px', 3.2)

    if cally.group_min_separation_px is None:
        resolved_min_separation = fwhm_px * cally.group_min_separation_fwhm_factor
        print(f'  group_min_separation_px derived from measured FWHM: {fwhm_px:.2f}px * '
              f'{cally.group_min_separation_fwhm_factor} = {resolved_min_separation:.2f}px')
    else:
        resolved_min_separation = cally.group_min_separation_px

    try:
        result_table = photometry(cally.data, epsf, matched_positions, cally.daofind,
                                  progress_bar=False, max_iter=30, tol=1e-4, size=11,
                                  min_separation=resolved_min_separation, fwhm=fwhm_px,
                                  max_group_size=cally.max_group_size,
                                  use_grouping=cally.use_grouping)
    except PSFGroupingError as e:
        print(f'  STOP: {e}')
        print('  (the real pipeline would record this as a distinct, recorded failure --')
        print('  see core_reduction.py\'s PSFGroupingError handler -- rather than crashing.')
        print('  Try --sweep-grouping on this file, or lower --max-group-size.)')
        print()
        return cally
    result_table = result_table.to_pandas()
    print(f'  IterativePSFPhotometry returned {len(result_table)} fitted rows '
          f'(input was {len(matched_positions)} positions)')

    result_table = cally._attach_ref_id(result_table, cally.sources, tol=cally.match_tol_px)
    n_before_refid = len(result_table)
    result_table = result_table[np.isfinite(result_table['ref_id'])].copy()
    print(f'  after re-matching PSF-fit positions back to ref_id (tol={cally.match_tol_px}px): '
          f'{len(result_table)}/{n_before_refid} rows kept an identity')
    result_table['ref_id'] = result_table['ref_id'].astype(int)

    if len(result_table) == 0:
        print('  STOP: no rows survived ref_id re-matching -- match_tol_px may be tighter than')
        print('  the actual PSF-fit centroid shift for this frame. Compare x_fit/y_fit in')
        print('  result_table against the input matched_positions to see the real offsets.')
        print()
        return cally

    psf_positions = result_table[['x_fit', 'y_fit']].values
    flux_ap, err_ap, snr_ap, bkg_term_psf = do_aperture_photometry(
        cally.data, psf_positions, fwhm=3.2, redflat_systematic=cally.redflat_systematic)
    result_table['flux_err'] = np.sqrt(bkg_term_psf**2 + result_table['flux_err'].values**2)
    result_table['flux_ap'] = flux_ap
    result_table['flux_ap_err'] = err_ap

    with np.errstate(invalid='ignore', divide='ignore'):
        raw_ratio = result_table['flux_ap'].values / result_table['flux_fit'].values
    finite_raw_ratio = raw_ratio[np.isfinite(raw_ratio) & (result_table['flux_fit'].values > 0)]
    if finite_raw_ratio.size:
        print(f'  RAW flux_ap/flux_fit ratio BEFORE any correction: '
              f'median={np.median(finite_raw_ratio):.4g}  '
              f'mad_std={mad_std(finite_raw_ratio):.4g}  n={finite_raw_ratio.size}')
        print('    (should be close to 1.0 with modest scatter; a large median AND scatter')
        print('    comparable to it -- like ~54 median vs ~59 mad_std -- means flux_fit and')
        print('    flux_ap disagree systematically AND star-to-star, i.e. individual PSF')
        print('    group-fits are unstable, not just an overall flux-scale offset)')

    corr, corr_err = compute_aperture_correction(
        cally.data, psf_positions, result_table['flux_fit'].values,
        fwhm=3.2, redflat_systematic=cally.redflat_systematic)
    print(f'  aperture correction: corr={corr}  corr_err={corr_err}')
    if np.isfinite(corr) and corr > 0:
        result_table['flux_fit_corr'] = result_table['flux_fit'] * corr
        result_table['flux_err_corr'] = result_table['flux_err'] * corr
    else:
        result_table['flux_fit_corr'] = result_table['flux_fit']
        result_table['flux_err_corr'] = result_table['flux_err']
        print('  WARNING: aperture correction failed -- using uncorrected PSF flux')

    error_scale = inflate_psf_errors(result_table, psf_flux_col='flux_fit_corr',
                                      psf_err_col='flux_err_corr', ap_flux_col='flux_ap', min_snr=10,
                                      fwhm=fwhm_px, max_scale=8.0)
    print(f'  PSF error inflation scale: {error_scale:.3f}')

    fwhm_px = cally.frame_quality.metrics.get('fwhm_px', 3.5)
    all_xy = result_table[['x_fit', 'y_fit']].values
    all_flux = result_table['flux_fit_corr'].values
    contam_flux, contam_frac = local_flux_contamination(
        all_xy, all_flux, all_xy, all_flux, fwhm_px=fwhm_px, max_radius_px=cally.isolation_radius_px)
    deblended_flux = result_table['flux_fit_corr'].values - contam_flux
    bad_deblend = ~np.isfinite(deblended_flux) | (deblended_flux <= 0)
    deblended_flux[bad_deblend] = result_table['flux_fit_corr'].values[bad_deblend]
    result_table['flux_fit_deblend'] = deblended_flux
    result_table['flux_err_corr'] = np.sqrt(result_table['flux_err_corr'].values**2 + (0.3 * contam_flux) ** 2)
    print(f'  flux deblending: median contamination_frac={np.nanmedian(contam_frac):.4f}, '
          f'{int((contam_frac > 0.01).sum())}/{len(result_table)} stars corrected >1%, '
          f'{int(bad_deblend.sum())} fell back to uncorrected (would-be non-positive)')

    result_table['snr'] = result_table['flux_fit_deblend'] / result_table['flux_err_corr']
    filt_result_table = result_table[result_table['snr'] > 10].copy().reset_index(drop=True)
    print(f'  after final snr>10 cut (post-deblend): {len(filt_result_table)}/{len(result_table)}')

    calibration_df = cally._calibration_photometry(filt_result_table)
    n_cal = 0 if calibration_df is None else len(calibration_df)
    print(f'  after _calibration_photometry() (dedupe by ref_id): {n_cal}')

    if calibration_df is None or len(calibration_df) < 5:
        print('  STOP: <5 calibration stars -- the real pipeline aborts here '
              '("Too few calibration stars after matching").')
        print()
        return cally

    calibration_df['sysmag'] = -2.5 * np.log10(calibration_df['flux_fit_deblend'])
    calibration_df['sysmag_err'] = mag_error(calibration_df['flux_fit_deblend'].values,
                                              calibration_df['flux_err_corr'].values, 0.0)
    cally.calibration_df = calibration_df

    try:
        cally._load_sauron()
    except Exception as e:
        print(f'  STOP: _load_sauron() raised: {e}')
        print('  (missing/mismatched calibrimbore sauron .npy state file for this filter + '
              'hemisphere (skymapper if dec<-25 else ps1) + cal_model)')
        print()
        return cally

    cally.predict_mags()
    pred_mag_arr = np.asarray(cally.pred_mag, dtype=float)
    n_pred_finite = int(np.isfinite(pred_mag_arr).sum())
    print(f'  predict_mags(): {n_pred_finite}/{len(calibration_df)} stars got a finite '
          f'calibrimbore-predicted magnitude')
    if n_pred_finite < len(calibration_df):
        print('    (the rest are NaN from sauron.estimate_mag -- typically PS1/Skymapper '
              'g-r colour outside calibrimbore\'s trained gr_lims range, or no PS1/Skymapper '
              'cross-match at all for that star/field. If n_pred_finite is small or zero for a')
        print('    field that should have plenty of calibration stars, check whether this '
              'field\'s declination/region has real PS1 or Skymapper coverage -- see cal_sys')
        print(f'    selection: {cally.cal_sys!r} was used (skymapper if dec<-25 else ps1).')

    cally.estimate_zeropoint(mag_limit=19.0, sigma=3.0, maxiters=5, zp_floor=cally.zp_floor)
    if cally.zeropoint_results is None:
        print('  STOP: estimate_zeropoint() produced no result -- see "Too few calibration '
              'stars after magnitude cut/clipping" (this is the mag_limit=19.0 cut combined '
              'with predict_mags() finiteness above, then iterative sigma-clipping).')
    else:
        zr = cally.zeropoint_results
        print(f'  SUCCESS: zp_median={zr["zp_median"]:.4f}  zp_err={zr["zp_err"]:.4f}  '
              f'N_eff={zr["N_eff"]}')
    print()
    return cally

def report_grouping_sweep(filepath, match_tol_px=2.5, isolation_radius_px=21.0,
                           max_contamination_frac=0.05, max_calibration_stars=150,
                           group_separations=(2, 3, 4, 5, 8, 10, 13, 18, 25, 35)):
    """
    Reuse ONE ePSF build (the expensive part) across a sweep of
    SourceGrouper `min_separation` values (cheap to vary once the ePSF
    model exists), reporting how the raw flux_ap/flux_fit ratio's
    scatter and the final identity-matched survivor count change with
    group size.

    This directly tests whether "too many stars fit simultaneously in
    one group" is the dominant driver of the per-star flux instability
    seen via compute_aperture_correction's sanity-bound rejections
    (--full trace), independent of max_calibration_stars -- e.g. if a
    field still shows a bad correction even at the brightest 150 stars,
    this sweep tells you whether shrinking min_separation further would
    actually help (mad_ratio shrinks noticeably at smaller separations)
    or whether the instability lives elsewhere entirely (mad_ratio stays
    large across the whole sweep).
    """

    print('=== Grouping-stability sweep (SourceGrouper min_separation) ===')

    cally = cal_photom(filepath, run=False, match_tol_px=match_tol_px,
                        isolation_radius_px=isolation_radius_px,
                        max_contamination_frac=max_contamination_frac,
                        max_calibration_stars=max_calibration_stars)

    cally._load_image()
    cally._clean_cosmics()
    cally._starfinding()

    cally.frame_quality = assess_frame_quality(cally.sources, cally.background_rms,
                                               redflat_systematic=cally.header.get('REDFLAT', None),
                                               pixel_scale_arcsec=cally._pixel_scale_arcsec)
    if cally.frame_quality.verdict == 'fail':
        print('  STOP: frame quality FAIL -- cannot proceed to grouping sweep.')
        print()
        return

    cally.redflat_systematic = cally.header.get('REDFLAT', None)
    if cally.redflat_systematic is not None and not np.isfinite(cally.redflat_systematic):
        cally.redflat_systematic = None

    _, _, snr, _ = do_aperture_photometry(cally.data, cally.positions, fwhm=3.2,
                                           redflat_systematic=cally.redflat_systematic)
    cally.create_saturation_mask()
    keep = (snr > 10) & cally.sat_mask
    cally.positions = cally.positions[keep]
    cally.sources = cally.sources[keep]
    cally.sources['ref_id'] = np.arange(len(cally.sources))

    cally.catalogue_sources()
    cally.matching_sources(max_contamination_frac=cally.max_contamination_frac,
                           max_calibration_stars=cally.max_calibration_stars)

    if len(cally.sources) < 5:
        print('  STOP: <5 stars after matching_sources -- cannot proceed to grouping sweep.')
        print()
        return

    print(f'  n calibration candidates for this sweep: {len(cally.sources)}')

    epsf_data, epsf, epsf_quality = build_epsf(cally.data, cally.sources, sampling=2)
    if epsf is None:
        print('  STOP: ePSF build failed -- cannot proceed to grouping sweep.')
        print()
        return

    fwhm_px = cally.frame_quality.metrics.get('fwhm_px', 3.2)
    matched_positions = np.column_stack((cally.sources['x'], cally.sources['y']))

    print(f'  fwhm_px used for aperture_radius: {fwhm_px:.2f}')
    ref_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
    ref_str = ', '.join(f'{f}x={fwhm_px * f:.2f}px' for f in ref_factors)
    print(f'  for reference, group_min_separation_fwhm_factor values at this fwhm_px: {ref_str}')
    print('  (the production default is 2.0x -- compare where that lands against the sweep below)')
    print('  NOTE: auto_limit_group_size is deliberately OFF for this sweep so you see the RAW')
    print('  behaviour at exactly each requested min_separation, including crashes -- the real')
    print('  pipeline runs WITH protection on by default (see photometry()\'s docstring) and')
    print('  would not actually crash the way a "CRASHED" row below does.')
    print(f'  {"min_sep":>8} {"n_fit":>7} {"n_id_ok":>8} {"med_ratio":>10} {"mad_ratio":>10} {"n_ratio":>8}')
    for min_sep in group_separations:
        try:
            result_table = photometry(cally.data, epsf, matched_positions, cally.daofind,
                                       progress_bar=False, max_iter=30, tol=1e-4, size=11,
                                       min_separation=min_sep, fwhm=fwhm_px,
                                       auto_limit_group_size=False, backoff_attempts=0)
        except (RecursionError, PSFGroupingError) as e:
            print(f'  {min_sep:>8} {"CRASHED":>7} {"--":>8} {"--":>10} {"--":>10} {"--":>8}   '
                  f'({type(e).__name__})')
            continue
        result_table = result_table.to_pandas()
        n_fit = len(result_table)

        result_table = cally._attach_ref_id(result_table, cally.sources, tol=cally.match_tol_px)
        result_table = result_table[np.isfinite(result_table['ref_id'])].copy()
        n_id_ok = len(result_table)

        if n_id_ok == 0:
            print(f'  {min_sep:>8} {n_fit:>7} {n_id_ok:>8} {"--":>10} {"--":>10} {0:>8}')
            continue

        psf_positions = result_table[['x_fit', 'y_fit']].values
        flux_ap, err_ap, snr_ap, bkg_term_psf = do_aperture_photometry(
            cally.data, psf_positions, fwhm=fwhm_px, redflat_systematic=cally.redflat_systematic)

        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = flux_ap / result_table['flux_fit'].values
        finite_ratio = ratio[np.isfinite(ratio) & (result_table['flux_fit'].values > 0)]

        if finite_ratio.size:
            med = np.median(finite_ratio)
            mad = mad_std(finite_ratio)
            print(f'  {min_sep:>8} {n_fit:>7} {n_id_ok:>8} {med:>10.4g} {mad:>10.4g} {finite_ratio.size:>8}')
        else:
            print(f'  {min_sep:>8} {n_fit:>7} {n_id_ok:>8} {"--":>10} {"--":>10} {0:>8}')

    # Comparison row: bypass simultaneous group-fitting entirely (every
    # source fit independently, grouper=None). This structurally cannot
    # hit the astropy.modeling recursion crash (there is no joint model
    # to build), and is the direct answer to "is ANY simultaneous
    # grouping making this worse than fitting independently?"
    try:
        result_table = photometry(cally.data, epsf, matched_positions, cally.daofind,
                                   progress_bar=False, max_iter=30, tol=1e-4, size=11,
                                   fwhm=fwhm_px, use_grouping=False)
        result_table = result_table.to_pandas()
        n_fit = len(result_table)

        result_table = cally._attach_ref_id(result_table, cally.sources, tol=cally.match_tol_px)
        result_table = result_table[np.isfinite(result_table['ref_id'])].copy()
        n_id_ok = len(result_table)

        if n_id_ok == 0:
            print(f'  {"none":>8} {n_fit:>7} {n_id_ok:>8} {"--":>10} {"--":>10} {0:>8}')
        else:
            psf_positions = result_table[['x_fit', 'y_fit']].values
            flux_ap, err_ap, snr_ap, bkg_term_psf = do_aperture_photometry(
                cally.data, psf_positions, fwhm=fwhm_px, redflat_systematic=cally.redflat_systematic)
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = flux_ap / result_table['flux_fit'].values
            finite_ratio = ratio[np.isfinite(ratio) & (result_table['flux_fit'].values > 0)]
            if finite_ratio.size:
                med = np.median(finite_ratio)
                mad = mad_std(finite_ratio)
                print(f'  {"none":>8} {n_fit:>7} {n_id_ok:>8} {med:>10.4g} {mad:>10.4g} {finite_ratio.size:>8}')
            else:
                print(f'  {"none":>8} {n_fit:>7} {n_id_ok:>8} {"--":>10} {"--":>10} {0:>8}')
    except Exception as e:
        print(f'  {"none":>8}   FAILED: {e}')

    print()
    print('  ^ the "none" row above bypasses simultaneous group-fitting entirely (every source')
    print('  fit independently) -- compare its med_ratio/mad_ratio directly against the swept')
    print('  min_sep rows. If "none" is comparable to or better than the best swept value, use')
    print('  use_grouping=False as the default for this target type rather than tuning a')
    print('  min_separation number; the pipeline\'s flux-deblending step still corrects for')
    print('  residual neighbour contamination afterward using actual measured fluxes.')
    print()
    print('  Interpretation:')
    print('  - med_ratio does NOT need to sit near 1.0 to be "fine" -- a constant overall PSF-')
    print('    normalization offset just becomes part of the fitted zeropoint, harmlessly.')
    print('  - What matters is mad_ratio (scatter). If it shrinks noticeably as min_separation')
    print('    decreases, larger simultaneous fit groups ARE the dominant driver of instability')
    print('    -- a smaller group_min_separation_px default for this target type should reduce')
    print('    reliance on the sanity-bound rejection/fallback in compute_aperture_correction.')
    print('  - If mad_ratio stays large across the WHOLE sweep (including the smallest value),')
    print('    group size is not the (or not the only) cause -- look elsewhere: ePSF model flux')
    print('    normalization/oversampling, or genuinely noisy individual fits from crowding or')
    print('    background systematics (check REDFLAT for this frame).')
    print('  - Watch n_id_ok too: a very small min_separation can split a genuinely blended')
    print('    pair into two independent, poorly-constrained fits -- if n_id_ok drops sharply at')
    print('    the smallest values tested, that is a sign you have gone too far in that')
    print('    direction, not further evidence that smaller is always better.')
    print()

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('fits_file', help='Path to a WCS-solved (wcs/*.fits.gz) frame.')
    p.add_argument('--snr-cuts', type=float, nargs='+', default=[3, 5, 10, 15, 20])
    p.add_argument('--contamination-fracs', type=float, nargs='+',
                    default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument('--match-tol-px', type=float, default=2.5)
    p.add_argument('--isolation-radius-px', type=float, default=21.0)
    p.add_argument('--max-contamination-frac', type=float, default=0.05)
    p.add_argument('--max-calibration-stars', type=int, default=150,
                    help='Cap on stars fed into ePSF/PSF photometry in the --full trace, '
                         'keeping the brightest N. 0 or negative disables the cap.')
    p.add_argument('--max-group-size', type=int, default=25,
                    help='Ceiling on simultaneous PSF-fit group size used in the --full trace '
                         '(auto-shrinks SourceGrouper min_separation to avoid it). See '
                         'psf_photometry.PSFGroupingError.')
    p.add_argument('--group-min-separation-px', type=float, default=None,
                    help='For --full: override SourceGrouper min_separation (px). Default '
                         '(unset) derives it per-frame from measured FWHM -- see '
                         '--group-min-separation-fwhm-factor.')
    p.add_argument('--group-min-separation-fwhm-factor', type=float, default=2.0,
                    help='For --full: multiplier on measured fwhm_px used to derive '
                         'group_min_separation_px when --group-min-separation-px is unset.')
    p.add_argument('--no-grouping', dest='use_grouping', action='store_false',
                    help='For --full: bypass simultaneous group-fitting entirely (every source '
                         'fit independently). Use --sweep-grouping first to see whether this '
                         'is likely to help for a given field (compare its "none" row against '
                         'the swept min_separation rows).')
    p.set_defaults(use_grouping=True)
    p.add_argument('--snr-cut-for-gaia-match', type=float, default=10.0,
                    help='SNR cut used when reporting how many Gaia stars match a detection.')
    p.add_argument('--outdir', type=str, default='.')
    p.add_argument('--full', action='store_true',
                    help='Also run the full stage-by-stage cal_photom trace '
                         '(report_full_calibration_verbose).')
    p.add_argument('--sweep-grouping', action='store_true',
                    help='Also run the grouping-stability sweep (report_grouping_sweep) -- '
                         'tests whether SourceGrouper min_separation is a driver of unstable '
                         'per-star flux measurements, independent of max_calibration_stars.')
    p.add_argument('--group-separations', type=float, nargs='+',
                    default=[5, 8, 10, 13, 18, 25, 35],
                    help='min_separation (px) values to test in --sweep-grouping.')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.basename(args.fits_file).split('.fits')[0]

    data, header, wcs = _load_frame(args.fits_file)
    report_header(header)

    sources, positions, background_rms = report_detection(data)
    fq = report_frame_quality(sources, background_rms, header)

    redflat = header.get('REDFLAT', None)
    if redflat is not None and not np.isfinite(redflat):
        redflat = None

    snr = report_snr_sweep(data, positions, redflat, args.snr_cuts)

    cat = report_gaia(header, wcs, data, sources, snr, args.snr_cut_for_gaia_match, 
                      args.match_tol_px, args.outdir, stem)

    fwhm_px = fq.metrics.get('fwhm_px', 3.5)
    report_contamination_sweep(cat, fwhm_px, args.contamination_fracs,
                               max_radius_px=args.isolation_radius_px)

    if args.full:
        max_cal_stars = args.max_calibration_stars if args.max_calibration_stars > 0 else None
        report_full_calibration_verbose(args.fits_file, args.match_tol_px,
                                        args.isolation_radius_px,
                                        args.max_contamination_frac,
                                        max_cal_stars,
                                        args.max_group_size,
                                        args.use_grouping,
                                        args.group_min_separation_px,
                                        args.group_min_separation_fwhm_factor)

    if args.sweep_grouping:
        max_cal_stars = args.max_calibration_stars if args.max_calibration_stars > 0 else None
        report_grouping_sweep(args.fits_file, args.match_tol_px, args.isolation_radius_px,
                              args.max_contamination_frac, max_cal_stars, group_separations=args.group_separations)

if __name__ == '__main__':
    main()