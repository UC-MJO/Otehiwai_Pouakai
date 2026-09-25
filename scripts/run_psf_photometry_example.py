"""
Example: running PSF photometry on an already-reduced (background-
subtracted, WCS-solved) FITS frame using psf_photometry.py directly,
without going through the full otehiwai_pouakai/calibration_saurus
pipeline.

This is a standalone tutorial/reference module, not a pipeline module --
copy and adapt it rather than importing from it in production code. It's
class-based rather than a command-line tool: import `PSFPhotometryRunner`
and call `.run(...)` with whatever you have on hand.

One entry point, auto-dispatched by what you pass in
-------------------------------------------------------
`runner.run(...)` figures out what kind of photometry you want from
its arguments -- no separate methods to remember, no CSV required
unless you already have one:

  - `x=1024.3, y=987.1`               -- one target, pixel coordinates
  - `ra=83.6331, dec=-5.3911`         -- one target, sky coordinates
  - `x=[10, 20, 30], y=[15, 25, 35]`  -- several targets, straight from
                                          lists/arrays (same length)
  - `ra=[...], dec=[...]`             -- several targets, sky coordinates
  - `targets='my_targets.csv'`        -- many targets from a CSV
                                          (columns x,y OR ra,dec)
  - `targets=<DataFrame or (N,2) array>` -- same, already in memory
  - nothing at all                    -- every detected source in the
                                          frame passing snr > snr_min

A single (x, y) and a list of (x, y) go through the exact same code
path (a scalar is just treated as a length-1 array) -- so "one target"
is not a special case, it's just the N=1 case of "a list of targets".

Usage
-----
    from run_psf_photometry_example import PSFPhotometryRunner

    runner = PSFPhotometryRunner('reduced_frame.fits')

    result = runner.run(x=1024.3, y=987.1)                     # single, pixel
    result = runner.run(ra=83.6331, dec=-5.3911)                # single, sky
    result = runner.run(x=[10, 20, 30], y=[15, 25, 35])         # list, pixel
    result = runner.run(targets='my_targets.csv')               # list, CSV
    result = runner.run(snr_min=10)                              # all sources
    result = runner.run(snr_min=10, spatial=True, nx=3, ny=3)   # spatial ePSF

    runner.save(result, 'my_output.csv')

See the __main__ block at the bottom for a runnable end-to-end example
-- edit the file path and arguments there and run this file directly.
"""

import logging

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats

from photutils.detection import DAOStarFinder

from otehiwai_pouakai.frame_quality import sep_extract_sources, assess_frame_quality
from otehiwai_pouakai.psf_photometry import (do_aperture_photometry, build_epsf_adaptive,
                             photometry, mag_error)
from otehiwai_pouakai.spatial_epsf import build_spatial_epsf

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('run_psf_photometry_example')


class PSFPhotometryRunner:
    """
    Loads one reduced FITS frame, detects reference stars to build an
    ePSF from, and runs PSF photometry on whatever you ask `.run()` for.

    Parameters
    ----------
    fits_file : str
        Path to an already-reduced (background-subtracted, WCS-solved)
        FITS frame.
    fwhm_guess : float
        Starting FWHM guess (px), used only for the initial reference-
        star SNR cut before the real FWHM is measured from the frame.
    snr_min_reference : float
        SNR threshold for the reference-star sample the ePSF is built
        from (kept separate from any per-target SNR cut you use later).
    max_reference_stars : int
        Cap on how many (brightest) reference stars are used to build
        the ePSF -- a few hundred clean stars is already plenty.
    sampling_candidates : sequence of int
        ePSF oversampling factors to try, most preferred first -- see
        psf_photometry.build_epsf_adaptive.
    """

    def __init__(self, fits_file, fwhm_guess=3.2, snr_min_reference=20,
                 max_reference_stars=200, sampling_candidates=(3, 2)):
        self.fits_file = fits_file
        self.sampling_candidates = tuple(sampling_candidates)

        self.data, self.header, self.wcs = self._load_frame(fits_file)
        logger.info(f'Loaded {fits_file}: shape={self.data.shape}')

        self.stars_tbl, self.background_rms, self.daofind = self._detect_reference_stars(
            fwhm_guess=fwhm_guess, snr_min=snr_min_reference, max_stars=max_reference_stars,
        )

        fq = assess_frame_quality(self.stars_tbl, self.background_rms, pixel_scale_arcsec=0.72)
        self.fwhm_px = fq.metrics.get('fwhm_px', fwhm_guess)
        logger.info(f'Measured frame FWHM: {self.fwhm_px:.2f}px')

        self._global_epsf = None      # lazily built, cached
        self._global_epsf_data = None
        self._global_quality = None
        self._spatial_epsf = None     # lazily built, cached

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_frame(filepath):
        with fits.open(filepath, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float64)
            header = hdul[0].header.copy()
        wcs = WCS(header)
        return data, header, wcs

    def _detect_reference_stars(self, fwhm_guess=3.2, snr_min=20, max_stars=200):
        sources, positions, background_rms, crowd_frac = sep_extract_sources(
            self.data, thresh_sigma=5.0, minarea=5,
        )
        logger.info(f'Detected {len(sources)} raw sources ({crowd_frac:.1%} of frame excluded '
                    f'as crowded/core region)')

        _, _, snr, _ = do_aperture_photometry(self.data, positions, fwhm=fwhm_guess)
        stars_tbl = sources[snr > snr_min]
        logger.info(f'{len(stars_tbl)} stars pass snr > {snr_min} (used for ePSF build)')

        if len(stars_tbl) > max_stars:
            order = np.argsort(-np.asarray(stars_tbl['flux']))
            stars_tbl = stars_tbl[order[:max_stars]]
            logger.info(f'Capped to the {max_stars} brightest for the ePSF build')

        _, _, std = sigma_clipped_stats(self.data, sigma=3)
        daofind = DAOStarFinder(fwhm=4, threshold=10.0 * std)

        return stars_tbl, background_rms, daofind

    def _get_global_epsf(self):
        if self._global_epsf is None:
            epsf_data, epsf, quality = build_epsf_adaptive(
                self.data, self.stars_tbl, sampling_candidates=self.sampling_candidates,
            )
            logger.info(f'ePSF build verdict: {quality.verdict}  metrics: {quality.metrics}')
            for r in quality.reasons:
                logger.info(f'  {r}')
            if epsf is None:
                raise RuntimeError(f'ePSF build failed: {"; ".join(quality.reasons)}')
            self._global_epsf, self._global_epsf_data, self._global_quality = epsf_data, epsf, quality
        return self._global_epsf_data, self._global_epsf, self._global_quality

    def _get_spatial_epsf(self, nx=3, ny=3, min_stars_per_node=15):
        if self._spatial_epsf is None:
            self._spatial_epsf = build_spatial_epsf(
                self.data, self.stars_tbl, self.data.shape, nx=nx, ny=ny,
                min_stars_per_node=min_stars_per_node,
                sampling_candidates=self.sampling_candidates, wcs=self.wcs,
            )
            logger.info('Spatial ePSF grid:\n' + self._spatial_epsf.node_summary())
        return self._spatial_epsf

    # ------------------------------------------------------------------
    # Target resolution -- everything funnels into an (N, 2) pixel array
    # ------------------------------------------------------------------

    def _xy_from_scalars_or_arrays(self, x, y):
        """
        A bare scalar and a list/array both work here: np.atleast_1d
        turns a scalar into a length-1 array, so `x=1024.3, y=987.1`
        and `x=[1024.3, 5.0], y=[987.1, 12.0]` go through the exact
        same code path -- "one target" is just the N=1 case of "a list
        of targets", not a special case.
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        if x_arr.shape != y_arr.shape:
            raise ValueError(f'x and y must have the same shape, got {x_arr.shape} and {y_arr.shape}')
        return np.column_stack([x_arr, y_arr])

    def _xy_from_table(self, targets):
        """targets: CSV path (str), pandas DataFrame, or (N, 2) array-like."""
        if isinstance(targets, str):
            df = pd.read_csv(targets)
        elif isinstance(targets, pd.DataFrame):
            df = targets
        else:
            arr = np.asarray(targets, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError('Array targets must be shape (N, 2) -- (x, y) pairs')
            return arr

        if {'x', 'y'}.issubset(df.columns):
            return df[['x', 'y']].values.astype(float)
        if {'ra', 'dec'}.issubset(df.columns):
            x, y = self.wcs.all_world2pix(df['ra'].values, df['dec'].values, 0)
            return np.column_stack([x, y]).astype(float)
        raise ValueError(f'Expected columns (x, y) or (ra, dec), got {list(df.columns)}')

    def _detect_all(self, snr_min=10):
        sources, positions, _, _ = sep_extract_sources(self.data, thresh_sigma=5.0, minarea=5)
        _, _, snr, _ = do_aperture_photometry(self.data, positions, fwhm=self.fwhm_px)
        targets_xy = positions[snr > snr_min]
        logger.info(f'{len(targets_xy)} sources pass snr > {snr_min}')
        return targets_xy

    def _resolve(self, x, y, ra, dec, targets, snr_min):
        """
        Single dispatch point -- figures out which of the run() inputs
        were actually given and returns an (N, 2) pixel-coordinate
        array, or falls back to detecting every qualifying source.
        """
        if targets is not None:
            return self._xy_from_table(targets)

        if x is not None and y is not None:
            return self._xy_from_scalars_or_arrays(x, y)

        if ra is not None and dec is not None:
            px, py = self.wcs.all_world2pix(np.atleast_1d(ra), np.atleast_1d(dec), 0)
            return np.column_stack([px, py])

        return self._detect_all(snr_min=snr_min)

    # ------------------------------------------------------------------
    # Photometry
    # ------------------------------------------------------------------

    def _run_global(self, targets_xy):
        epsf_data, epsf, quality = self._get_global_epsf()
        result = photometry(self.data, epsf, targets_xy, self.daofind,
                             progress_bar=False, max_iter=30, size=11, fwhm=self.fwhm_px,
                             min_separation=self.fwhm_px * 2.0)
        return result.to_pandas()

    def _run_spatial(self, targets_xy, nx=3, ny=3, min_stars_per_node=15):
        spatial = self._get_spatial_epsf(nx=nx, ny=ny, min_stars_per_node=min_stars_per_node)
        node_idx = np.array([spatial.nearest_node(x, y) for x, y in targets_xy])

        results = []
        for idx in np.unique(node_idx):
            group_targets = targets_xy[node_idx == idx]
            epsf, epsf_data, used_global = spatial.get_epsf(spatial.node_x[idx], spatial.node_y[idx])
            logger.info(f'Node {idx}: {len(group_targets)} targets, using '
                        f'{"GLOBAL fallback" if used_global else "local"} ePSF')
            result = photometry(self.data, epsf, group_targets, self.daofind,
                                 progress_bar=False, max_iter=30, size=11, fwhm=self.fwhm_px,
                                 min_separation=self.fwhm_px * 2.0)
            results.append(result.to_pandas())
        return pd.concat(results, ignore_index=True)

    def _finalize(self, result_df, zeropoint=None):
        ra, dec = self.wcs.all_pix2world(result_df['x_fit'].values, result_df['y_fit'].values, 0)
        result_df['ra'] = ra
        result_df['dec'] = dec
        result_df['mag_inst'] = -2.5 * np.log10(result_df['flux_fit'].clip(lower=1e-10))
        result_df['mag_inst_err'] = mag_error(
            result_df['flux_fit'].values, result_df['flux_err'].values, 0.0,
        )
        if zeropoint is not None:
            result_df['mag_approx'] = result_df['mag_inst'] + zeropoint
        return result_df

    # ------------------------------------------------------------------
    # The one public entry point
    # ------------------------------------------------------------------

    def run(self, x=None, y=None, ra=None, dec=None, targets=None,
            snr_min=10, spatial=False, nx=3, ny=3, min_stars_per_node=15,
            zeropoint=None):
        """
        Run PSF photometry, automatically deciding what you want based
        on which arguments are given (checked in this order):

          1. `targets` set (CSV path / DataFrame / (N, 2) array)
             -> photometer exactly those targets.
          2. `x`/`y` set (scalar OR list/array, same length)
             -> photometer exactly those pixel positions.
          3. `ra`/`dec` set (scalar OR list/array, same length)
             -> convert to pixel positions via this frame's WCS, then
             photometer those.
          4. none of the above
             -> detect every source in the frame passing
             `snr > snr_min` and photometer all of them.

        A single value and a list of values for x/y (or ra/dec) are NOT
        different modes -- they go through the identical code path, so
        there's nothing extra to learn for "many targets" vs. "one
        target".

        Parameters
        ----------
        spatial : bool
            If True, use a spatially varying ePSF grid (spatial_epsf.py)
            instead of one global model -- each target gets its nearest
            grid node's own ePSF. See nx/ny/min_stars_per_node.
        zeropoint : float or None
            Optional zeropoint (mag) to also report an approximate
            calibrated magnitude -- a convenience for this example, NOT
            a substitute for the full calibration_saurus.cal_photom
            zeropoint pipeline.

        Returns
        -------
        pandas.DataFrame with fitted position, flux, sky coordinates,
        and instrumental magnitude -- one row per target.
        """
        targets_xy = self._resolve(x, y, ra, dec, targets, snr_min)
        logger.info(f'Photometering {len(targets_xy)} target(s)')

        result_df = (self._run_spatial(targets_xy, nx=nx, ny=ny, min_stars_per_node=min_stars_per_node)
                     if spatial else self._run_global(targets_xy))
        return self._finalize(result_df, zeropoint=zeropoint)

    @staticmethod
    def save(result_df, outfile):
        result_df.to_csv(outfile, index=False)
        logger.info(f'Wrote {len(result_df)} rows to {outfile}')


if __name__ == '__main__':
    # Edit this and run the file directly for a quick, runnable example
    # -- no command-line arguments involved.
    FITS_FILE = 'reduced_frame.fits'
    runner = PSFPhotometryRunner(FITS_FILE)

    # One target, pixel coordinates
    result = runner.run(x=1024.3, y=987.1)
    print(result[['x_fit', 'y_fit', 'ra', 'dec', 'flux_fit', 'mag_inst']])

    # Several targets, straight from lists -- same method, same code path
    result = runner.run(x=[1024.3, 512.0, 1800.5], y=[987.1, 512.0, 300.2])
    print(result[['x_fit', 'y_fit', 'ra', 'dec', 'flux_fit', 'mag_inst']])

    # Every usable source in the frame
    result = runner.run(snr_min=10)
    runner.save(result, 'all_sources_photometry.csv')