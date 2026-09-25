"""
Spatially varying ePSF support: build one ePSF per region of an image
rather than a single global one, and look up the locally-appropriate
model for any (x, y) or (RA, Dec).

Design choices (read before using)
-----------------------------------
photutils' PSFPhotometry/IterativePSFPhotometry take a SINGLE psf_model
per call -- there is no built-in way to pass a different model per star
within one simultaneous fit. Genuine spatial variation therefore has two
honest options:

1. PIECEWISE-CONSTANT regions (what this module does): partition the
   image into a grid, build one real ePSF per grid node from the
   calibration stars nearest that node (falling back to a single global
   ePSF, built from ALL stars, for any node with too few of its own),
   and look up the NEAREST node's ePSF for a given position. Using this
   for actual photometry means running photometry() ONCE PER REGION, on
   only the stars/targets in that region, rather than one call across
   the whole frame -- see run_psf_photometry_example.py's
   --spatial-epsf option for a worked example.

   Nearest-node (not smooth bilinear interpolation between regions) is
   a deliberate choice, not a shortcut: interpolating between two
   INDEPENDENTLY-BUILT ePSF pixel arrays only makes sense if both are
   registered to the same sub-pixel center, which isn't guaranteed --
   naively averaging two ePSFs offset from each other by even a
   fraction of a pixel blurs the result, silently degrading exactly the
   resolution a spatially-varying model exists to preserve. Nearest-node
   avoids that correctness risk entirely, at the cost of a visible seam
   at region boundaries rather than a smooth transition.

2. FOLD IT INTO THE ZEROPOINT SURFACE instead (the "cheeky 0th-order
   capture" approach): fit with a single global ePSF as normal, and let
   any residual spatially-varying photometric bias show up in, and get
   corrected by, the existing zeropoint-surface machinery
   (calibration_saurus.cal_photom.ZP_correction / Fit_surface). Lower
   risk, and reuses code that already exists and is already validated
   -- at the cost of only capturing the SCALAR flux effect of PSF
   variation (not its shape), and only at the precision the zeropoint-
   star grid itself supports. For most photometric-calibration purposes
   (as opposed to precision shape-fitting, e.g. weak lensing), this is
   very likely the better cost/benefit choice. Reach for the full
   SpatialEPSF machinery in this module only if PSF SHAPE (not just
   flux) genuinely needs to vary correctly across the frame.
"""

import numpy as np
import logging

from .psf_photometry import build_epsf_adaptive

logger = logging.getLogger(__name__)


class SpatialEPSF:
    """
    A grid of independently-built ePSF models across an image. Query
    with `get_epsf(x, y)` (or `get_epsf_radec(ra, dec)`) to get the
    model appropriate for that location -- the nearest grid node's own
    build, or the global fallback ePSF if that node didn't have enough
    stars of its own.

    Not usually constructed directly -- use `build_spatial_epsf()`.
    """

    def __init__(self, node_x, node_y, node_epsf, node_epsf_data, node_quality,
                 global_epsf, global_epsf_data, global_quality, wcs=None):
        self.node_x = np.asarray(node_x)
        self.node_y = np.asarray(node_y)
        self.node_epsf = node_epsf          # list; entries may be None (-> global fallback)
        self.node_epsf_data = node_epsf_data
        self.node_quality = node_quality    # list; entries may be None (node had too few stars)
        self.global_epsf = global_epsf
        self.global_epsf_data = global_epsf_data
        self.global_quality = global_quality
        self.wcs = wcs

    def nearest_node(self, x, y):
        d2 = (self.node_x - x) ** 2 + (self.node_y - y) ** 2
        return int(np.argmin(d2))

    def get_epsf(self, x, y):
        """
        Return (epsf_model, epsf_data, used_global) for pixel position
        (x, y) -- the nearest grid node's own ePSF, or the global
        fallback (used_global=True) if that node's build failed or was
        skipped for having too few nearby stars.
        """
        idx = self.nearest_node(x, y)
        if self.node_epsf[idx] is not None:
            return self.node_epsf[idx], self.node_epsf_data[idx], False
        return self.global_epsf, self.global_epsf_data, True

    def get_epsf_radec(self, ra, dec, wcs=None):
        """Same as get_epsf, but takes sky coordinates (needs a WCS --
        either passed here or supplied when this object was built)."""
        wcs = wcs or self.wcs
        if wcs is None:
            raise ValueError('No WCS available -- pass one explicitly or build with wcs=...')
        x, y = wcs.all_world2pix(ra, dec, 0)
        return self.get_epsf(float(x), float(y))

    def node_summary(self):
        """One line per grid node: center, star count, verdict -- a quick sanity check."""
        lines = []
        for i in range(len(self.node_x)):
            q = self.node_quality[i]
            if q is None:
                lines.append(f'  node {i}: ({self.node_x[i]:.0f}, {self.node_y[i]:.0f}) -- '
                              f'too few nearby stars; using global ePSF')
                continue
            status = 'OK' if self.node_epsf[i] is not None else 'FAILED -> global fallback'
            n = q.metrics.get('n_input_stars', '?')
            oversamp = q.metrics.get('oversampling_used', '?')
            lines.append(f'  node {i}: ({self.node_x[i]:.0f}, {self.node_y[i]:.0f}) -- {status} '
                         f'({n} stars, {oversamp}x oversampling, verdict={q.verdict})')
        return '\n'.join(lines)


def build_spatial_epsf(data, stars_tbl, image_shape, nx=3, ny=3,
                        min_stars_per_node=15, sampling_candidates=(3, 2),
                        size=11, wcs=None, **build_epsf_kwargs):
    """
    Build a SpatialEPSF: one real ePSF per node of an nx-by-ny grid
    covering the image, each built ONLY from calibration stars nearer
    to that node than to any other, plus a single global ePSF (built
    from every star) used as the fallback for any node without
    min_stars_per_node of its own.

    Parameters
    ----------
    data : 2D ndarray
        The (background-subtracted) image.
    stars_tbl : astropy.table.Table
        Calibration star positions (needs 'x', 'y' columns), e.g.
        cal_photom.sources after matching_sources().
    image_shape : (ny_pix, nx_pix)
        Full image shape, used to place grid nodes evenly across it.
        Nodes are NOT placed at the image edges -- a node exactly on an
        edge would have systematically fewer/no stars on one side,
        biasing its own star count for a reason unrelated to real PSF
        variation.
    nx, ny : int
        Grid dimensions. 3x3 (default) is a reasonable starting point
        for a 2048x2048 frame with a few hundred calibration stars --
        a finer grid needs proportionally more calibration stars to
        keep min_stars_per_node satisfied at every node.
    min_stars_per_node : int
        A node with fewer calibration stars nearer to it than to any
        other node falls back to the global ePSF instead of building an
        unreliable model from too few stars.
    sampling_candidates, size, **build_epsf_kwargs :
        Passed through to build_epsf_adaptive for every node AND the
        global fallback build.
    wcs : astropy.wcs.WCS or None
        Stored on the returned SpatialEPSF so get_epsf_radec can be
        called without passing a WCS every time.

    Returns
    -------
    SpatialEPSF
    """
    ny_pix, nx_pix = image_shape

    node_x_1d = (np.arange(nx) + 0.5) * (nx_pix / nx)
    node_y_1d = (np.arange(ny) + 0.5) * (ny_pix / ny)
    grid_x, grid_y = np.meshgrid(node_x_1d, node_y_1d)
    node_x = grid_x.ravel()
    node_y = grid_y.ravel()

    star_x = np.asarray(stars_tbl['x'])
    star_y = np.asarray(stars_tbl['y'])

    dist2 = (star_x[:, None] - node_x[None, :]) ** 2 + (star_y[:, None] - node_y[None, :]) ** 2
    assigned_node = np.argmin(dist2, axis=1)

    logger.info(f'build_spatial_epsf: building global fallback ePSF from all {len(stars_tbl)} stars')
    global_epsf_data, global_epsf, global_quality = build_epsf_adaptive(
        data, stars_tbl, sampling_candidates=sampling_candidates, size=size, **build_epsf_kwargs
    )
    if global_epsf is None:
        raise RuntimeError(
            f'build_spatial_epsf: global fallback ePSF build failed '
            f'({"; ".join(global_quality.reasons) if global_quality else "unknown reason"}) -- '
            f'cannot proceed without at least one working ePSF.'
        )

    node_epsf, node_epsf_data, node_quality = [], [], []
    for i in range(len(node_x)):
        node_stars = stars_tbl[assigned_node == i]
        n_here = len(node_stars)

        if n_here < min_stars_per_node:
            logger.info(f'build_spatial_epsf: node {i} ({node_x[i]:.0f}, {node_y[i]:.0f}) has only '
                        f'{n_here} nearest stars (< {min_stars_per_node}); using global ePSF there')
            node_epsf.append(None)
            node_epsf_data.append(None)
            node_quality.append(None)
            continue

        epsf_data, epsf, quality = build_epsf_adaptive(
            data, node_stars, sampling_candidates=sampling_candidates, size=size, **build_epsf_kwargs
        )
        if epsf is None:
            logger.warning(f'build_spatial_epsf: node {i} ({node_x[i]:.0f}, {node_y[i]:.0f}) ePSF '
                            f'build failed ({"; ".join(quality.reasons)}); using global ePSF there '
                            f'instead')
        node_epsf.append(epsf)
        node_epsf_data.append(epsf_data)
        node_quality.append(quality)

    n_ok = sum(1 for e in node_epsf if e is not None)
    logger.info(f'build_spatial_epsf: {n_ok}/{len(node_x)} nodes have their own ePSF; '
                f'{len(node_x) - n_ok} fall back to the global ePSF')

    return SpatialEPSF(node_x, node_y, node_epsf, node_epsf_data, node_quality,
                        global_epsf, global_epsf_data, global_quality, wcs=wcs)