"""
Gaia source queries for astrometric/photometric calibration.

Provides a cone-search function (`gaia_cone`) against the live Gaia
archive, suitable for calling from many parallel worker processes as
part of a pipeline's calibration stage. Three things make this safe and
efficient to call at scale:

1. On-disk caching, keyed on a coarse rounding of the query parameters,
   so repeated or overlapping queries (e.g. adjacent frames of the same
   field, or re-running a pipeline stage) don't hit the network again.
2. Cross-process concurrency limiting (`cross_process_semaphore.
   CrossProcessSemaphore`), so a pipeline running many worker processes
   at once doesn't overwhelm the archive with simultaneous queries.
3. Retry with exponential backoff and jitter, plus an automatic fallback
   to a documented alternate TAP mirror, for resilience against
   transient archive-side issues.

`get_gaia_region` is a simpler, Vizier-based alternative kept for
standalone/fallback use; it is not currently used by the main pipeline
and does not share `gaia_cone`'s caching or concurrency limiting.
"""

import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy import log

from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astroquery.utils.tap.core import TapPlus

import os
import time
import random
import hashlib
import warnings
import logging

from .cross_process_semaphore import CrossProcessSemaphore

logging.getLogger('astroquery').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

logger = logging.getLogger(__name__)

# Default on-disk cache location and concurrency-limiter lock directory.
# Both can be overridden per-call; these defaults exist so caching and
# rate limiting are active out of the box without every caller needing
# to remember to pass them.
_DEFAULT_CACHE_DIR = os.path.expanduser('~/.cache/pouakai_gaia_cache/')
_DEFAULT_SEMAPHORE_DIR = os.path.expanduser('~/.cache/pouakai_gaia_semaphore/')

# Mirrors tried in order after the primary ESA archive. ARI Heidelberg is
# a long-standing, documented alternate Gaia TAP mirror (see e.g.
# astropy/astroquery#1524) used here as an automatic fallback if the
# primary archive is unreachable.
_GAIA_MIRRORS = [
    None,  # None = use astroquery's default GaiaClass (ESA gea.esac.esa.int)
    'http://gaia.ari.uni-heidelberg.de/tap',
]

# Maximum number of worker PROCESSES (not threads) across the whole
# pipeline run allowed to have a Gaia query in flight at the same time.
# Keeping this bounded avoids overwhelming the archive when many workers
# run in parallel. Override via the POUAKAI_GAIA_MAX_CONCURRENT
# environment variable without editing source.
_DEFAULT_MAX_CONCURRENT_QUERIES = int(os.environ.get('POUAKAI_GAIA_MAX_CONCURRENT', '4'))


def _cache_key(ra, dec, radius_arcmin, magnitude_limit):
    """
    Build a cache filename key for a given query, by coarsely rounding
    its parameters.

    Rounding to ~1 arcsec in position and 0.1 arcmin in radius is coarse
    enough that repeated queries for "the same field" reliably hit the
    cache, while still being fine enough that genuinely different
    pointings don't collide.

    `ra`/`dec` are explicitly cast to `float` before rounding: they may
    arrive as 0-d numpy arrays (e.g. from `WCS.all_pix2world` with scalar
    inputs), and Python's built-in `round()` does not accept those
    directly.
    """
    ra = float(ra)
    dec = float(dec)
    radius_arcmin = float(radius_arcmin)
    key_str = f"{round(ra, 4)}_{round(dec, 4)}_{round(radius_arcmin, 1)}_{magnitude_limit}"
    return hashlib.sha1(key_str.encode()).hexdigest()


def get_gaia_region(ra, dec, size=0.4, magnitude_limit=21):
    """
    Get the coordinates and magnitude of all Gaia sources in a field of
    view, via Vizier's Gaia DR2 mirror (catalog I/345/gaia2).

    Parameters
    ----------
    ra, dec : array_like (deg)
        Field center.
    size : float
        Search radius, in arcsec.
    magnitude_limit : float
        Faint cutoff on Gmag.

    Returns
    -------
    result : pandas.DataFrame

    Notes
    -----
    This queries a different service (CDS Vizier) than `gaia_cone` (ESA
    Gaia TAP+), so it does not share that function's on-disk cache or
    cross-process concurrency limiting -- calling this from many parallel
    workers at once would need its own rate limiting. `gaia_cone` is the
    function actually used by the calibration pipeline; this one is kept
    as a standalone utility and is not currently called elsewhere in the
    pipeline.
    """
    c1 = SkyCoord(ra, dec, unit='deg')
    Vizier.ROW_LIMIT = -1

    result = Vizier.query_region(
        c1, catalog=["I/345/gaia2"],
        radius=Angle(size, "arcsec"), column_filters={'Gmag': f'<{magnitude_limit}'},
    )

    if result is None or len(result) == 0:
        raise ValueError(
            'Either no sources were found in the query region or Vizier is unavailable'
        )

    result = result['I/345/gaia2'].to_pandas()
    return result


def _query_mirror(mirror_url, coord, radius_arcmin):
    """
    Run a single cone-search attempt against either the default GaiaClass
    (mirror_url=None) or an explicit alternate TAP server URL.
    """
    if mirror_url is None:
        Gaia.ROW_LIMIT = 1_000_000
        job = Gaia.cone_search_async(coord, radius=u.Quantity(radius_arcmin, u.arcmin))
        return job.get_results().to_pandas()

    tap = TapPlus(url=mirror_url)
    ra_deg, dec_deg = coord.ra.deg, coord.dec.deg
    radius_deg = (radius_arcmin * u.arcmin).to(u.deg).value
    query = (
        "SELECT * FROM gaiadr3.gaia_source WHERE "
        f"1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg}))"
    )
    job = tap.launch_job_async(query)
    return job.get_results().to_pandas()


def gaia_cone(ra, dec, radius_arcmin, magnitude_limit=21, max_retries=5,
              retry_backoff=3.0, retry_backoff_max=60.0,
              cache_dir=_DEFAULT_CACHE_DIR,
              max_concurrent_queries=_DEFAULT_MAX_CONCURRENT_QUERIES,
              semaphore_dir=_DEFAULT_SEMAPHORE_DIR,
              try_mirrors=True):
    """
    Cone search against the live Gaia archive (TAP+, via astroquery.gaia),
    with cross-process concurrency limiting, exponential backoff with
    jitter, and an on-disk cache. This is the function used by the
    calibration pipeline; safe to call from many parallel worker
    processes at once.

    Parameters
    ----------
    ra, dec : float (deg)
        Field center.
    radius_arcmin : float
        Search radius.
    magnitude_limit : float
        Faint cutoff on phot_g_mean_mag, applied client-side after the
        query.
    max_retries : int
        Attempts per mirror before giving up on that mirror and moving to
        the next one (see `try_mirrors`).
    retry_backoff : float
        Base seconds for exponential backoff: attempt n waits
        `min(retry_backoff * 2**(n-1), retry_backoff_max)` seconds, plus
        up to 50% random jitter. Exponential backoff gives the archive
        meaningfully more recovery time on later attempts than a linear
        schedule would, which matters if the archive is actively shedding
        load rather than experiencing a one-off blip; the jitter avoids
        many worker processes retrying in lockstep and re-creating the
        same burst that caused the failure.
    retry_backoff_max : float
        Cap on the exponential backoff, so a high `max_retries` doesn't
        produce absurdly long waits.
    cache_dir : str or None
        Directory to cache query results as CSV files, keyed on a coarse
        rounding of (ra, dec, radius, mag_limit) via `_cache_key`. Enabled
        by default at `~/.cache/pouakai_gaia_cache/`, so repeated or
        overlapping queries across a pipeline run don't need the network
        at all once cached. Pass None to disable.
    max_concurrent_queries : int
        Maximum number of worker PROCESSES (not threads) allowed to have
        a live Gaia query in flight at once, enforced via
        `cross_process_semaphore.CrossProcessSemaphore`. Override with
        the POUAKAI_GAIA_MAX_CONCURRENT environment variable, or pass
        0/None to disable limiting entirely.
    semaphore_dir : str
        Directory for the concurrency limiter's lock files.
    try_mirrors : bool
        If True (default), after exhausting `max_retries` against the
        primary ESA archive, automatically try the ARI Heidelberg mirror
        before giving up entirely.

    Returns
    -------
    gaia_sources : pandas.DataFrame
    """
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(ra, dec, radius_arcmin, magnitude_limit)
        cache_path = os.path.join(cache_dir, f'gaia_{key}.csv')
        if os.path.exists(cache_path):
            try:
                return pd.read_csv(cache_path)
            except Exception as e:
                logger.warning(f'Failed to read Gaia cache file {cache_path} ({e}); re-querying')

    coord = SkyCoord(ra, dec, unit='deg')

    mirrors = _GAIA_MIRRORS if try_mirrors else _GAIA_MIRRORS[:1]

    sem = None
    if max_concurrent_queries:
        sem = CrossProcessSemaphore(semaphore_dir, max_concurrent=max_concurrent_queries)

    last_exc = None
    for mirror_url in mirrors:
        mirror_label = mirror_url or 'ESA primary (gea.esac.esa.int)'

        for attempt in range(1, max_retries + 1):
            try:
                if sem is not None:
                    with sem:
                        gaia_sources = _query_mirror(mirror_url, coord, radius_arcmin)
                else:
                    gaia_sources = _query_mirror(mirror_url, coord, radius_arcmin)

                gaia_sources = gaia_sources[gaia_sources['phot_g_mean_mag'] < magnitude_limit].reset_index(drop=True)

                if cache_dir is not None:
                    try:
                        gaia_sources.to_csv(cache_path, index=False)
                    except Exception as e:
                        logger.warning(f'Failed to write Gaia cache file {cache_path} ({e})')

                return gaia_sources

            except Exception as e:
                last_exc = e
                logger.warning(f'Gaia query attempt {attempt}/{max_retries} against {mirror_label} failed: {e}')
                if attempt < max_retries:
                    backoff = min(retry_backoff * (2 ** (attempt - 1)), retry_backoff_max)
                    backoff *= 1.0 + random.uniform(0, 0.5)  # up to 50% jitter
                    time.sleep(backoff)

        if mirror_url is not mirrors[-1]:
            logger.warning(f'Exhausted {max_retries} attempts against {mirror_label}; trying next mirror')

    raise RuntimeError(
        f'Gaia cone search failed after {max_retries} attempts against each of '
        f'{len(mirrors)} mirror(s): {last_exc}'
    ) from last_exc