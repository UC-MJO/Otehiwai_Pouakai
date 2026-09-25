"""
Astrometric WCS solving via a local astrometry.net `solve-field` install.

`wcs_astrometrynet_local` is the function used by the pipeline: it runs
`solve-field` as a subprocess, using a header-derived RA/Dec position
hint when available (falling back to a blind solve otherwise), enforces
both a soft (`--cpulimit`) and hard (subprocess) timeout so a single
problematic frame can't hang an unattended run indefinitely, and returns
an explicit success/failure result rather than requiring the caller to
infer success from the presence of an output file. Already-solved and
previously-failed frames are tracked via `failure_ledger`, so repeated
pipeline runs don't keep re-attempting the same frames.

`wcs_astrometrynet_local_legacy` is a simpler, deprecated version kept
only for reference; it is not called anywhere in the pipeline.
"""

import os
import re
import subprocess
import logging
from pathlib import Path

from astropy.io import fits

from . import config  # noqa: F401 -- import side effect: ensures
# solve-field's directory is on PATH even if this module is imported
# before anything else in the package happens to import config first
# (see config._ensure_astrometry_on_path's docstring).
from .failure_ledger import record_failure, clear_failure, is_known_failure

logger = logging.getLogger(__name__)

_STAGE_WCS = 'wcs'

tmp = os.environ.get('TMPDIR', '/tmp')

# Header keys tried, in order, for a position hint. Different telescope
# control software uses different conventions; the common ones are tried
# in turn, falling back to a blind solve if none are present/parseable.
_RA_KEYS = ['RA', 'OBJCTRA', 'TELRA', 'CRVAL1']
_DEC_KEYS = ['DEC', 'OBJCTDEC', 'TELDEC', 'CRVAL2']


def _parse_sexagesimal_or_deg(value, is_ra):
    """
    Best-effort parse of a header RA/Dec value that may be a sexagesimal
    string ('12:34:56.7' or '12 34 56.7') or already in decimal degrees.
    Returns degrees, or None if unparseable.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if s == '':
        return None

    # Plain decimal degrees
    try:
        return float(s)
    except ValueError:
        pass

    # Sexagesimal: split on ':' or whitespace
    parts = re.split(r'[:\s]+', s)
    if len(parts) != 3:
        return None

    try:
        sign = -1.0 if parts[0].strip().startswith('-') else 1.0
        h_or_d = abs(float(parts[0]))
        m = float(parts[1])
        sec = float(parts[2])
    except ValueError:
        return None

    value_deg = h_or_d + m / 60.0 + sec / 3600.0
    if is_ra:
        value_deg *= 15.0  # hours -> degrees
    return sign * value_deg


def _get_position_hint(header):
    """
    Try to extract an approximate (ra_deg, dec_deg) pointing from a FITS
    header, for use as a `solve-field` search hint.

    Tries each of `_RA_KEYS`/`_DEC_KEYS` in turn, returning the first
    successfully parsed value for each coordinate. Returns (None, None)
    if nothing usable is found, signalling to the caller that a blind
    solve is needed.
    """
    ra_deg = None
    for key in _RA_KEYS:
        if key in header:
            ra_deg = _parse_sexagesimal_or_deg(header[key], is_ra=True)
            if ra_deg is not None:
                break

    dec_deg = None
    for key in _DEC_KEYS:
        if key in header:
            dec_deg = _parse_sexagesimal_or_deg(header[key], is_ra=False)
            if dec_deg is not None:
                break

    return ra_deg, dec_deg


def wcs_astrometrynet_local(savepath, filename, order=3,
                             search_radius_deg=3.0, cpulimit=300,
                             subprocess_timeout=330, downsample=2,
                             overwrite=True, skip_known_failures=True):
    """
    Calculate the astrometric WCS solution for a reduced FITS frame using
    a local astrometry.net `solve-field` install.

    If a usable RA/Dec position hint can be read from the frame's FITS
    header, it's passed to `solve-field` along with `search_radius_deg`,
    constraining the search around the telescope's actual pointing
    (faster, and less prone to rare false-positive matches than a blind
    solve of the whole sky). If no hint is available, a blind solve is
    attempted instead.

    Frames that are already solved (either awaiting rename as a `.new`
    file, or already renamed and gzipped) are skipped immediately. If
    `skip_known_failures` is set, frames recorded as failed in a previous
    run (via `failure_ledger`) are also skipped without re-attempting
    them.

    Parameters
    ----------
    savepath : str
        Pipeline save-location root; output goes to `savepath/wcs/`.
    filename : str
        Path to the input (reduced, dark/flat-corrected) FITS file.
    order : int
        SIP tweak polynomial order.
    search_radius_deg : float
        Search radius (degrees) around the header-derived position hint.
        Only used if a position hint could be extracted from the header.
    cpulimit : float
        Seconds passed to solve-field's own `--cpulimit`, which tells
        astrometry.net to give up gracefully after this many CPU-seconds.
    subprocess_timeout : float
        Hard wall-clock backstop (seconds) enforced by `subprocess`
        itself, in case `--cpulimit` alone doesn't bound wall-clock time
        (e.g. if the process is I/O- or swap-bound). Should be somewhat
        larger than `cpulimit`.
    downsample : int
        Passed to `--downsample`; speeds up source extraction for large
        images at negligible cost to solve accuracy for typical FOVs.
    overwrite : bool
        Passed to `-O`/`--overwrite` if True.
    skip_known_failures : bool
        If True (default), skip this frame without re-attempting it if it
        is recorded in the failure ledger from a previous run (e.g. a
        frame that previously timed out or had no solution). Set False to
        force a retry, e.g. after increasing cpulimit/subprocess_timeout
        or fixing the position-hint extraction.

    Returns
    -------
    result : dict
        {'success': bool, 'reason': str, 'new_file': str or None,
         'n_match': int or None}
    """
    true_save_path = savepath + 'wcs/'
    os.makedirs(true_save_path, exist_ok=True)
    true_save_path = str(Path(true_save_path))

    base_name = filename.split('/')[-1].split('.fits.gz')[0] + '_wcs'
    new_file = str(Path(true_save_path) / (base_name + '.new'))

    # Skip if this frame has already been solved (either still sitting as
    # a .new file awaiting rename, or already renamed+gzipped by
    # matau.rename_wcs). This keeps repeated pipeline runs cheap: only
    # newly reduced frames need solving, not the entire historical
    # dataset every time.
    final_name = base_name.replace('_wcs', '') + '.fits.gz'
    final_path = str(Path(true_save_path) / final_name)
    if os.path.exists(new_file) or os.path.exists(final_path):
        return {'success': True, 'reason': 'already solved', 'new_file': new_file if os.path.exists(new_file) else None, 'n_match': None}

    if skip_known_failures:
        known_reason = is_known_failure(savepath, _STAGE_WCS, filename)
        if known_reason is not None:
            logger.info(f'{filename}: skipping (known failure: {known_reason})')
            return {'success': False, 'reason': f'skipped (known failure: {known_reason})', 'new_file': None, 'n_match': None}

    cmd = [
        'solve-field',
        '--no-plots',
        '--scale-units', 'arcminwidth',
        '--scale-low', '24',
        '--scale-high', '26',
        '--temp-dir', tmp,
        '-o', base_name,
        '--dir', true_save_path,
        '--tweak-order', str(order),
        '--downsample', str(downsample),
        '--cpulimit', str(cpulimit),
    ]
    if overwrite:
        cmd.append('--overwrite')

    ra_deg, dec_deg = None, None
    try:
        with fits.open(filename, memmap=False) as hdul:
            header = hdul[0].header
            ra_deg, dec_deg = _get_position_hint(header)
    except Exception as e:
        logger.warning(f'{filename}: could not read header for position hint ({e}); falling back to blind solve')

    if ra_deg is not None and dec_deg is not None:
        cmd += ['--ra', str(ra_deg), '--dec', str(dec_deg), '--radius', str(search_radius_deg)]
    else:
        logger.info(f'{filename}: no usable RA/Dec header hint found; blind-solving (slower, slightly higher false-match risk)')

    cmd.append(filename)

    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=subprocess_timeout, text=True,
        )
    except subprocess.TimeoutExpired:
        reason = 'timeout'
        logger.error(f'{filename}: solve-field timed out after {subprocess_timeout}s (wall-clock backstop)')
        record_failure(savepath, _STAGE_WCS, filename, reason)
        return {'success': False, 'reason': reason, 'new_file': None, 'n_match': None}
    except FileNotFoundError:
        reason = 'solve-field not found'
        logger.error('solve-field executable not found on PATH')
        record_failure(savepath, _STAGE_WCS, filename, reason)
        return {'success': False, 'reason': reason, 'new_file': None, 'n_match': None}

    if not os.path.exists(new_file):
        reason = 'no solution (.new file not produced)'
        logger.warning(f'{filename}: {reason}. solve-field output tail:\n{proc.stdout[-1000:] if proc.stdout else ""}')
        record_failure(savepath, _STAGE_WCS, filename, reason)
        return {'success': False, 'reason': reason, 'new_file': None, 'n_match': None}

    n_match = None
    # astrometry.net's `.new` header does not reliably expose a standard
    # match-count/RMS keyword across versions. For per-frame solve-
    # quality tracking, pass `--corr <path>.corr` to solve-field and read
    # the resulting correspondence table (one row per matched source)
    # rather than trying to parse it from the header.

    clear_failure(savepath, _STAGE_WCS, filename)
    return {'success': True, 'reason': 'solved', 'new_file': new_file, 'n_match': n_match}


def wcs_astrometrynet_local_legacy(savepath, filename, order=3):
    """
    Deprecated: a simpler version of `wcs_astrometrynet_local` kept only
    for reference (no position hint, no timeout, no explicit result).
    Not called anywhere in the pipeline -- use `wcs_astrometrynet_local`
    instead.
    """
    true_save_path = savepath + 'wcs/'
    os.makedirs(true_save_path, exist_ok=True)
    true_save_path = str(Path(true_save_path))
    base_name = filename.split('/')[-1].split('.fits.gz')[0] + '_wcs'

    astrom_call = (
        f"solve-field --no-plots --scale-units arcminwidth --scale-low 24 "
        f"--scale-high 26 --temp-dir {tmp} -O -o {base_name} --dir {true_save_path} "
        f"--tweak-order {order} {filename}"
    )

    with subprocess.Popen(astrom_call, stdout=subprocess.PIPE, shell=True) as proc:
        stdout, _ = proc.communicate()