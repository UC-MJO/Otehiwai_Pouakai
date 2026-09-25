"""
Pipeline provenance metadata, for inclusion in output FITS headers.

The pipeline already records THAT a processing step happened (e.g.
REDPIPE='reduction_script', CALPIPE='cal_photom' in the FITS header), but
not exactly which code version, calibration inputs, or parameter set
produced a given output. This module captures that: which master
dark/flat, which sauron state file, which match_tol_px/
isolation_radius_px, and which pipeline code version were used to process
a given frame -- so that if a systematic is found in a batch of frames
later, it's possible to reconstruct exactly what produced them.

`get_pipeline_version()` reports the current git commit hash if the code
is running from a git checkout, falling back to a static version string
otherwise. `build_provenance_dict(...)` packages whatever calibration
inputs/parameters are available into a flat dict of FITS-header-ready
(key, (value, comment)) pairs, truncating string values to the FITS
68-character limit, ready to merge directly into an
`astropy.io.fits.Header`.
"""

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

# Bump this when cutting a release, especially if not deploying from a
# git checkout (in which case get_pipeline_version() falls back to this
# string rather than a commit hash).
PIPELINE_VERSION = '2026.1-dev'

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_cached_git_hash = None
_git_hash_attempted = False


def get_git_commit_hash():
    """
    Return the short git commit hash of the directory containing this
    file, or None if it's not a git checkout or git is unavailable.

    Cached after the first call within a given process, since this
    doesn't change during a pipeline run and calling out to a subprocess
    for every single frame would be wasteful.
    """
    global _cached_git_hash, _git_hash_attempted
    if _git_hash_attempted:
        return _cached_git_hash

    _git_hash_attempted = True
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=_THIS_DIR, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=5, text=True,
        )
        if result.returncode == 0:
            _cached_git_hash = result.stdout.strip()
        else:
            _cached_git_hash = None
    except Exception as e:
        logger.debug(f'Could not determine git commit hash: {e}')
        _cached_git_hash = None

    return _cached_git_hash


def get_pipeline_version():
    """
    Return a best-effort pipeline version string: the git short hash if
    running from a git checkout (preferred -- unambiguous and exact),
    otherwise the static `PIPELINE_VERSION` constant.
    """
    git_hash = get_git_commit_hash()
    if git_hash:
        return f'git:{git_hash}'
    return f'static:{PIPELINE_VERSION}'


def build_provenance_dict(dark_filename=None, flat_filename=None,
                           sauron_state_filename=None,
                           match_tol_px=None, isolation_radius_px=None,
                           zp_floor=None, extra=None):
    """
    Build a flat dict of FITS-header-ready (key, (value, comment)) pairs
    describing exactly what was used to process a frame.

    Only keys corresponding to arguments that were actually supplied are
    included, so callers that don't have a particular piece of provenance
    available (e.g. no calibrimbore sauron state for this stage) simply
    omit it rather than writing a placeholder value.

    Parameters
    ----------
    dark_filename, flat_filename, sauron_state_filename : str or None
        Exact calibration input files used for this frame.
    match_tol_px, isolation_radius_px, zp_floor : float or None
        Key calibration parameters used for this frame's processing.
    extra : dict or None
        Any additional {key: (value, comment)} pairs to merge in (e.g.
        instrument-specific provenance not covered by the standard set
        above).

    Returns
    -------
    dict of {FITS_KEY: (value, comment)}, ready to assign directly into
    an astropy.io.fits.Header.
    """
    d = {
        'PIPEVER': (get_pipeline_version(), 'Pipeline code version (git hash or static)'),
    }

    if dark_filename is not None:
        d['PROVDARK'] = (os.path.basename(str(dark_filename))[:68], 'Master dark file used')
    if flat_filename is not None:
        d['PROVFLAT'] = (os.path.basename(str(flat_filename))[:68], 'Master flat file used')
    if sauron_state_filename is not None:
        d['PROVSAUR'] = (os.path.basename(str(sauron_state_filename))[:68], 'Calibrimbore sauron state file used')
    if match_tol_px is not None:
        d['PRMMTOL'] = (float(match_tol_px), 'match_tol_px parameter used')
    if isolation_radius_px is not None:
        d['PRMISOR'] = (float(isolation_radius_px), 'isolation_radius_px parameter used')
    if zp_floor is not None:
        d['PRMZPFL'] = (float(zp_floor), 'zp_floor parameter used')

    if extra:
        d.update(extra)

    return d