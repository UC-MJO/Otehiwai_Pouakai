"""Filesystem locations and Astrometry.net configuration for Pouakai.

DEFAULTS lists the environment variables handled by this module and their
fallback values. Data-location helpers read the environment on each call and
return absolute paths with user-home expansion. Catalogue, calibration-state and
master-frame directories default to subdirectories of POUAKAI_HOME, which
defaults to pouakai_data/ in the current working directory. Individual
POUAKAI_* directory variables override those defaults. The raw archive requires
POUAKAI_RAW_ARCHIVE_DIR; PYSYN_CDBS is optional and has no default.

require_input_dir() and require_input_file() validate readable inputs;
ensure_output_dir() creates output directories. Directory paths are returned
as strings with trailing separators. ConfigurationError reports invalid
settings, inaccessible inputs and output-directory creation failures.

astrometry_command() returns the solve-field executable and its subprocess
environment, using POUAKAI_ASTROMETRY_BIN when set or PATH otherwise.
validate_astrometry() also checks the engine's configuration and index discovery.
"""

import os
from pathlib import Path
import shutil
import subprocess


# The catalogue, calibration-state, and master defaults are relative to
# POUAKAI_HOME. Its default is relative to the working directory.
DEFAULTS = {
    'POUAKAI_HOME': 'pouakai_data',
    'POUAKAI_RAW_ARCHIVE_DIR': None,  # Required for archive discovery.
    'POUAKAI_CAL_LIST_DIR': 'cal_lists',
    'POUAKAI_CAL_FILES_DIR': 'cal_files',
    'POUAKAI_MASTER_DARK_DIR': 'masters/darks',
    'POUAKAI_MASTER_FLAT_DIR': 'masters/flats',
    'POUAKAI_ASTROMETRY_BIN': None,  # Use PATH when unset or empty.
    'PYSYN_CDBS': None,  # Optional external pysynphot reference data.
}


class ConfigurationError(ValueError):
    """A required deployment setting or input location is unusable."""


def absolute_path(value):
    """Expand a user path against the working directory, preserving symlinks."""
    return os.path.abspath(os.path.expanduser(value))


def _path(value, setting):
    if value is None or not os.fspath(value).strip():
        raise ConfigurationError(f'{setting} must specify a non-empty path.')
    return Path(absolute_path(value))


def _directory_string(path):
    return os.path.join(str(path), '')


def pouakai_home():
    """Data root, defaulting to pouakai_data/ in the caller's working directory."""
    return _directory_string(_path(
        os.environ.get('POUAKAI_HOME', DEFAULTS['POUAKAI_HOME']), 'POUAKAI_HOME'))


def _resolve_dir(env_var):
    value = os.environ.get(env_var)
    if value is None:
        value = Path(pouakai_home()) / DEFAULTS[env_var]
    return _directory_string(_path(value, env_var))


def raw_archive_dir():
    """Resolve the explicitly configured raw archive; never create it."""
    value = os.environ.get('POUAKAI_RAW_ARCHIVE_DIR', DEFAULTS['POUAKAI_RAW_ARCHIVE_DIR'])
    if value is None:
        raise ConfigurationError(
            'Set POUAKAI_RAW_ARCHIVE_DIR or pass fli_dir=... to organise_fli_files.')
    return _directory_string(_path(value, 'POUAKAI_RAW_ARCHIVE_DIR'))


def cal_list_dir():
    return _resolve_dir('POUAKAI_CAL_LIST_DIR')


def cal_files_dir():
    return _resolve_dir('POUAKAI_CAL_FILES_DIR')


def master_dark_dir():
    return _resolve_dir('POUAKAI_MASTER_DARK_DIR')


def master_flat_dir():
    return _resolve_dir('POUAKAI_MASTER_FLAT_DIR')


def pysyn_cdbs_dir():
    """Return the optional, user-supplied CDBS location without setting it."""
    value = os.environ.get('PYSYN_CDBS', DEFAULTS['PYSYN_CDBS'])
    return None if value is None else _directory_string(_path(value, 'PYSYN_CDBS'))


def require_input_dir(value, setting):
    path = _path(value, setting)
    if not path.is_dir():
        raise ConfigurationError(f'{setting}: input directory does not exist or is not a directory: {path}')
    if not os.access(path, os.R_OK | os.X_OK):
        raise ConfigurationError(f'{setting}: input directory is not readable/searchable: {path}')
    return _directory_string(path)


def require_input_file(value, setting):
    path = _path(value, setting)
    if not path.is_file():
        raise ConfigurationError(f'{setting}: input file does not exist or is not a regular file: {path}')
    if not os.access(path, os.R_OK):
        raise ConfigurationError(f'{setting}: input file is not readable: {path}')
    return str(path)


def ensure_output_dir(value, setting):
    """Create an output directory only when its stage is about to write."""
    path = _path(value, setting)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(f'{setting}: cannot create output directory {path}: {exc}') from exc
    return _directory_string(path)


def catalogue_file(filename):
    """Require an existing input catalogue, without creating its parent."""
    return require_input_file(Path(cal_list_dir()) / filename, 'POUAKAI_CAL_LIST_DIR')


def validate_calibration_inputs():
    """Require calibration states and, if configured, an existing CDBS tree."""
    states = require_input_dir(cal_files_dir(), 'POUAKAI_CAL_FILES_DIR')
    cdbs = pysyn_cdbs_dir()
    if cdbs is not None:
        require_input_dir(cdbs, 'PYSYN_CDBS')
    return states


def astrometry_command():
    """Return solve-field and its child environment, leaving os.environ alone.

    An explicit POUAKAI_ASTROMETRY_BIN takes precedence and must contain an
    executable solve-field. Otherwise use PATH (including an active conda env).
    Prepend an explicit directory only in the child, for Astrometry helpers.
    """
    env = os.environ.copy()
    directory = env.get('POUAKAI_ASTROMETRY_BIN', DEFAULTS['POUAKAI_ASTROMETRY_BIN'])
    if directory:
        directory = require_input_dir(directory, 'POUAKAI_ASTROMETRY_BIN')
        executable = shutil.which(str(Path(directory) / 'solve-field'))
        env['PATH'] = os.pathsep.join([str(Path(directory)), env.get('PATH', '')])
    else:
        executable = shutil.which('solve-field')
    if executable is None:
        raise ConfigurationError(
            'solve-field is not executable or was not found; install Astrometry.net on PATH '
            'or set POUAKAI_ASTROMETRY_BIN to its bin directory.')
    return executable, env


def validate_astrometry(timeout=30):
    """Check the configured indexes without solving any images.

    Match solve-field's engine lookup: beside solve-field first, then PATH.
    An empty input list makes the engine validate its usual astrometry.cfg and
    discover indexes, then exit without processing images.
    """
    executable, env = astrometry_command()
    sibling = Path(executable).absolute().with_name('astrometry-engine')
    engine = shutil.which(str(sibling)) or shutil.which('astrometry-engine', path=env.get('PATH'))
    if engine is None:
        raise ConfigurationError(
            f'astrometry-engine was not found beside {executable} or on its PATH; '
            'install the complete Astrometry.net package.')
    try:
        result = subprocess.run(
            [engine, '--inputs-from', '-'], input='', text=True, errors='replace',
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConfigurationError(
            f'Astrometry.net index check timed out after {timeout}s: {engine}. '
            'Check astrometry.cfg and access to its index directories.') from exc
    except OSError as exc:
        raise ConfigurationError(f'Cannot run Astrometry.net index check: {engine}: {exc}') from exc
    if result.returncode != 0:
        detail = (result.stdout or '').strip()[-4000:]
        raise ConfigurationError(
            f'Astrometry.net index check failed (exit {result.returncode}): {engine}. '
            'Check astrometry.cfg and ensure its configured index files are installed '
            f'and readable.\n{detail}')
