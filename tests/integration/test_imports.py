"""Integration tests for public imports and deferred calibration dependencies."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from otehiwai_pouakai import config


pytestmark = pytest.mark.integration


_SUPPORT_DIR = Path(__file__).resolve().parents[1]


def _run_python_check(script, tmp_path, *args, env, timeout=40):
    """Run a standalone check with fresh imports and report its captured output."""
    command = [sys.executable, '-B', str(_SUPPORT_DIR / script), *args]
    result = subprocess.run(command, env=env, cwd=tmp_path, text=True,
                            capture_output=True, timeout=timeout)
    assert result.returncode == 0, (
        f'{script} {args} exited with {result.returncode}\n{result.stdout}{result.stderr}')


@pytest.mark.parametrize('configured', [False, True])
@pytest.mark.parametrize('action', ['version', 'exports', 'help'])
def test_import_and_help_without_io_or_science_dependencies(tmp_path, configured, action):
    env = os.environ.copy()
    if configured:
        for name in ('POUAKAI_RAW_ARCHIVE_DIR', 'POUAKAI_CAL_LIST_DIR',
                     'POUAKAI_CAL_FILES_DIR', 'POUAKAI_MASTER_DARK_DIR',
                     'POUAKAI_MASTER_FLAT_DIR', 'POUAKAI_ASTROMETRY_BIN', 'PYSYN_CDBS'):
            env[name] = str(tmp_path / 'missing' / name)
    else:
        env.pop('POUAKAI_HOME', None)
    _run_python_check('test_support_imports.py', tmp_path, action,
                      env=env, timeout=20)
    assert not list(tmp_path.iterdir())


def test_processing_imports_do_not_load_catalogue_clients(tmp_path):
    env = os.environ.copy()
    env['POUAKAI_RAW_ARCHIVE_DIR'] = str(tmp_path / 'absent-archive')
    env['POUAKAI_GAIA_MAX_CONCURRENT'] = 'not-configured-yet'
    _run_python_check('test_support_imports.py', tmp_path, 'processing', env=env)


def test_deferred_calibration_import_loads_real_state(tmp_path):
    env = os.environ.copy()
    # No reference data is needed, but an empty local directory prevents
    # pysynphot's import-time fallback lookup of remote CDBS metadata.
    cdbs = tmp_path / 'empty-cdbs'
    cdbs.mkdir()
    env['PYSYN_CDBS'] = str(cdbs)
    _run_python_check('test_support_calibration.py', tmp_path, 'saved-state', env=env)


def test_calibration_inputs_are_read_only(monkeypatch, tmp_path):
    from otehiwai_pouakai.calibration_saurus import cal_photom

    cal_photom(run=False)
    states = Path(config.cal_files_dir())
    with pytest.raises(config.ConfigurationError, match='POUAKAI_CAL_FILES_DIR'):
        cal_photom()
    assert not states.exists()
    states.mkdir(parents=True)
    monkeypatch.setenv('PYSYN_CDBS', str(tmp_path / 'missing-cdbs'))
    with pytest.raises(config.ConfigurationError, match='PYSYN_CDBS'):
        config.validate_calibration_inputs()
    assert not (tmp_path / 'missing-cdbs').exists()
    monkeypatch.delenv('PYSYN_CDBS')
    assert Path(config.validate_calibration_inputs()) == states


def test_calibration_state_validated_before_loading_client(tmp_path):
    env = os.environ.copy()
    env['POUAKAI_CAL_FILES_DIR'] = str(tmp_path / 'states')
    _run_python_check('test_support_calibration.py', tmp_path, 'state-validation',
                      env=env)
