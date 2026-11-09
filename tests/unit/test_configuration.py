"""Unit tests for path resolution and dependency checks."""

import os
from pathlib import Path
import subprocess

import pytest

from otehiwai_pouakai import config


pytestmark = pytest.mark.unit


def test_home_and_overrides_resolve_without_creating(monkeypatch, tmp_path):
    monkeypatch.delenv('POUAKAI_HOME')
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / 'pouakai_data'
    assert Path(config.pouakai_home()) == expected
    assert Path(config.cal_list_dir()) == expected / 'cal_lists'
    assert Path(config.master_dark_dir()) == expected / 'masters/darks'
    assert Path(config.master_flat_dir()) == expected / 'masters/flats'
    assert Path(config.cal_files_dir()) == expected / 'cal_files'
    monkeypatch.setenv('POUAKAI_HOME', '~/second')
    assert Path(config.master_dark_dir()) == tmp_path / 'home/second/masters/darks'
    assert Path(config.master_flat_dir()) == tmp_path / 'home/second/masters/flats'
    assert Path(config.cal_files_dir()) == tmp_path / 'home/second/cal_files'
    monkeypatch.setenv('POUAKAI_CAL_LIST_DIR', str(tmp_path / 'lists-override'))
    assert Path(config.cal_list_dir()) == tmp_path / 'lists-override'
    assert config.cal_list_dir().endswith(os.sep)
    assert config.pysyn_cdbs_dir() is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('variable,helper', [
    ('POUAKAI_HOME', config.pouakai_home),
    ('POUAKAI_CAL_LIST_DIR', config.cal_list_dir),
    ('POUAKAI_RAW_ARCHIVE_DIR', config.raw_archive_dir),
])
def test_empty_paths_are_errors(monkeypatch, variable, helper):
    monkeypatch.setenv(variable, '')
    with pytest.raises(config.ConfigurationError, match=variable):
        helper()


def test_astrometry_prefers_path_unless_explicit(monkeypatch, tmp_path, fake_astrometry):
    path_solver = fake_astrometry(tmp_path / 'on-path')
    explicit_solver = fake_astrometry(tmp_path / 'explicit')
    monkeypatch.setenv('PATH', str(path_solver.parent))
    original = dict(os.environ)
    executable, env = config.astrometry_command()
    assert executable == str(path_solver)
    assert env == original == dict(os.environ)
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(explicit_solver.parent))
    original = dict(os.environ)
    executable, env = config.astrometry_command()
    assert executable == str(explicit_solver)
    assert env['PATH'].split(os.pathsep)[0] == str(explicit_solver.parent)
    assert dict(os.environ) == original
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', '')
    assert config.astrometry_command()[0] == str(path_solver)
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(tmp_path / 'missing'))
    with pytest.raises(config.ConfigurationError, match='POUAKAI_ASTROMETRY_BIN'):
        config.astrometry_command()


def test_missing_engine_reports_configuration_error(monkeypatch, tmp_path, fake_astrometry):
    solver = fake_astrometry(tmp_path / 'astrometry')
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(solver.parent))
    monkeypatch.setenv('PATH', str(solver.parent))
    (solver.parent / 'astrometry-engine').unlink()
    with pytest.raises(config.ConfigurationError, match='astrometry-engine was not found'):
        config.validate_astrometry()


@pytest.mark.parametrize('timeout,error,message', [
    pytest.param(None, OSError('Exec format error'), 'Cannot run Astrometry.net index check',
                 id='launch-error'),
    pytest.param(None, subprocess.TimeoutExpired('astrometry-engine', 30),
                 'index check timed out', id='default-timeout'),
    pytest.param(0.25, subprocess.TimeoutExpired('astrometry-engine', 0.25),
                 'index check timed out', id='custom-timeout'),
])
def test_engine_errors_report_configuration_error(monkeypatch, tmp_path, timeout, error, message,
                                                 fake_astrometry):
    solver = fake_astrometry(tmp_path / 'astrometry')
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(solver.parent))

    def fail_to_run(*args, **kwargs):
        assert kwargs['timeout'] == (30 if timeout is None else timeout)
        raise error

    monkeypatch.setattr(config.subprocess, 'run', fail_to_run)
    with pytest.raises(config.ConfigurationError, match=message):
        if timeout is None:
            config.validate_astrometry()
        else:
            config.validate_astrometry(timeout=timeout)
