"""Offline tests for pipeline configuration, dependency checks, and filesystem behaviour."""

import os
from pathlib import Path
import sys
import socket
import subprocess

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from otehiwai_pouakai import config, Pouakai


_TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def local_environment(monkeypatch, tmp_path):
    """Isolate each test's settings and data paths, and block socket connections."""
    for name in os.environ:
        if name.startswith('POUAKAI_') or name == 'PYSYN_CDBS':
            monkeypatch.delenv(name)
    for name, directory in {
        'HOME': 'home', 'POUAKAI_HOME': 'data', 'MPLCONFIGDIR': 'mpl',
        'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config',
        'IPYTHONDIR': 'ipython',
    }.items():
        monkeypatch.setenv(name, str(tmp_path / directory))
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '1')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')

    def no_network(*args, **kwargs):
        raise AssertionError('Tests must not access the network')

    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(socket, 'getaddrinfo', no_network)


def _run_python_check(script, tmp_path, *args, env, no_site=False, timeout=40):
    """Run a standalone check with fresh imports and report its captured output."""
    env = env.copy()
    env['PYTHONPATH'] = str(_TEST_DIR.parent / 'src')
    command = [sys.executable, '-B']
    if no_site:
        command.append('-S')
    command.extend([str(_TEST_DIR / script), *args])
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
    # -S also proves these public entry points work without site-packages.
    _run_python_check('test_support_imports.py', tmp_path, action,
                      env=env, no_site=True, timeout=20)
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


def test_archive_is_required_and_never_created(monkeypatch, tmp_path):
    from otehiwai_pouakai.organise_files import organise_fli_files

    with pytest.raises(config.ConfigurationError, match='POUAKAI_RAW_ARCHIVE_DIR'):
        organise_fli_files()
    absent = tmp_path / 'absent-archive'
    monkeypatch.setenv('POUAKAI_RAW_ARCHIVE_DIR', str(absent))
    with pytest.raises(config.ConfigurationError, match='input directory'):
        organise_fli_files()
    assert not absent.exists()
    assert not Path(config.cal_list_dir()).exists()
    regular_file = tmp_path / 'not-a-directory'
    regular_file.write_text('')
    with pytest.raises(config.ConfigurationError, match='input directory'):
        organise_fli_files(fli_dir=regular_file)


def test_discovery_uses_settings_changed_after_import(monkeypatch, tmp_path):
    from otehiwai_pouakai.organise_files import organise_fli_files

    raw = tmp_path / 'raw'
    raw.mkdir()
    header = fits.Header({'IMAGETYP': 'science', 'EXPTIME': 30, 'JD': 2460000,
                          'DATE-OBS': '2023-02-24', 'READOUTM': '2MHz', 'FILTER': 'V'})
    fits.writeto(raw / 'science.fits', np.ones((8, 8)), header)
    monkeypatch.setenv('POUAKAI_RAW_ARCHIVE_DIR', str(tmp_path / 'wrong'))
    for name in ('first', 'second'):
        lists = tmp_path / name / 'lists'
        monkeypatch.setenv('POUAKAI_CAL_LIST_DIR', str(lists))
        organise_fli_files(fli_dir=raw)
        catalogue = pd.read_csv(lists / 'bc_science_image_list.csv')
        assert catalogue['filename'].tolist() == [str(raw / 'science.fits')]
    assert not Path(config.pouakai_home()).exists()
    assert not (tmp_path / 'wrong').exists()


def test_discovery_defaults_to_current_project_directory(monkeypatch, tmp_path):
    from otehiwai_pouakai.organise_files import organise_fli_files

    # Changing directories after importing must select the calling project's data.
    raw = tmp_path / 'raw'
    raw.mkdir()
    monkeypatch.delenv('POUAKAI_HOME')
    for name in ('first-project', 'second-project'):
        project = tmp_path / name
        project.mkdir()
        monkeypatch.chdir(project)
        organise_fli_files(fli_dir=raw)
        lists = project / 'pouakai_data' / 'cal_lists'
        assert (lists / 'bc_all_image_list.csv').is_file()
        assert pd.read_csv(lists / 'bc_science_image_list.csv').empty
        assert not (project / 'pouakai_data' / 'masters').exists()
    assert not (tmp_path / 'home').exists()
    assert not (tmp_path / 'pouakai_data').exists()


def test_relative_science_inputs_match_absolute_catalogue_paths(monkeypatch, tmp_path):
    from otehiwai_pouakai.matau import get_file_paths, update_df
    from otehiwai_pouakai.organise_files import organise_fli_files

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('POUAKAI_RAW_ARCHIVE_DIR', 'raw')
    monkeypatch.setenv('POUAKAI_CAL_LIST_DIR', 'lists')
    Path('raw').mkdir()
    relative = Path('raw/science.fits')
    absolute = tmp_path / relative
    fits.writeto(relative, np.ones((8, 8)), fits.Header({
        'IMAGETYP': 'science', 'EXPTIME': 30, 'JD': 2460000.,
        'DATE-OBS': '2023-02-24', 'READOUTM': '2MHz', 'FILTER': 'V'}))
    organise_fli_files()
    for name in ('all', 'science'):
        catalogue = pd.read_csv(tmp_path / 'lists' / f'bc_{name}_image_list.csv')
        assert catalogue['filename'].tolist() == [str(absolute)]

    for inputs in ([str(relative)], [relative], [absolute],
                   ['raw/../raw/science.fits'], get_file_paths('raw/*.fits')):
        selected = update_df(inputs)
        assert selected['filename'].tolist() == [str(absolute)]
    assert get_file_paths('raw/*.fits') == [str(absolute)]


@pytest.mark.parametrize('kind', ['dark', 'flat', 'science'])
def test_missing_input_catalogues_fail_without_creating(kind):
    from otehiwai_pouakai.dark_masters import get_master_dark, make_master_darks
    from otehiwai_pouakai.flat_masters import get_master_flat, make_master_flats
    from otehiwai_pouakai.matau import update_df

    calls = {
        'dark': [make_master_darks, lambda: get_master_dark(2460000, 30, '2MHz')],
        'flat': [make_master_flats, lambda: get_master_flat(2460000, '2MHz', 'V')],
        'science': [lambda: update_df([])],
    }
    for call in calls[kind]:
        with pytest.raises(config.ConfigurationError, match='POUAKAI_CAL_LIST_DIR'):
            call()
    assert not Path(config.pouakai_home()).exists()


def test_noop_pipeline_does_not_require_inputs_or_create_outputs(tmp_path):
    output = tmp_path / 'output'
    Pouakai([], output, run=False, organise_files=False, make_masters=False)
    assert not output.exists()
    assert not Path(config.pouakai_home()).exists()


def test_invalid_mode_fails_before_setup(tmp_path):
    with pytest.raises(ValueError, match='mode must be'):
        Pouakai([], tmp_path / 'output', mode='invalid')
    assert not (tmp_path / 'output').exists()


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


def _solver(directory):
    directory.mkdir()
    for name, script in {
        'solve-field': 'test_support_solve_field.py',
        'astrometry-engine': 'test_support_astrometry_engine.py',
    }.items():
        executable = directory / name
        executable.write_text(f'#!{sys.executable}\n' + (_TEST_DIR / script).read_text())
        executable.chmod(0o755)
    helper = directory / 'pouakai-test-helper'
    helper.write_text('#!/bin/sh\nexit 0\n')
    helper.chmod(0o755)
    return directory / 'solve-field'


def test_astrometry_prefers_path_unless_explicit(monkeypatch, tmp_path):
    path_solver = _solver(tmp_path / 'on-path')
    explicit_solver = _solver(tmp_path / 'explicit')
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


@pytest.mark.parametrize('input_source', ['files', 'glob', 'directory'])
def test_wcs_only_needs_solver_and_stage_inputs(monkeypatch, tmp_path, input_source):
    solver = _solver(tmp_path / 'astrometry')
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(solver.parent))
    project = tmp_path / 'project with spaces'
    project.mkdir()
    monkeypatch.chdir(project)
    input_dir = project / 'input'
    input_dir.mkdir()
    input_file = input_dir / 'reduced.fits.gz'
    fits.writeto(input_file, np.ones((8, 8)))
    overrides = {
        'files': {'wcs_input_files': [Path('input/reduced.fits.gz')]},
        'glob': {'wcs_input_glob': 'input/*.fits.gz'},
        'directory': {'wcs_input_dir': 'input'},
    }
    output = project / 'output'
    original_path = os.environ['PATH']
    Pouakai([], 'output/', mode='wcs', organise_files=False, make_masters=False,
            **overrides[input_source])
    products = list((output / 'wcs').glob('*.fits.gz'))
    assert len(products) == 1
    np.testing.assert_array_equal(fits.getdata(products[0]), np.ones((8, 8)))
    manifest = pd.read_csv(output / 'logs/manifest.csv')
    assert manifest['input_path'].tolist() == [str(input_file)]
    assert os.environ['PATH'] == original_path
    assert not Path(config.pouakai_home()).exists()
    assert not (output / 'cal').exists()
    assert not (output / 'red').exists()


def test_index_check_matches_solver_engine_lookup(monkeypatch, tmp_path):
    path_solver = _solver(tmp_path / 'on-path')
    explicit_solver = _solver(tmp_path / 'explicit')
    monkeypatch.setenv('PATH', str(path_solver.parent))
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(explicit_solver.parent))
    original = dict(os.environ)
    config.validate_astrometry()
    assert (explicit_solver.parent / 'index-checks').read_text() == 'checked\n'
    assert not (path_solver.parent / 'index-checks').exists()
    # solve-field falls back to PATH if its own directory has no engine.
    (explicit_solver.parent / 'astrometry-engine').unlink()
    config.validate_astrometry()
    assert (path_solver.parent / 'index-checks').read_text() == 'checked\n'
    assert dict(os.environ) == original


@pytest.mark.parametrize('mode', ['modulo', 'wcs'])
def test_missing_indexes_fail_before_any_processing(monkeypatch, tmp_path, mode):
    solver = _solver(tmp_path / 'astrometry')
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(solver.parent))

    def missing_indexes(command, **kwargs):
        return subprocess.CompletedProcess(command, 255, stdout=(
            'You must list at least one index in the config file (example.cfg)\n'))

    def unexpected_stage(*args, **kwargs):
        pytest.fail('Processing started before the index check')

    monkeypatch.setattr(config.subprocess, 'run', missing_indexes)
    monkeypatch.setattr(Pouakai, '_timed_stage', unexpected_stage)
    monkeypatch.setattr(Pouakai, '_run_reduction', unexpected_stage)
    output = tmp_path / 'output'
    with pytest.raises(config.ConfigurationError, match='example.cfg'):
        Pouakai(['raw.fit'], output, mode=mode)
    assert not output.exists()  # No products or failed-science-frame ledger.
    assert not Path(config.pouakai_home()).exists()


def test_missing_engine_reports_configuration_error(monkeypatch, tmp_path):
    solver = _solver(tmp_path / 'astrometry')
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
def test_engine_errors_report_configuration_error(monkeypatch, tmp_path, timeout, error, message):
    solver = _solver(tmp_path / 'astrometry')
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


def test_index_check_runs_once_per_batch_and_again_on_next_call(monkeypatch, tmp_path):
    solver = _solver(tmp_path / 'astrometry')
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(solver.parent))
    files = [tmp_path / f'frame-{i}.fits.gz' for i in range(3)]
    for filename in files:
        fits.writeto(filename, np.ones((8, 8)))
    for run in range(2):
        output = tmp_path / f'output-{run}'
        Pouakai([], output, mode='wcs', organise_files=False, make_masters=False,
                num_cores=2, wcs_input_files=files)
        assert len(list((output / 'wcs').glob('*.fits.gz'))) == len(files)
        assert (solver.parent / 'index-checks').read_text() == 'checked\n' * (run + 1)


def test_reduction_only_does_not_require_astrometry(monkeypatch, tmp_path):
    from otehiwai_pouakai import matau

    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(tmp_path / 'missing-solver'))
    monkeypatch.setattr(matau, 'update_df', lambda files: pd.DataFrame())
    Pouakai([], tmp_path / 'output', mode='red', organise_files=False, make_masters=False)
    assert not (tmp_path / 'output').exists()


def test_missing_stage_directory_is_not_created(monkeypatch, tmp_path):
    solver = _solver(tmp_path / 'astrometry')
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(solver.parent))
    with pytest.raises(config.ConfigurationError, match='stage input_dir'):
        Pouakai([], tmp_path / 'output', mode='wcs', organise_files=False,
                make_masters=False, wcs_input_dir=tmp_path / 'missing')
    assert not (tmp_path / 'output').exists()
    assert not (tmp_path / 'missing').exists()


def test_unusable_solver_fails_before_creating_outputs(monkeypatch, tmp_path):
    from otehiwai_pouakai.wcs_compute import wcs_astrometrynet_local

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(bin_dir))
    for contents in (None, '#!/bin/sh\nexit 0\n'):
        if contents is not None:
            (bin_dir / 'solve-field').write_text(contents)  # No execute permission.
        with pytest.raises(config.ConfigurationError, match='solve-field'):
            wcs_astrometrynet_local(str(tmp_path / 'output'), 'input.fits')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('stage', ['wcs', 'cal'])
@pytest.mark.parametrize('status', ['completed', 'known_failure'])
@pytest.mark.parametrize('batch', [False, True])
def test_skipped_frames_do_not_require_unused_dependencies(monkeypatch, tmp_path, stage, status, batch):
    from otehiwai_pouakai.core_reduction import calibrating_internal
    from otehiwai_pouakai.failure_ledger import record_failure
    from otehiwai_pouakai.wcs_compute import wcs_astrometrynet_local

    output = tmp_path / 'output'
    filename = str(tmp_path / ('frame.fits.gz' if stage == 'wcs' else 'frame_wcs.fits.gz'))
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(tmp_path / 'missing-solver'))
    monkeypatch.setenv('POUAKAI_CAL_FILES_DIR', str(tmp_path / 'missing-states'))
    monkeypatch.setenv('PYSYN_CDBS', str(tmp_path / 'missing-cdbs'))
    if batch and stage == 'wcs':
        # Pipeline startup checks the installation even on a completed run;
        # direct per-frame calls still skip before checking dependencies.
        solver = _solver(tmp_path / 'astrometry')
        monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(solver.parent))
    if status == 'completed':
        # Extend this case to renamed WCS .fits.gz products when fixing issue #19:
        # https://github.com/UC-MJO/Otehiwai_Pouakai/issues/19
        product = output / ('wcs/frame_wcs.new' if stage == 'wcs' else 'cal/frame_cal.fits.gz')
        product.parent.mkdir(parents=True)
        fits.writeto(product, np.ones((8, 8)))
    else:
        ledger_stage = 'wcs' if stage == 'wcs' else 'calibration'
        record_failure(output, ledger_stage, filename, 'previous failure')

    if batch:
        Pouakai([], output, mode=stage, organise_files=False, make_masters=False,
                **{f'{stage}_input_files': [filename]})
    elif stage == 'wcs':
        result = wcs_astrometrynet_local(str(output), filename)
        assert result['success'] == (status == 'completed')
    else:
        assert calibrating_internal(filename, str(output)) is None

    if status == 'completed':
        if stage == 'wcs' and batch:
            product = output / 'wcs/frame_wcs.fits.gz'
        np.testing.assert_array_equal(fits.getdata(product), np.ones((8, 8)))
    else:
        assert not (output / stage).exists()
    assert not (output / 'phot_table').exists()
    assert not (output / 'zp').exists()
    assert not (tmp_path / 'missing-solver').exists()
    assert not (tmp_path / 'missing-states').exists()
    assert not (tmp_path / 'missing-cdbs').exists()
    assert not Path(config.pouakai_home()).exists()


@pytest.mark.parametrize('stage', ['wcs', 'cal'])
def test_pending_batch_requires_stage_dependencies(monkeypatch, tmp_path, stage):
    monkeypatch.setenv('POUAKAI_ASTROMETRY_BIN', str(tmp_path / 'missing-solver'))
    output = tmp_path / 'output'
    setting = 'POUAKAI_ASTROMETRY_BIN' if stage == 'wcs' else 'POUAKAI_CAL_FILES_DIR'
    with pytest.raises(config.ConfigurationError, match=setting):
        Pouakai([], output, mode=stage, organise_files=False, make_masters=False,
                **{f'{stage}_input_files': ['pending.fits.gz']})
    assert not output.exists()


def test_failure_ledger_reads_do_not_create_outputs(tmp_path):
    from otehiwai_pouakai import failure_ledger as ledger

    output = tmp_path / 'output'
    assert ledger.is_known_failure(output, 'wcs', 'frame.fits.gz') is None
    assert ledger.load_known_failures(output, 'wcs') == set()
    assert ledger.load_all_failures(output).empty
    assert ledger.summarize(output).empty
    ledger.clear_failure(output, 'wcs', 'frame.fits.gz')
    assert not output.exists()

    ledger.record_failure(output, 'wcs', 'frame.fits.gz', 'test failure')
    assert ledger.is_known_failure(output, 'wcs', 'frame.fits.gz') == 'test failure'
    ledger.clear_failure(output, 'wcs', 'frame.fits.gz')
    assert ledger.is_known_failure(output, 'wcs', 'frame.fits.gz') is None


def test_calibration_state_validated_before_loading_client(tmp_path):
    env = os.environ.copy()
    env['POUAKAI_CAL_FILES_DIR'] = str(tmp_path / 'states')
    _run_python_check('test_support_calibration.py', tmp_path, 'state-validation',
                      env=env)


def test_master_writers_create_only_their_output_directory(monkeypatch, tmp_path):
    from otehiwai_pouakai.dark_masters import _bc_master_darks
    from otehiwai_pouakai.flat_masters import _bc_master_flats

    files = []
    for i in range(3):
        filename = tmp_path / f'input-{i}.fits'
        fits.writeto(filename, np.full((8, 8), i + 1.))
        files.append(str(filename))
    frames = pd.DataFrame({'filename': files, 'master_name': 'example', 'readout': '2MHz',
                           'jd': 2460000., 'exptime': 30., 'date': '2023-02-24', 'shape': 8,
                           'band': 'V', 'master_dark': files[0]})
    dark_dir = tmp_path / 'dark outputs'
    flat_dir = tmp_path / 'flat outputs'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('POUAKAI_MASTER_DARK_DIR', dark_dir.name)
    monkeypatch.setenv('POUAKAI_MASTER_FLAT_DIR', flat_dir.name)
    assert _bc_master_darks(frames.iloc[:2]) is None
    assert _bc_master_flats(frames.iloc[:2]) is None
    assert not dark_dir.exists() and not flat_dir.exists()
    dark = _bc_master_darks(frames)
    assert Path(dark.iloc[0]['filename']).parent == dark_dir
    assert Path(dark.iloc[0]['filename']).is_file()
    assert not flat_dir.exists()
    # Non-identical, positive flat frames avoid the separately tracked master-clipping bug.
    for i, filename in enumerate(files):
        fits.writeto(filename, np.random.default_rng(i).uniform(10, 20, (8, 8)), overwrite=True)
    frames['master_dark'] = dark.iloc[0]['filename']
    flat = _bc_master_flats(frames)
    assert Path(flat.iloc[0]['filename']).parent == flat_dir
    assert Path(flat.iloc[0]['filename']).is_file()
    assert np.isfinite(fits.getdata(flat.iloc[0]['filename'])).all()
    assert not Path(config.pouakai_home()).exists()


@pytest.mark.parametrize('compression_fails', [False, True])
def test_reduction_records_output_only_after_compression(monkeypatch, tmp_path, compression_fails):
    from otehiwai_pouakai import core_reduction

    if compression_fails:
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        gzip = bin_dir / 'gzip'
        gzip.write_text('#!/bin/sh\nexit 1\n')
        gzip.chmod(0o755)
        monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])

    science, dark, flat = [tmp_path / f'{name}.fits' for name in ('science', 'dark', 'flat')]
    for filename, value in ((science, 100.), (dark, 10.), (flat, 2.)):
        fits.writeto(filename, np.full((16, 16), value))
    lists = Path(config.cal_list_dir())
    lists.mkdir(parents=True)
    shared = {'jd': 2460000., 'exptime': 30., 'readout': '2MHz', 'shape': 16, 'band': 'V'}
    for name, filename in (('dark', dark), ('flat', flat)):
        pd.DataFrame([{**shared, 'filename': str(filename)}]).to_csv(
            lists / f'bc_master_{name}_list.csv', index=False)
    frames = pd.DataFrame([{**shared, 'filename': str(science), 'master_name': 'science'}])
    monkeypatch.setattr(core_reduction, 'determine_background',
                        lambda image, **kwargs: (np.zeros_like(image), np.ones_like(image)))
    monkeypatch.setattr(core_reduction, 'build_source_and_nebula_masks',
                        lambda image: (None, None, np.zeros_like(image, dtype=bool)))
    monkeypatch.setattr(core_reduction, 'background_residual_flatness', lambda *a, **k: (0., None))
    output = tmp_path / 'output with spaces'
    core_reduction.reduction_script(frames, 0, str(output), extra_conds={
        'dark_exp_tol': 1, 'dark_date_tol': 3, 'flat_date_tol': 30, 'shape': 16})
    product = output / 'red/science_reduced.fits.gz'
    manifest = pd.read_csv(output / 'logs/manifest.csv')
    if compression_fails:
        assert not product.exists()
        assert product.with_suffix('').is_file()
        assert manifest['status'].tolist() == ['failed']
        failures = pd.read_csv(output / 'logs/failed_files.csv')
        assert failures['reason'].str.contains('failed to write output', regex=False).all()
    else:
        np.testing.assert_array_equal(fits.getdata(product), np.full((16, 16), 45.))
        assert not product.with_suffix('').exists()
        assert manifest['status'].tolist() == ['success']
        assert manifest['output_path'].tolist() == [str(product)]
    assert not (output / 'wcs').exists()
    assert not (output / 'cal').exists()


def test_calibration_only_does_not_require_raw_catalogues(monkeypatch, tmp_path):
    from otehiwai_pouakai import core_reduction

    Path(config.cal_files_dir()).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(core_reduction, 'calibrating_internal', lambda *a, **k: calls.append(a))
    Pouakai([], tmp_path / 'output', mode='cal', organise_files=False,
            make_masters=False, cal_input_files=['prepared.fits'])
    assert len(calls) == 1 and calls[0][0] == str(tmp_path / 'prepared.fits')
    assert not Path(config.cal_list_dir()).exists()
    assert not Path(config.master_dark_dir()).exists()
