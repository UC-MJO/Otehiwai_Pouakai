"""Integration tests for stage orchestration, dependency checks and output handling."""

import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from otehiwai_pouakai import config, Pouakai


pytestmark = pytest.mark.integration


def test_noop_pipeline_does_not_require_inputs_or_create_outputs(tmp_path):
    output = tmp_path / 'output'
    Pouakai([], output, run=False, organise_files=False, make_masters=False)
    assert not output.exists()
    assert not Path(config.pouakai_home()).exists()


def test_invalid_mode_fails_before_setup(tmp_path):
    with pytest.raises(ValueError, match='mode must be'):
        Pouakai([], tmp_path / 'output', mode='invalid')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('input_source', ['files', 'glob', 'directory'])
def test_wcs_only_needs_solver_and_stage_inputs(monkeypatch, tmp_path, input_source, fake_astrometry):
    solver = fake_astrometry(tmp_path / 'astrometry')
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


def test_index_check_matches_solver_engine_lookup(monkeypatch, tmp_path, fake_astrometry):
    path_solver = fake_astrometry(tmp_path / 'on-path')
    explicit_solver = fake_astrometry(tmp_path / 'explicit')
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
def test_missing_indexes_fail_before_any_processing(monkeypatch, tmp_path, mode, fake_astrometry):
    solver = fake_astrometry(tmp_path / 'astrometry')
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


def test_index_check_runs_once_per_batch_and_again_on_next_call(monkeypatch, tmp_path, fake_astrometry):
    solver = fake_astrometry(tmp_path / 'astrometry')
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


def test_missing_stage_directory_is_not_created(monkeypatch, tmp_path, fake_astrometry):
    solver = fake_astrometry(tmp_path / 'astrometry')
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
def test_skipped_frames_do_not_require_unused_dependencies(monkeypatch, tmp_path, stage, status,
                                                         batch, fake_astrometry):
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
        solver = fake_astrometry(tmp_path / 'astrometry')
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
