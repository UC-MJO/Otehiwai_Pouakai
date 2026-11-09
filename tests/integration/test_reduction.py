"""Integration tests for master creation and reduction using small synthetic images."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from otehiwai_pouakai import config


pytestmark = pytest.mark.integration


def test_master_writers_create_only_their_output_directory(monkeypatch, tmp_path,
                                                         write_fits, make_catalogue):
    from otehiwai_pouakai.dark_masters import _bc_master_darks
    from otehiwai_pouakai.flat_masters import _bc_master_flats

    files = []
    for i in range(3):
        filename = write_fits(f'input-{i}.fits', np.full((8, 8), i + 1.),
                             header={'IMAGETYP': 'dark'})
        files.append(str(filename))
    frames = make_catalogue(files, master_name='example', master_dark=files[0])
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
        write_fits(filename, np.random.default_rng(i).uniform(10, 20, (8, 8)),
                   header={'IMAGETYP': 'flat'}, overwrite=True)
    frames['master_dark'] = dark.iloc[0]['filename']
    flat = _bc_master_flats(frames)
    assert Path(flat.iloc[0]['filename']).parent == flat_dir
    assert Path(flat.iloc[0]['filename']).is_file()
    assert np.isfinite(fits.getdata(flat.iloc[0]['filename'])).all()
    assert not Path(config.pouakai_home()).exists()


@pytest.mark.parametrize('compression_fails', [False, True])
def test_reduction_records_output_only_after_compression(monkeypatch, tmp_path, compression_fails,
                                                       write_fits, make_catalogue):
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
        write_fits(filename, np.full((16, 16), value), header={'IMAGETYP': filename.stem})
    lists = Path(config.cal_list_dir())
    lists.mkdir(parents=True)
    for name, filename in (('dark', dark), ('flat', flat)):
        make_catalogue([filename], shape=16).to_csv(
            lists / f'bc_master_{name}_list.csv', index=False)
    frames = make_catalogue([science], shape=16, master_name='science')
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
