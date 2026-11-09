"""Integration tests for archive discovery and catalogue selection."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from otehiwai_pouakai import config


pytestmark = pytest.mark.integration


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


def test_discovery_uses_settings_changed_after_import(monkeypatch, tmp_path, write_fits):
    from otehiwai_pouakai.organise_files import organise_fli_files

    raw = tmp_path / 'raw'
    raw.mkdir()
    write_fits(raw / 'science.fits', np.ones((8, 8)))
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


def test_relative_science_inputs_match_absolute_catalogue_paths(monkeypatch, tmp_path, write_fits):
    from otehiwai_pouakai.matau import get_file_paths, update_df
    from otehiwai_pouakai.organise_files import organise_fli_files

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('POUAKAI_RAW_ARCHIVE_DIR', 'raw')
    monkeypatch.setenv('POUAKAI_CAL_LIST_DIR', 'lists')
    Path('raw').mkdir()
    relative = Path('raw/science.fits')
    absolute = tmp_path / relative
    write_fits(relative, np.ones((8, 8)))
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
