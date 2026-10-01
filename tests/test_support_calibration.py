"""Standalone checks of calibration-state validation, imports and prediction."""

import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from test_support_imports import install_audit_guard


__test__ = False


def check_saved_state():
    attempts = install_audit_guard()
    from otehiwai_pouakai import calibration_saurus, config

    assert 'astroquery' not in sys.modules
    assert 'calibrimbore' not in sys.modules
    import numpy as np
    import pandas as pd

    states = Path(config.cal_files_dir())
    states.mkdir(parents=True)
    # A minimal saved model selecting the reference r magnitude exactly. This
    # exercises the real loader and prediction without survey or CDBS data.
    np.save(states / 'sloan_r_ps1_ckmodel.npy', {
        'band': None, 'coeff': [0., 1., 0., 0., 0.], 'system': 'ps1',
        'R_coeff': [0., 0.], 'gi_lims': None, 'gr_lims': None,
        'cubic_corr': False, 'mag_system': 'ab', 'sys_filters': 'r',
        'cubic_coeff': None, 'color_correction': False,
    })
    obj = calibration_saurus.cal_photom(run=False)
    obj.band = 'sloan_r'
    obj.cal_model = 'ckmodel'
    obj.calibration_df = pd.DataFrame({'x_fit': [1.], 'y_fit': [1.]})
    obj.wcs = SimpleNamespace(all_pix2world=lambda *a: (np.array([100.]), np.array([0.])))
    obj._load_sauron()
    mags = pd.DataFrame({'g': [16., 17.], 'r': [15.5, 16.5], 'i': [15.3, 16.3],
                         'z': [15.2, 16.2], 'y': [15.1, 16.1]})
    np.testing.assert_allclose(obj.sauron.estimate_mag(mags=mags, extinction=False), mags['r'])
    from astroquery import log
    from astropy.logger import AstropyLogger

    assert isinstance(log, AstropyLogger)
    assert log.level == logging.WARNING
    assert not attempts, attempts
    assert not list(Path(os.environ['PYSYN_CDBS']).iterdir())


def check_state_validation():
    attempts = install_audit_guard()
    from otehiwai_pouakai import config
    from otehiwai_pouakai.calibration_saurus import cal_photom
    import numpy as np
    import pandas as pd

    obj = cal_photom(run=False)
    obj.band = 'bessell_V'
    obj.cal_model = 'calspec'
    obj.calibration_df = pd.DataFrame({'x_fit': [1.], 'y_fit': [1.]})
    obj.wcs = SimpleNamespace(all_pix2world=lambda *args: (np.array([1.]), np.array([-40.])))
    states = Path(config.cal_files_dir())
    states.mkdir(parents=True)
    state_file = states / 'bessell_V_skymapper_calspec.npy'
    try:
        obj._load_sauron()
    except config.ConfigurationError as exc:
        assert state_file.name in str(exc)
    else:
        raise AssertionError('Missing calibration state was accepted')
    assert not {'calibrimbore', 'astroquery'} & sys.modules.keys()

    state_file.write_bytes(b'fixture')
    client = SimpleNamespace(sauron=lambda *, load_state: ('loaded', load_state))
    with patch.dict(sys.modules, calibrimbore=client):
        obj._load_sauron()
        assert obj.sauron == ('loaded', str(state_file))
    assert 'PYSYN_CDBS' not in os.environ
    assert not attempts, attempts


if __name__ == '__main__':
    if sys.argv[1] == 'saved-state':
        check_saved_state()
    elif sys.argv[1] == 'state-validation':
        check_state_validation()
    else:
        raise ValueError(f'Unknown calibration check: {sys.argv[1]}')
