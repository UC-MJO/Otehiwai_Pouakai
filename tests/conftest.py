"""Shared isolation and input factories for pipeline tests."""

import os
from pathlib import Path
import socket
import sys

import pytest


@pytest.fixture(autouse=True)
def local_environment(monkeypatch, tmp_path):
    """Give each test temporary settings and a working directory; block network access."""
    for name in tuple(os.environ):
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
    monkeypatch.chdir(tmp_path)

    def no_network(*args, **kwargs):
        raise AssertionError('Tests must not access the network')

    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(socket, 'getaddrinfo', no_network)


@pytest.fixture
def write_fits(tmp_path):
    """Write explicit pixel data with standard test headers and optional overrides."""
    from astropy.io import fits

    def write(filename, data, header=None, *, overwrite=False):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = fits.Header({
            'IMAGETYP': 'science', 'EXPTIME': 30., 'JD': 2460000.,
            'DATE-OBS': '2023-02-24', 'READOUTM': '2MHz', 'FILTER': 'V',
        })
        if header is not None:
            metadata.update(header)
        fits.writeto(path, data, metadata, overwrite=overwrite)
        return path

    return write


@pytest.fixture
def make_catalogue():
    """Build image catalogue rows with common metadata and per-column overrides."""
    import pandas as pd

    def make(files, **columns):
        return pd.DataFrame({
            'filename': [str(Path(filename).absolute()) for filename in files],
            'jd': 2460000., 'date': '2023-02-24', 'exptime': 30.,
            'readout': '2MHz', 'shape': 8, 'band': 'V',
            **columns,
        })

    return make


@pytest.fixture
def fake_astrometry():
    """Create fake Astrometry.net executables in the requested directory."""
    def create(directory):
        directory.mkdir()
        for name, script in {
            'solve-field': 'test_support_solve_field.py',
            'astrometry-engine': 'test_support_astrometry_engine.py',
        }.items():
            executable = directory / name
            executable.write_text(f'#!{sys.executable}\n' + Path(__file__).with_name(script).read_text())
            executable.chmod(0o755)
        helper = directory / 'pouakai-test-helper'
        helper.write_text('#!/bin/sh\nexit 0\n')
        helper.chmod(0o755)
        return directory / 'solve-field'

    return create
