"""Shared isolation for offline pipeline tests."""

import os
import socket

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
