"""Standalone import checks, run in a fresh interpreter by configuration tests."""

import importlib
import os
from pathlib import Path
import pkgutil
import sys


__test__ = False


def install_audit_guard(*, forbid_side_effects=False):
    """Reject network access, optionally other side effects, and record attempts."""
    attempts = []

    def audit(event, args):
        forbidden = event in ('socket.connect', 'socket.getaddrinfo')
        if forbid_side_effects:
            forbidden |= event in ('os.mkdir', 'os.putenv', 'os.unsetenv',
                                  'subprocess.Popen', 'os.system')
            if event == 'open':
                _, mode, flags = args
                forbidden |= bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if forbidden:
            attempts.append(event)
            raise AssertionError(event)

    sys.addaudithook(audit)
    return attempts


def check_public_imports(action):
    before = dict(os.environ)
    attempts = install_audit_guard(forbid_side_effects=True)
    try:
        if action == 'version':
            import otehiwai_pouakai

            print(otehiwai_pouakai.__version__)
        elif action == 'exports':
            from otehiwai_pouakai import Pouakai, setup_logging

            assert callable(Pouakai) and callable(setup_logging)
        elif action == 'help':
            from otehiwai_pouakai.pipeline import main

            main(['--help'])
        else:
            raise ValueError(f'Unknown import check: {action}')
    except SystemExit as exc:
        assert exc.code == 0
    assert not attempts, attempts
    assert dict(os.environ) == before
    assert not {'numpy', 'pandas', 'calibrimbore', 'pysynphot', 'astroquery'} & sys.modules.keys()


def check_processing_imports():
    attempts = install_audit_guard()
    before = (os.environ.get('PATH'), os.environ.get('PYSYN_CDBS'))
    import otehiwai_pouakai

    for item in pkgutil.iter_modules(otehiwai_pouakai.__path__):
        importlib.import_module('otehiwai_pouakai.' + item.name)
    assert not attempts, attempts
    assert before == (os.environ.get('PATH'), os.environ.get('PYSYN_CDBS'))
    assert not {'calibrimbore', 'pysynphot', 'astroquery.gaia', 'astroquery.vizier', 'dl'} & sys.modules.keys()
    assert not Path(os.environ['POUAKAI_HOME']).exists()
    assert not Path(os.environ['POUAKAI_RAW_ARCHIVE_DIR']).exists()


if __name__ == '__main__':
    if sys.argv[1] == 'processing':
        check_processing_imports()
    else:
        check_public_imports(sys.argv[1])
