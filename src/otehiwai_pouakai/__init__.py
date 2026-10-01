"""Otehiwai Pouakai astronomical reduction and calibration pipeline."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('otehiwai-pouakai')
except PackageNotFoundError:
    __version__ = 'unknown'

__all__ = ['Pouakai', 'setup_logging', '__version__']


def __getattr__(name):
    if name in {'Pouakai', 'setup_logging'}:
        from . import pipeline
        value = getattr(pipeline, name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
