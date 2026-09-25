"""
Otehiwai Pouakai: end-to-end FLI/B&C astronomical reduction, astrometric,
and photometric calibration pipeline.

This registers the project's warning-suppression filters (see
`suppress_warnings.py`) as the very first thing that happens on import,
before any submodule gets a chance to trigger the import-time warning
they're meant to silence (see that module's docstring for why import
order matters here) -- so callers no longer need to remember to
`import otehiwai_pouakai.suppress_warnings` first themselves; simply
`import otehiwai_pouakai` (or any submodule of it) is now sufficient.

Commonly used entry points are re-exported here for convenience:

    from otehiwai_pouakai import Pouakai, setup_logging

is equivalent to the more verbose

    from otehiwai_pouakai.pipeline import Pouakai, setup_logging

Everything else (calibration, reduction, photometry, etc.) is available
via its own submodule, e.g. `otehiwai_pouakai.calibration_saurus`,
`otehiwai_pouakai.psf_photometry`.
"""

from . import suppress_warnings  # noqa: F401 -- must be first, see above

from .pipeline import Pouakai, setup_logging  # noqa: F401

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('otehiwai-pouakai')
except PackageNotFoundError:
    # Package not installed (e.g. running directly from a source
    # checkout without `pip install -e .`) -- not fatal, just unknown.
    __version__ = 'unknown'

__all__ = ['Pouakai', 'setup_logging', '__version__']
