"""
Warning-suppression bootstrap: silences specific known-noisy,
known-harmless warnings (currently the "pkg_resources is deprecated as an
API" UserWarning triggered by sklearn.cluster) before any other project
or third-party module gets a chance to import the library that raises
them.

Why import order matters here
------------------------------
The "pkg_resources is deprecated as an API" warning is raised the FIRST
time anything imports `sklearn.cluster` (it comes from
`sklearn/utils/fixes.py` doing `from pkg_resources import parse_version`
at IMPORT time, not at call time). A `warnings.filterwarnings(...)` call
only suppresses warnings emitted by `warnings.warn(...)` calls that
happen AFTER the filter is registered, in actual process execution
order -- not source-file order, and there's no way to retroactively
silence a warning that already printed. Since `sklearn.cluster` is only
ever imported once per process (Python caches imports), whichever module
imports it first is what determines whether the warning is already
suppressed at that point. This module's job is simply to make sure the
filters below are registered before that first import happens anywhere
in the pipeline.

Usage
-----
Import this module FIRST, before any other project or third-party
imports, at the top of every entry-point script:

    import suppress_warnings  # noqa: F401  (must be first import)
    import otehiwai_pouakai
    ...

For the Pouakai package itself, this is imported as the very first line
of `otehiwai_pouakai.py`, `run_test_20250914.py`, and
`calibration_diagnostics.py`.
"""

import warnings

# Exact, deliberately narrow filters -- this module suppresses SPECIFIC

# Exact, deliberately narrow filters -- this module suppresses SPECIFIC
# known-noisy, known-harmless warnings, not warnings wholesale. A blanket
# "ignore everything" for a specific module is a separate, coarser
# choice made locally in a few pipeline files where it's needed (e.g.
# dark_masters.py, background_subtraction.py), and is unaffected by this
# module.

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

# sklearn itself has emitted various UserWarnings tied to pkg_resources/
# setuptools deprecation across versions; matching by module is a
# slightly wider net that also catches near-duplicate message text from
# minor version differences, without silencing UserWarnings from OTHER
# modules (e.g. genuine pipeline-code UserWarnings you'd still want to
# see).
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")