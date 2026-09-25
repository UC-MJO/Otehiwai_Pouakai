"""
Logging helpers for propagating the main process's logging configuration
into joblib "loky" backend worker processes (separate OS processes, not
threads), used by the calibration stage.

Background: a logging handler configured in the main process (e.g. via
some `setup_logging()` call) only exists in that process. A worker
process spawned by joblib's "loky" backend starts with a bare root
logger with no handlers, so anything it logs falls back to Python's
`logging.lastResort` handler -- a bare stderr handler fixed at WARNING,
regardless of what console/file logging level the main process actually
requested. `get_current_logging_config` reads back the main process's
current logging setup, and `configure_process_logging` applies it inside
a worker process, so calibration-stage logging behaves consistently with
the rest of the pipeline. Stages that use joblib's "threading" backend
instead (a shared process) don't need this, since they inherit the main
process's already-configured handlers directly.
"""

import logging


def get_current_logging_config():
    """
    Read back the current process's already-configured root logger to
    recover (log_file, level, console_level), so those same settings can
    be passed through to a loky worker process via
    `configure_process_logging`.

    Returns
    -------
    (log_file, level, console_level) : (str or None, int, int)
        log_file is None if no FileHandler is found on the root logger
        (e.g. logging was never explicitly configured) -- in that case,
        `configure_process_logging` will simply do nothing when given it.
    """
    root_logger = logging.getLogger()
    log_file = None
    level = logging.INFO
    console_level = logging.ERROR

    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler):
            log_file = h.baseFilename
            level = h.level
        elif isinstance(h, logging.Handler) and not isinstance(h, logging.FileHandler):
            console_level = h.level

    return log_file, level, console_level


def configure_process_logging(log_file, level=logging.INFO, console_level=logging.ERROR):
    """
    Configure logging in the CURRENT process to match another process's
    setup -- writing to the same log file and respecting the same
    console level. Intended to be called at the top of a worker task
    running in a separate process (e.g. under joblib's "loky" backend),
    using the values returned by `get_current_logging_config` in the
    main process.

    Idempotent per process: does nothing if the root logger already has
    handlers, so calling this at the top of every worker task is cheap
    after the first call in a given (reused) worker process. Also a
    no-op if `log_file` is None -- callers should only invoke this when
    they actually have a log file to pass through.

    Parameters
    ----------
    log_file : str or None
        Path to the shared log file to write to.
    level : int
        Logging level for the file handler.
    console_level : int
        Logging level for the console (stderr) handler.
    """
    if log_file is None:
        return

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # already configured in this process

    root_logger.setLevel(min(level, console_level))

    fmt = logging.Formatter('%(asctime)s %(levelname)-8s %(name)s: %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # Plain StreamHandler (not a tqdm-aware handler) -- worker processes
    # don't render the main process's progress bar anyway, and at
    # console_level=CRITICAL/ERROR this essentially never fires.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)