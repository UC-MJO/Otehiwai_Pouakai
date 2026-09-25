"""
Lightweight, dependency-free persistent failure tracking shared across
pipeline stages.

Why this exists
----------------
Previously, every pipeline stage (organise / reduce / WCS-solve /
calibrate) re-attempted every candidate file on every run, with no memory
of files that failed in a previous run for reasons unlikely to change
(e.g. a frame with genuinely no matching master dark, or a field with too
few isolated calibration stars). For a nightly cron job re-run repeatedly
over an ever-growing archive, this means wasted CPU time re-attempting
known-bad files indefinitely.

This module stores one row per (stage, file) failure in a CSV under
`<save_location>/logs/failed_files.csv`, with the failure reason and a
timestamp. Before reprocessing a file at a given stage, callers check
`is_known_failure(...)`; on failure, callers call `record_failure(...)`.
On success, callers should call `clear_failure(...)` so a file that
later succeeds (e.g. after a new master dark becomes available) is not
permanently stuck as "known failed" -- the ledger represents the most
recent outcome, not a permanent blacklist.
"""

import os
import time
import socket
import random
import logging
import threading

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMNS = ['stage', 'filename', 'reason', 'timestamp']

# Per-process thread lock as a fast first line of defense for the common
# case (joblib "threading" backend, shared process) -- avoids unnecessary
# NFS round-trips when only threads within this process are contending.
# The link-based lock below is what actually protects across separate
# processes/machines.
_thread_locks = {}
_thread_locks_guard = threading.Lock()

_HOSTNAME = socket.gethostname()
_PID = os.getpid()


def _ledger_path(save_location):
    log_dir = os.path.join(save_location, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, 'failed_files.csv')


def _lock_path(path):
    return path + '.lock'


def _get_thread_lock(path):
    with _thread_locks_guard:
        if path not in _thread_locks:
            _thread_locks[path] = threading.Lock()
        return _thread_locks[path]


class _NFSSafeFileLock:
    """
    NFS-safe exclusive lock via link()-based claiming (see module
    docstring). Blocking, with exponential backoff + jitter while
    waiting, and a staleness timeout so a crashed process holding the
    lock cannot wedge the pipeline forever.
    """

    def __init__(self, lock_path, max_wait_seconds=120, stale_after_seconds=300):
        self.lock_path = lock_path
        self.max_wait_seconds = max_wait_seconds
        self.stale_after_seconds = stale_after_seconds
        self.claim_path = f'{lock_path}.claim.{_HOSTNAME}.{_PID}.{random.randint(0, 1_000_000)}'

    def _try_acquire_once(self):
        # Create the claim file fresh each attempt -- cheap, and avoids
        # any ambiguity about its state from a previous failed attempt.
        with open(self.claim_path, 'w') as f:
            f.write(f'{_HOSTNAME} {_PID} {time.time()}\n')

        try:
            os.link(self.claim_path, self.lock_path)
        except FileExistsError:
            # Someone else holds the lock (or a stale lock file is
            # sitting there) -- check staleness before giving up on this
            # attempt.
            self._break_if_stale()
            try:
                os.remove(self.claim_path)
            except OSError:
                pass
            return False
        except OSError as e:
            # Transient NFS hiccup (e.g. ESTALE) -- treat as "not
            # acquired this attempt" rather than raising, consistent with
            # documented NFS lock-file handling advice.
            logger.debug(f'Transient error linking claim file for {self.lock_path}: {e}')
            try:
                os.remove(self.claim_path)
            except OSError:
                pass
            return False

        # link() reported success, but per the documented NFS caveat,
        # don't fully trust that alone -- confirm via stat() that the
        # claim file's link count actually reflects two names now
        # pointing at the same inode (this process's claim file, and the
        # shared lock path).
        try:
            st = os.stat(self.claim_path)
            if st.st_nlink >= 2:
                return True
        except OSError:
            pass

        # link() didn't actually take (or we couldn't confirm it) --
        # clean up and report failure for this attempt.
        try:
            os.remove(self.claim_path)
        except OSError:
            pass
        try:
            os.remove(self.lock_path)
        except OSError:
            pass
        return False

    def _break_if_stale(self):
        try:
            mtime = os.path.getmtime(self.lock_path)
        except OSError:
            return
        if time.time() - mtime > self.stale_after_seconds:
            logger.warning(
                f'{self.lock_path}: breaking stale lock (older than '
                f'{self.stale_after_seconds}s) -- a previous holder likely '
                f'crashed without releasing it'
            )
            try:
                os.remove(self.lock_path)
            except OSError:
                pass

    def acquire(self):
        start = time.time()
        attempt = 0
        while True:
            if self._try_acquire_once():
                return
            attempt += 1
            if time.time() - start > self.max_wait_seconds:
                logger.warning(
                    f'{self.lock_path}: waited {self.max_wait_seconds}s for the lock; '
                    f'proceeding anyway to avoid deadlocking the pipeline. If this '
                    f'happens often, check for a stuck/crashed process holding the lock.'
                )
                return
            backoff = min(0.1 * (2 ** min(attempt, 6)), 5.0)
            backoff *= 1.0 + random.uniform(0, 0.5)
            time.sleep(backoff)

    def release(self):
        try:
            os.remove(self.lock_path)
        except OSError:
            pass
        try:
            os.remove(self.claim_path)
        except OSError:
            pass


class _LedgerLock:
    """
    Combines the per-process threading.Lock (fast path) with the
    NFS-safe link-based file lock (actual cross-process/cross-machine
    guarantee).
    """

    def __init__(self, path):
        self.path = path
        self.thread_lock = _get_thread_lock(path)
        self.file_lock = _NFSSafeFileLock(_lock_path(path))

    def __enter__(self):
        self.thread_lock.acquire()
        self.file_lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.file_lock.release()
        finally:
            self.thread_lock.release()
        return False


def _write_csv_locked(df, path):
    """
    Write `df` to `path` while holding the lock (caller is responsible
    for already holding it). Writes directly to `path` -- NOT via a
    temp-file-then-rename, since rename is exactly the operation that is
    unreliable across concurrent NFS readers (see module docstring).
    Because this is only ever called while `_LedgerLock` is held, no
    other cooperating reader/writer using this module will be mid-read
    at the same time. Readers that do NOT go through the lock (see
    `is_known_failure`'s docstring for why reads are unlocked) accept a
    small window where they might see a part-written file; `_load`
    treats that as "empty ledger" rather than crashing, and the caller's
    own behaviour (skip nothing, proceed as if not a known failure) is
    safe in that case -- worst case is a redundant reprocess, not data
    loss or a crash.

    Uses pandas' default QUOTE_MINIMAL (no explicit `quoting=` override)
    -- only fields that actually need escaping (contain a comma, quote
    character, or newline) get wrapped in quotes. Most rows (stage,
    filename, timestamp, and most reason strings) have none of those and
    print unquoted; a reason string that happens to contain a comma
    (e.g. "recursion error (..., see comment): ...") is quoted
    automatically so the comma isn't mistaken for a column separator.
    `pd.read_csv` (used everywhere this module reads the ledger back)
    handles both cases identically either way.
    """
    with open(path, 'w') as f:
        df.to_csv(f, index=False)


def _load(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=_COLUMNS)
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 0:
            # Genuinely empty/header-less file -- can legitimately happen
            # if a reader catches the ledger mid-write (reads are not
            # lock-serialized; see is_known_failure). Treat as an empty
            # ledger rather than raising: the caller will simply not
            # treat anything as a known failure this one time, which is
            # safe (a redundant reprocess, not a correctness problem).
            return pd.DataFrame(columns=_COLUMNS)
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[_COLUMNS]
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_COLUMNS)
    except Exception as e:
        logger.debug(f'Failed to read failure ledger {path} ({e}); treating as empty for this read')
        return pd.DataFrame(columns=_COLUMNS)


def record_failure(save_location, stage, filename, reason):
    """
    Record that `filename` failed at `stage` with `reason`. Overwrites any
    previous entry for the same (stage, filename) pair, so the ledger
    always reflects the most recent failure reason.
    """
    path = _ledger_path(save_location)

    with _LedgerLock(path):
        df = _load(path)
        mask = (df['stage'] == stage) & (df['filename'] == str(filename))
        df = df[~mask]

        new_row = pd.DataFrame([{
            'stage': stage, 'filename': str(filename),
            'reason': str(reason), 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        _write_csv_locked(df, path)


def clear_failure(save_location, stage, filename):
    """
    Remove any recorded failure for `filename` at `stage` -- call this on
    success so a file that previously failed but now succeeds (e.g. after
    a new master dark/flat becomes available, or after a parameter
    change) is no longer treated as a known failure on the next run.
    """
    path = _ledger_path(save_location)
    if not os.path.exists(path):
        return

    with _LedgerLock(path):
        df = _load(path)
        mask = (df['stage'] == stage) & (df['filename'] == str(filename))
        if mask.any():
            df = df[~mask]
            _write_csv_locked(df, path)


def is_known_failure(save_location, stage, filename):
    """
    Return the recorded failure reason (str) if `filename` is a known
    failure at `stage`, otherwise None.

    Reads deliberately do NOT take the lock. Reads happen once per
    candidate file, every run (potentially hundreds of times), while
    writes are comparatively rare (only on actual failures/clears) --
    serializing every read through the NFS-safe lock (which involves a
    link()+stat() round-trip) would add substantial latency for no real
    safety benefit here: in the rare case a read catches the ledger
    mid-write, `_load` treats it as an empty ledger, and the only
    consequence is that this one file is (incorrectly, but safely)
    treated as "not a known failure" and gets reprocessed -- a wasted
    retry, not data corruption or a crash.
    """
    path = _ledger_path(save_location)
    if not os.path.exists(path):
        return None

    df = _load(path)
    mask = (df['stage'] == stage) & (df['filename'] == str(filename))
    matches = df[mask]
    if len(matches) == 0:
        return None
    return matches.iloc[-1]['reason']


def load_known_failures(save_location, stage):
    """
    Return the full set of filenames known to have failed at `stage`, as
    a Python set, for efficient bulk filtering of a candidate file list
    (avoids one CSV read per file when checking a large batch).
    """
    path = _ledger_path(save_location)
    if not os.path.exists(path):
        return set()

    df = _load(path)
    return set(df.loc[df['stage'] == stage, 'filename'].astype(str))


def summarize(save_location):
    """
    Return a per-stage failure-count summary (pandas DataFrame), useful
    for printing at the end of a test/diagnostic run.
    """
    path = _ledger_path(save_location)
    if not os.path.exists(path):
        return pd.DataFrame(columns=['stage', 'count'])

    df = _load(path)
    if len(df) == 0:
        return pd.DataFrame(columns=['stage', 'count'])

    return df.groupby('stage').size().reset_index(name='count')


def load_all_failures(save_location):
    """
    Return the full failure ledger as a DataFrame (columns: stage,
    filename, reason, timestamp), or an empty DataFrame with those
    columns if nothing has been recorded yet. Public counterpart to the
    internal `_load`, for callers (e.g. diagnostic/test scripts) that
    want a full reason-level breakdown rather than just counts.
    """
    path = _ledger_path(save_location)
    return _load(path)