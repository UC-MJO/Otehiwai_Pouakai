"""
Cross-process counting semaphore, safe for use on NFS-mounted shared
storage.

This limits how many cooperating processes (potentially across multiple
machines, e.g. joblib's "loky" backend spawning separate OS processes,
not threads) may be doing something at once -- for example, limiting how
many simultaneous external API queries a parallel pipeline stage makes.
A plain `threading.Semaphore` only limits concurrency within a single
process, and `fcntl.flock` is documented as unreliable over NFS (it
depends on the NFS lock daemon and kernel version; see the flock(2) man
page's NFS-specific caveats), so this instead implements a counting
semaphore using a hard-link-based locking primitive that is safe on NFS.

How it works: for each of `max_concurrent` "slots", acquiring means
successfully creating a uniquely-named claim file and hard-linking it to
that slot's lock path, then confirming via `stat()` that the resulting
link count actually became 2 -- this doesn't rely on `link()`'s return
value alone, since an NFS server's success acknowledgement can be lost in
transit even though the link was actually created underneath. Releasing
removes both the lock path and the claim file. This is the same locking
scheme `failure_ledger.py` uses for a single shared resource, applied
here per-slot to allow more than one concurrent holder.
"""

import os
import time
import socket
import random
import logging

logger = logging.getLogger(__name__)

_HOSTNAME = socket.gethostname()
_PID = os.getpid()

class CrossProcessSemaphore:
    """
    A counting semaphore that works correctly across separate processes
    and machines sharing NFS-mounted storage.

    Usage
    -----
        sem = CrossProcessSemaphore(lock_dir, max_concurrent=4)
        with sem:
            ... do at most `max_concurrent` of these across ALL
            cooperating processes/machines at once ...

    Parameters
    ----------
    lock_dir : str
        Shared directory (e.g. on NFS) used for this semaphore's lock
        files. Created if it doesn't already exist.
    max_concurrent : int
        Maximum number of simultaneous holders across all cooperating
        processes/machines.
    poll_interval : float
        Base seconds to wait between attempts while all slots are taken,
        with random jitter added (see `acquire`).
    max_wait_seconds : float
        Maximum time to keep waiting for a free slot before giving up and
        proceeding anyway (see `acquire`).
    stale_after_seconds : float
        A slot's lock file older than this is assumed to belong to a
        crashed holder that never released it, and is broken (removed) so
        a new holder can claim it (see `_break_if_stale`).
    """

    def __init__(self, lock_dir, max_concurrent=4, poll_interval=0.5,
                 max_wait_seconds=300, stale_after_seconds=600):
        self.lock_dir = lock_dir
        self.max_concurrent = max_concurrent
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds
        self.stale_after_seconds = stale_after_seconds
        os.makedirs(lock_dir, exist_ok=True)
        self._held_slot_path = None
        self._held_claim_path = None

    def _slot_paths(self):
        """Return the lock file paths for all `max_concurrent` slots."""
        return [os.path.join(self.lock_dir, f'slot_{i}.lock')
                for i in range(self.max_concurrent)]

    def _break_if_stale(self, slot_path):
        """
        If `slot_path`'s lock file is older than `stale_after_seconds`,
        assume its holder crashed without releasing it and remove the
        lock file so another process can claim the slot.
        """
        try:
            mtime = os.path.getmtime(slot_path)
        except OSError:
            return
        if time.time() - mtime > self.stale_after_seconds:
            logger.warning(f'{slot_path}: breaking stale semaphore slot (older than '
                           f'{self.stale_after_seconds}s) -- a previous holder likely '
                           f'crashed without releasing it')
            try:
                os.remove(slot_path)
            except OSError:
                pass

    def _try_claim_slot(self, slot_path):
        """
        Attempt to claim a single slot via the hard-link scheme described
        in the module docstring.

        Returns
        -------
        claim_path : str or None
            The path of this process's own claim file if the slot was
            successfully claimed (needed later to release it), or None if
            the slot is already held by someone else (or the claim
            couldn't be confirmed).
        """
        claim_path = f'{slot_path}.claim.{_HOSTNAME}.{_PID}.{random.randint(0, 1_000_000)}'

        try:
            with open(claim_path, 'w') as f:
                f.write(f'{_HOSTNAME} {_PID} {time.time()}\n')
        except OSError as e:
            logger.debug(f'Could not create claim file {claim_path}: {e}')
            return None

        try:
            os.link(claim_path, slot_path)
        except FileExistsError:
            self._break_if_stale(slot_path)
            try:
                os.remove(claim_path)
            except OSError:
                pass
            return None
        except OSError as e:
            logger.debug(f'Transient error linking claim file for {slot_path}: {e}')
            try:
                os.remove(claim_path)
            except OSError:
                pass
            return None

        try:
            st = os.stat(claim_path)
            if st.st_nlink >= 2:
                return claim_path
        except OSError:
            pass

        # Could not confirm the link actually took -- clean up both
        # paths and report this attempt as unsuccessful.
        try:
            os.remove(claim_path)
        except OSError:
            pass
        try:
            os.remove(slot_path)
        except OSError:
            pass
        return None

    def acquire(self):
        """
        Block until a slot is claimed (or `max_wait_seconds` elapses).

        Slots are tried in random order each pass, so many processes
        contending at once don't all pile onto the same slot first.
        Between passes, waits `poll_interval` seconds plus up to 50%
        random jitter, so contending processes don't retry in lockstep.
        If no slot becomes free within `max_wait_seconds`, proceeds
        anyway (logging a warning) rather than deadlocking the caller
        indefinitely.
        """
        slots = self._slot_paths()
        random.shuffle(slots)

        start = time.time()
        while True:
            for slot_path in slots:
                claim_path = self._try_claim_slot(slot_path)
                if claim_path is not None:
                    self._held_slot_path = slot_path
                    self._held_claim_path = claim_path
                    return

            if time.time() - start > self.max_wait_seconds:
                logger.warning(f'CrossProcessSemaphore: waited {self.max_wait_seconds}s for a free slot '
                               f'(max_concurrent={self.max_concurrent}) in {self.lock_dir}; proceeding anyway '
                               f'to avoid deadlocking the pipeline -- consider raising max_concurrent or '
                               f'investigating why slots are held this long.')
                return

            time.sleep(self.poll_interval * (1.0 + random.uniform(0, 0.5)))

    def release(self):
        """Release the currently held slot, if any."""
        if self._held_slot_path is not None:
            try:
                os.remove(self._held_slot_path)
            except OSError:
                pass
        if self._held_claim_path is not None:
            try:
                os.remove(self._held_claim_path)
            except OSError:
                pass
        self._held_slot_path = None
        self._held_claim_path = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False