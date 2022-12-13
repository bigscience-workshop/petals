import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DEFAULT_CACHE_DIR = os.getenv("PETALS_CACHE", Path(Path.home(), ".cache", "petals"))

CLEANUP_LOCK_FILENAME = "cleanup.lock"


@contextmanager
def cleanup_lock(cache_dir: Optional[str], mode: int):
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    lock_path = Path(cache_dir, CLEANUP_LOCK_FILENAME)

    with open(lock_path, "wb") as lock_fd:
        fcntl.flock(lock_fd.fileno(), mode)
        yield


def block_cache_removals(cache_dir: Optional[str]):
    # Shared lock: multiple processes can read the cache and add files simultaneously
    return cleanup_lock(cache_dir, fcntl.LOCK_SH)


def allow_cache_removals(cache_dir: Optional[str]):
    # Exclusive lock: no one reads while we clean the cache
    return cleanup_lock(cache_dir, fcntl.LOCK_EX)


def free_space_for(size: int, cache_dir: Optional[str], max_disk_space: int):
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Estimate how much we need to remove
    # Problem: Maybe 2 concurrent processes simulatenously decide that that
    # they don't need to remove anything, then compete for the same space

    # Solution: under a lock, do the check, then write block_size zero bytes?

    with allow_cache_removals(cache_dir):
        # Note: When we enter the lock, the free space may have increased, but we don't count that
        # since it's possible that a concurrent process has freed this space for itself.

        # Remove
        ...
