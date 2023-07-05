import fcntl
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import huggingface_hub
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = os.getenv("PETALS_CACHE", Path(Path.home(), ".cache", "petals"))

BLOCKS_LOCK_FILE = "blocks.lock"


@contextmanager
def _blocks_lock(cache_dir: Optional[str], mode: int):
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    lock_path = Path(cache_dir, BLOCKS_LOCK_FILE)

    os.makedirs(lock_path.parent, exist_ok=True)
    with open(lock_path, "wb") as lock_fd:
        fcntl.flock(lock_fd.fileno(), mode)
        # The OS will release the lock when lock_fd is closed or the process is killed
        yield


def allow_cache_reads(cache_dir: Optional[str]):
    """Allows simultaneous reads, guarantees that blocks won't be removed along the way (shared lock)"""
    return _blocks_lock(cache_dir, fcntl.LOCK_SH)


def allow_cache_writes(cache_dir: Optional[str]):
    """Allows saving new blocks and removing the old ones (exclusive lock)"""
    return _blocks_lock(cache_dir, fcntl.LOCK_EX)


def free_disk_space_for(
    size: int,
    *,
    cache_dir: Optional[str],
    max_disk_space: Optional[int],
    os_quota: int = 1024**3,  # Minimal space we should leave to keep OS function normally
):
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_info = huggingface_hub.scan_cache_dir(cache_dir)

    available_space = shutil.disk_usage(cache_dir).free - os_quota
    if max_disk_space is not None:
        available_space = min(available_space, max_disk_space - cache_info.size_on_disk)

    gib = 1024**3
    logger.debug(f"Disk space: required {size / gib:.1f} GiB, available {available_space / gib:.1f} GiB")
    if size <= available_space:
        return

    cached_files = [file for repo in cache_info.repos for revision in repo.revisions for file in revision.files]

    # Remove as few least recently used files as possible
    removed_files = []
    freed_space = 0
    extra_space_needed = size - available_space
    for file in sorted(cached_files, key=lambda file: file.blob_last_accessed):
        os.remove(file.file_path)  # Remove symlink
        os.remove(file.blob_path)  # Remove contents

        removed_files.append(file)
        freed_space += file.size_on_disk
        if freed_space >= extra_space_needed:
            break
    if removed_files:
        logger.info(f"Removed {len(removed_files)} files to free {freed_space / gib:.1f} GiB of disk space")
        logger.debug(f"Removed paths: {[str(file.file_path) for file in removed_files]}")

    if freed_space < extra_space_needed:
        raise RuntimeError(
            f"Insufficient disk space to load a block. Please free {(extra_space_needed - freed_space) / gib:.1f} GiB "
            f"on the volume for {cache_dir} or increase --max_disk_space if you set it manually"
        )
