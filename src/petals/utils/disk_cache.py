import fcntl
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import huggingface_hub
from hivemind.utils.logging import get_logger

logger = get_logger(__file__)

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


def allow_cache_writes(
    cache_dir: Optional[str], *, reserve: Optional[int] = None, max_disk_space: Optional[int] = None
):
    """Allows saving new blocks and removing the old ones (exclusive lock)"""
    return _blocks_lock(cache_dir, fcntl.LOCK_EX)


def free_disk_space_for(
    model_name: str,
    size: int,
    *,
    cache_dir: Optional[str],
    max_disk_space: Optional[int],
    os_quota: int = 1024**3,  # Minimal space we should leave to keep OS function normally
):
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_info = huggingface_hub.scan_cache_dir(cache_dir)
    model_repos = [repo for repo in cache_info.repos if repo.repo_type == "model" and repo.repo_id == model_name]

    occupied_space = sum(repo.size_on_disk for repo in model_repos)
    available_space = shutil.disk_usage(cache_dir).free - os_quota
    if max_disk_space is not None:
        available_space = min(available_space, max_disk_space - occupied_space)
    if size <= available_space:
        return

    revisions = [revision for repo in model_repos for revision in repo.revisions]
    revisions.sort(key=lambda rev: max([item.blob_last_accessed for item in rev.files], default=rev.last_modified))

    # Remove as few least recently used blocks as possible
    pending_removal = []
    freed_space = 0
    extra_space_needed = size - available_space
    for rev in revisions:
        pending_removal.append(rev.commit_hash)
        freed_space += rev.size_on_disk
        if freed_space >= extra_space_needed:
            break

    if pending_removal:
        gib = 1024**3
        logger.info(f"Removing {len(pending_removal)} blocks to free {freed_space / gib:.1f} GiB of disk space")
        delete_strategy = cache_info.delete_revisions(*pending_removal)
        delete_strategy.execute()

    if freed_space < extra_space_needed:
        raise RuntimeError(
            f"Insufficient disk space to load a block. Please free {extra_space_needed - freed_space:.1f} GiB "
            f"on the volume for {cache_dir} or increase --max_disk_space if you set it manually"
        )
