import fcntl
import json
import os
import subprocess
import time
from hashlib import sha256
from pathlib import Path
from typing import Optional, Union

import torch
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from transformers.models.bloom.modeling_bloom import build_alibi_tensor

from petals.bloom.block import BloomBlock
from petals.bloom.modeling_utils import BloomConfig
from petals.server.block_utils import resolve_block_dtype
from petals.utils.convert_8bit import replace_8bit_linear
from petals.utils.disk_cache import DEFAULT_CACHE_DIR

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


def get_host_throughput(
    config: BloomConfig,
    device: torch.device,
    dtype: Union[str, torch.dtype],
    *,
    load_in_8bit: bool,
    force_eval: bool = False,
    cache_dir: Optional[str] = None,
) -> float:
    dtype = resolve_block_dtype(config, dtype)

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    lock_path = Path(cache_dir, "throughput.lock")
    cache_path = Path(cache_dir, "throughput_v2.json")

    # We use the system-wide lock since only one process at a time can measure the host throughput
    os.makedirs(lock_path.parent, exist_ok=True)
    with open(lock_path, "wb") as lock_fd:
        logger.info("Loading throughput info")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        # The OS will release the lock when lock_fd is closed or the process is killed

        cache_key = f"config_{sha256(str(config).encode()).hexdigest()[-16:]}"
        cache_key += f"_device_{get_device_name(device).replace(' ', '_')}"
        cache_key += f"_dtype_{get_dtype_name(dtype, load_in_8bit)}"

        cache = {}
        try:
            if not force_eval and os.path.exists(cache_path):
                with open(cache_path) as cache_fd:
                    cache = json.load(cache_fd)
                assert isinstance(cache, dict)
        except Exception:
            logger.exception(f"Failed to read throughput info from {cache_path}")
            cache = {}

        if cache_key not in cache:
            cache[cache_key] = measure_throughput_info(config, device, dtype, load_in_8bit=load_in_8bit)

            try:
                os.makedirs(cache_path.parent, exist_ok=True)
                with open(cache_path, "w") as cache_fd:
                    json.dump(cache, cache_fd)
            except Exception:
                logger.exception(f"Failed to save throughput info in {cache_path}")

    return cache[cache_key]


def measure_throughput_info(
    config: BloomConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    load_in_8bit: bool,
) -> float:
    """Measure network and compute throughput in forward pass tokens per second"""

    logger.info(
        "Measuring network and compute throughput. This takes about a minute and will be cached for future runs"
    )
    return min(
        measure_network_rps(config),
        measure_compute_rps(config, device, dtype, load_in_8bit=load_in_8bit),
    )


def measure_network_rps(config: BloomConfig) -> float:
    proc = subprocess.run("python3 -m petals.cli.speed_test --json", shell=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to measure network throughput (stdout: {proc.stdout}, stderr: {proc.stderr})")
    network_info = json.loads(proc.stdout)

    bits_per_request = config.hidden_size * 16  # Clients usually send 16-bit tensors for forward/backward
    network_rps = min(network_info["download"], network_info["upload"]) / bits_per_request

    logger.info(
        f"Network throughput: "
        f"{network_info['download'] / 1e6:.2f} Mbit/s on download, "
        f"{network_info['upload'] / 1e6:.2f} Mbit/s on upload, "
        f"{network_rps:.1f} RPS"
    )
    return network_rps


def measure_compute_rps(
    config: BloomConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    load_in_8bit: bool,
    n_tokens: int = 16,
    n_steps: int = 500,
) -> float:
    with torch.inference_mode():
        block = BloomBlock(config).to(dtype)
        if load_in_8bit:
            block = replace_8bit_linear(block)
        block = block.to(device)

        cache = None
        elapsed = 0
        for step in range(n_steps + 1):
            dummy_input = torch.randn(n_tokens, 1, config.hidden_size, device=device, dtype=dtype)
            alibi = build_alibi_tensor(step + 1, config.num_attention_heads, device=device, dtype=dtype)

            start_time = time.perf_counter()
            _, cache = block.forward(dummy_input, alibi=alibi, use_cache=True, layer_past=cache)
            if step >= 1:  # Skip the 1st step to exclude the initialization time
                elapsed += time.perf_counter() - start_time
        device_rps = n_steps * n_tokens / elapsed

    logger.info(
        f"Forward pass throughput ({get_device_name(device)}, {get_dtype_name(dtype, load_in_8bit)}): "
        f"{device_rps:.1f} RPS"
    )
    return device_rps


def get_device_name(device: torch.device) -> str:
    return f"{torch.cuda.get_device_name(device)} GPU" if device.type == "cuda" else "CPU"


def get_dtype_name(dtype: torch.dtype, load_in_8bit: bool) -> str:
    return "8-bit" if load_in_8bit else str(dtype)
