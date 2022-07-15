import fcntl
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Union

import torch
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from src import project_name
from src.bloom.block import BloomBlock
from src.bloom.model import BloomConfig
from src.bloom.ops import build_alibi_tensor

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


DEFAULT_CACHE_PATH = Path(Path.home(), ".cache", project_name, "throughput.json")
DEFAULT_LOCK_PATH = Path(tempfile.gettempdir(), project_name, "throughput.lock")

SPEED_TEST_PATH = Path(Path(__file__).absolute().parents[2], "cli", "speed_test.py")


@dataclass
class ThroughputInfo:
    network_rps: float
    device_rps: Dict[str, float]


def get_host_throughput(
    device: Union[str, torch.device],
    force_eval: bool = False,
    cache_path: str = DEFAULT_CACHE_PATH,
    lock_path: str = DEFAULT_LOCK_PATH,
) -> float:
    # We only keep the device type, assuming that the throughput is similar among all host's GPUs
    device = torch.device(device).type

    # We use the system-wide lock since only one process at a time can measure the host throughput
    os.makedirs(lock_path.parent, exist_ok=True)
    with open(lock_path, "wb") as lock_fd:
        logger.info("Loading throughput info")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        # The OS will release the lock when lock_fd is closed or the process is killed

        info = None
        try:
            if not force_eval and os.path.exists(cache_path):
                with open(cache_path) as cache_fd:
                    info = ThroughputInfo(**json.load(cache_fd))
                if device not in info.device_rps:
                    force_eval = True
        except Exception:
            logger.exception(f"Failed to read throughput info from {cache_path}")
            force_eval = True

        if force_eval or info is None:
            info = measure_throughput_info()
            try:
                os.makedirs(cache_path.parent, exist_ok=True)
                with open(cache_path, "w") as cache_fd:
                    json.dump(asdict(info), cache_fd)
            except Exception:
                logger.exception(f"Failed to save throughput info in {cache_path}")

    throughput = min(info.network_rps, info.device_rps[device])
    return throughput


def measure_throughput_info() -> ThroughputInfo:
    logger.info(
        "Measuring network, CPU, and GPU throughput. " "This takes about a minute and will be cached for future runs"
    )

    # We measure throughput in "(inference) requests per second" (RPS) using a fixed model
    config = BloomConfig.from_pretrained("bigscience/test-bloomd-6b3")

    network_rps = measure_network_rps(config)

    device_rps = {"cpu": measure_device_rps("cpu", config)}
    if torch.cuda.is_available():
        device_rps["cuda"] = measure_device_rps("cuda", config)

    return ThroughputInfo(network_rps=network_rps, device_rps=device_rps)


def measure_network_rps(config: BloomConfig) -> float:
    proc = subprocess.run([SPEED_TEST_PATH, "--json"], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to measure network throughput (stdout: {proc.stdout}, stderr: {proc.stderr})")
    network_info = json.loads(proc.stdout)

    bits_per_request = config.hidden_size * 32
    network_rps = min(network_info["download"], network_info["upload"]) / bits_per_request

    logger.info(
        f"Network throughput: "
        f"{network_info['download'] / 1e6:.2f} Mbit/s on download, "
        f"{network_info['upload'] / 1e6:.2f} Mbit/s on upload, "
        f"{network_rps:.2f} RPS"
    )
    return network_rps


def measure_device_rps(device: str, config: BloomConfig, layer_index: int = 0, n_steps: int = 500) -> float:
    with torch.inference_mode():
        block = BloomBlock(config, layer_index).to(device)
        cache = None
        elapsed = 0
        for i in range(n_steps):
            dummy_input = torch.randn(1, 1, config.hidden_size, device=device)
            alibi = build_alibi_tensor(i + 1, config.num_attention_heads, dtype=torch.float32, device=device)

            start_time = time.perf_counter()
            _, cache = block.forward(dummy_input, alibi=alibi, use_cache=True, layer_past=cache)
            elapsed += time.perf_counter() - start_time
        device_rps = n_steps / elapsed

    device_name = f"{torch.cuda.get_device_name(0)} GPU" if device == "cuda" else "CPU"
    logger.info(f"Compute throughput ({device_name}): {device_rps:.2f} RPS")

    return device_rps
