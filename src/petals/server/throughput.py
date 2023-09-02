from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import configargparse
import torch

import torch.mps
from hivemind.utils.logging import get_logger
from transformers import PretrainedConfig

from petals.constants import DTYPE_MAP
from petals.server.block_utils import resolve_block_dtype
from petals.utils.auto_config import AutoDistributedConfig
from petals.utils.convert_block import QuantType, convert_block
from petals.utils.disk_cache import DEFAULT_CACHE_DIR
from petals.utils.version import get_compatible_model_repo

logger = get_logger(__name__)

try:
    import speedtest
except ImportError:
    raise ImportError("Please `pip install speedtest-cli==2.1.3`")

if not hasattr(speedtest, "Speedtest"):
    raise ImportError(
        "You are using the wrong speedtest module. Please replace speedtest with speedtest-cli.\n"
        "To do that, run `pip uninstall -y speedtest`. Depending on your python environment, "
        "you may need to run uninstall speedtest two or more times, until it says 'not installed'.\n"
        "After that, please `pip install speedtest-cli==2.1.3` to install the correct version."
    )


def get_server_throughput(
    model_name: str,
    config: PretrainedConfig,
    device: torch.device,
    dtype: Union[str, torch.dtype],
    *,
    num_blocks: int,
    quant_type: QuantType,
    tensor_parallel_devices: Sequence[torch.device],
    reachable_via_relay: bool,
    relay_penalty: float = 0.2,
    force_eval: bool = False,
    cache_dir: Optional[str] = None,
) -> Dict[str, float]:
    dtype = resolve_block_dtype(config, dtype)

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    lock_path = Path(cache_dir, "throughput.lock")
    cache_path = Path(cache_dir, "throughput_v5.json")

    # We use the system-wide lock since only one process at a time can measure the host throughput
    os.makedirs(lock_path.parent, exist_ok=True)
    with open(lock_path, "wb") as lock_fd:
        logger.info("Loading throughput info")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        # The OS will release the lock when lock_fd is closed or the process is killed

        cache_key = f"model_{model_name}"
        cache_key += f"_device_{get_device_name(device).replace(' ', '_')}"
        cache_key += f"_dtype_{get_dtype_name(dtype, quant_type)}"
        if len(tensor_parallel_devices) > 1:
            for i, device_i in enumerate(tensor_parallel_devices):
                cache_key += f"_tp{i}_{get_device_name(device_i).replace(' ', '_')}"

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
            cache[cache_key] = measure_throughput_info(
                config, device, dtype, quant_type=quant_type, tensor_parallel_devices=tensor_parallel_devices
            )

            try:
                os.makedirs(cache_path.parent, exist_ok=True)
                with open(cache_path, "w") as cache_fd:
                    json.dump(cache, cache_fd)
            except Exception:
                logger.exception(f"Failed to save throughput info in {cache_path}")

    throughput_info = cache[cache_key]

    # Most requests start at some block hosted by a server, then use all next blocks hosted on this server.
    # Assuming the start block index is distributed uniformly, the average number of blocks used per request is
    # E[Uniform{1, 2, ..., num_blocks}] = (num_blocks + 1) / 2
    average_blocks_used = (num_blocks + 1) / 2
    throughput = throughput_info["forward_rps"] / average_blocks_used

    network_rps = throughput_info["network_rps"] * (relay_penalty if reachable_via_relay else 1)
    throughput = min(throughput, network_rps)

    throughput_info["throughput"] = throughput
    logger.info(f"Reporting throughput: {throughput:.1f} tokens/sec for {num_blocks} blocks")

    return throughput_info


def measure_throughput_info(
    config: PretrainedConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    quant_type: QuantType,
    tensor_parallel_devices: Sequence[torch.device],
    measure_network: bool = True,
) -> Dict[str, float]:
    logger.info(
        "Measuring network and compute throughput. This takes about a minute and will be cached for future runs"
    )
    return {
        "inference_rps": measure_compute_rps(
            config,
            device,
            dtype,
            quant_type=quant_type,
            tensor_parallel_devices=tensor_parallel_devices,
            n_tokens=1,
            n_steps=100,
            inference=True,
        ),
        "forward_rps": measure_compute_rps(
            config,
            device,
            dtype,
            quant_type=quant_type,
            tensor_parallel_devices=tensor_parallel_devices,
            n_tokens=1024,
            n_steps=10,
            inference=False,
        ),
        "network_rps": measure_network_rps(config, use_default=not measure_network),
    }


def measure_network_rps(
    config: PretrainedConfig, *, use_default=False, timeout: float = 60, default_speed: float = 100e6  # 100 Mbit/s
) -> Optional[float]:
    bits_per_request = config.hidden_size * 16  # Clients usually send 16-bit tensors for forward/backward
    if use_default:
        return default_speed / bits_per_request
    try:
        pipe_recv, pipe_send = mp.Pipe(duplex=False)
        process = mp.Process(target=_measure_bits_per_second, args=(pipe_send,))
        process.start()

        if not pipe_recv.poll(timeout):
            process.terminate()
            raise RuntimeError(f"speedtest did not finish in {timeout} seconds")
        network_info = pipe_recv.recv()
        if "exception" in network_info:
            raise RuntimeError(f"speedtest failed: {network_info['exception']}")

        network_rps = min(network_info["download"], network_info["upload"]) / bits_per_request
        if network_rps == 0:
            raise RuntimeError("speedtest has returned network_rps == 0")

        logger.info(
            f"Network throughput: {network_rps:.1f} tokens/sec "
            f"({network_info['download'] / 1e6:.2f} Mbit/s on download, "
            f"{network_info['upload'] / 1e6:.2f} Mbit/s on upload)"
        )
        return network_rps
    except RuntimeError as e:
        logger.info(f"Network throughput is not available: {e}. Using default of {default_speed / 1e6:.2f} Mbit/s")
        return default_speed / bits_per_request


def _measure_bits_per_second(pipe_send: mp.Pipe):
    try:
        s = speedtest.Speedtest()
        s.get_servers()
        s.get_best_server()
        s.download()
        s.upload()
        pipe_send.send(s.results.dict())
    except Exception as e:
        pipe_send.send({"exception": repr(e)})


def measure_compute_rps(
    config: PretrainedConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    quant_type: QuantType,
    tensor_parallel_devices: Sequence[torch.device],
    n_tokens: int,
    n_steps: int,
    inference: bool,
) -> float:
    device = torch.device(device)
    if not tensor_parallel_devices:
        tensor_parallel_devices = (device,)
    with torch.inference_mode():
        block = config.block_class(config).to(dtype)
        block = convert_block(block, 0, config, tensor_parallel_devices, device, quant_type=quant_type, freeze=True)

        cache = None
        elapsed = 0
        dummy_input = torch.randn(1, n_tokens, config.hidden_size, device=device, dtype=dtype)
        # with torch.profiler.profile(
        #         schedule=torch.profiler.schedule(wait=1, warmup=4, active=n_steps, repeat=1),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profbf16_70b_qkv'),
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=True
        # ) as prof:
        _, cache = block.forward(dummy_input, use_cache=True)  # Skip the 1st step to exclude the initialization time
        synchronize(device)
        # prof.step()

        start_time = time.perf_counter()
        for _ in range(n_steps):
            _, cache = block.forward(dummy_input, use_cache=True, layer_past=cache if inference else None)
            synchronize(device)
            # prof.step()
           
        elapsed = time.perf_counter() - start_time
        device_rps = n_steps * n_tokens / elapsed

    devices_repr = get_device_name(device)
    if len(tensor_parallel_devices) > 1:
        device_names = tuple(map(get_device_name, map(torch.device, tensor_parallel_devices)))
        devices_repr = ", ".join(f"{count}x {name}" for name, count in Counter(device_names).most_common())

    logger.info(
        f"{'Inference' if inference else 'Forward pass'} throughput: {device_rps:.1f} tokens/sec per block "
        f"({n_tokens} tokens/batch, {devices_repr}, {get_dtype_name(dtype, quant_type)})"
    )
    return device_rps


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def get_device_name(device: torch.device) -> str:
    return f"{torch.cuda.get_device_name(device)} GPU" if device.type == "cuda" else device.type.upper()


def get_dtype_name(dtype: torch.dtype, quant_type: QuantType) -> str:
    name = str(dtype).replace("torch.", "")
    if quant_type != QuantType.NONE:
        name += f", quantized to {quant_type.name.lower()}"
    return name


def parse_args():
    # fmt:off
    parser = configargparse.ArgParser(default_config_files=["config.yml"],
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--converted_model_name_or_path', type=str, default=None,
                       help="path or name of a pretrained model, converted with cli/convert_model.py")
    group.add_argument('model', nargs='?', type=str, help="same as --converted_model_name_or_path")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--token", type=str, default=None, help="Hugging Face hub auth token for .from_pretrained()")
    group.add_argument("--use_auth_token", action="store_true", dest="token",
                       help="Read token saved by `huggingface-cli login")

    parser.add_argument('--device', type=str, default=None, required=False,
                        help='all blocks will use this device in torch notation; default: cuda if available else cpu')
    parser.add_argument("--torch_dtype", type=str, choices=DTYPE_MAP.keys(), default="auto",
                        help="Use this dtype to store block weights and do computations. "
                             "By default, respect the dtypes in the pre-trained state dict.")
    parser.add_argument('--revision', type=str, default=None,
                        help="The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models"
                             "and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.")

    parser.add_argument('--quant_type', type=str, default=None, choices=[choice.name.lower() for choice in QuantType],
                        help="Quantize blocks to 8-bit (int8 from the LLM.int8() paper) or "
                             "4-bit (nf4 from the QLoRA paper) formats to save GPU memory. "
                             "Default: 'int8' if GPU is available, 'none' otherwise")
    parser.add_argument("--tensor_parallel_devices", nargs='+', default=None,
                        help=
                        "Split each block between the specified GPUs such that each device holds a portion of every "
                        "weight matrix. See https://huggingface.co/transformers/v4.9.0/parallelism.html#tensor-parallelism")

    # fmt:on
    args = parser.parse_args()
    args.converted_model_name_or_path = args.model
    return args


if __name__ == "__main__":
    args = parse_args()
    converted_model_name_or_path = get_compatible_model_repo(args.converted_model_name_or_path)
    config = AutoDistributedConfig.from_pretrained(
        converted_model_name_or_path,
        use_auth_token=args.token,
        revision=args.revision,
    )

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device(device.type, index=0)

    torch_dtype = resolve_block_dtype(config, DTYPE_MAP[args.torch_dtype])
    if device.type == "cpu" and torch_dtype == torch.float16:
        raise ValueError(
            f"Type float16 is not supported on CPU. Please use --torch_dtype float32 or --torch_dtype bfloat16"
        )
    if device.type == "mps" and torch_dtype == torch.bfloat16:
        logger.warning(f"Type bfloat16 is not supported on MPS, using float16 instead")
        torch_dtype = torch.float16

    quant_type = args.quant_type
    if quant_type is None:
        if device.type == "cuda":
            quant_type = QuantType.NF4 if config.model_type == "llama" else QuantType.INT8
        else:
            quant_type = QuantType.NONE

    if args.tensor_parallel_devices is None:
        args.tensor_parallel_devices = (device,)

    measure_throughput_info(
        config,
        device,
        torch_dtype,
        quant_type=quant_type,
        tensor_parallel_devices=args.tensor_parallel_devices,
        measure_network=False,
    )
