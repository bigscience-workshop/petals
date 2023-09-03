from __future__ import annotations

import gc
import math
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Union

import hivemind
import psutil
import torch
import torch.mps
from hivemind import DHT, MAX_DHT_TIME_DISCREPANCY_SECONDS, BatchTensorDescriptor, get_dht_time
from hivemind.moe.server.layers import add_custom_models_from_file
from hivemind.moe.server.runtime import Runtime
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger
from transformers import PretrainedConfig

import petals
from petals.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS
from petals.data_structures import CHAIN_DELIMITER, UID_DELIMITER, ModelInfo, ServerInfo, ServerState
from petals.server import block_selection
from petals.server.backend import TransformerBackend, merge_inference_pools_inplace
from petals.server.block_utils import get_block_size, resolve_block_dtype
from petals.server.from_pretrained import load_pretrained_block
from petals.server.handler import TransformerConnectionHandler
from petals.server.memory_cache import MemoryCache
from petals.server.reachability import ReachabilityProtocol, check_direct_reachability, validate_reachability
from petals.server.throughput import get_dtype_name, get_server_throughput
from petals.utils.auto_config import AutoDistributedConfig
from petals.utils.convert_block import QuantType, check_device_balance, convert_block
from petals.utils.dht import declare_active_modules, get_remote_module_infos
from petals.utils.misc import get_size_in_bytes
from petals.utils.ping import PingAggregator
from petals.utils.random import sample_up_to
from petals.utils.version import get_compatible_model_repo

logger = get_logger(__name__)


class Server:
    """
    Runs ModuleContainer, periodically checks that the network is balanced,
    restarts the ModuleContainer with other layers if the imbalance is significant
    """

    def __init__(
        self,
        *,
        initial_peers: List[str],
        dht_prefix: Optional[str],
        converted_model_name_or_path: str,
        public_name: Optional[str] = None,
        throughput: Union[float, str],
        num_blocks: Optional[int] = None,
        block_indices: Optional[str] = None,
        num_handlers: int = 8,
        inference_max_length: Optional[int] = None,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        max_chunk_size_bytes: int = 256 * 1024 * 1024,
        max_alloc_timeout: float = 600,
        attn_cache_tokens: Optional[int] = None,
        torch_dtype: str = "auto",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_disk_space: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        compression=CompressionType.NONE,
        stats_report_interval: Optional[int] = None,
        custom_module_path=None,
        update_period: float = 60,
        expiration: Optional[float] = None,
        request_timeout: float = 3 * 60,
        session_timeout: float = 30 * 60,
        step_timeout: float = 5 * 60,
        prefetch_batches: int = 1,
        sender_threads: int = 1,
        balance_quality: float = 0.75,
        mean_balance_check_period: float = 120,
        mean_block_selection_delay: float = 5,
        token: Optional[Union[str, bool]] = None,
        quant_type: Optional[QuantType] = None,
        tensor_parallel_devices: Optional[Sequence[torch.device]] = None,
        skip_reachability_check: bool = False,
        reachable_via_relay: Optional[bool] = None,
        use_relay: bool = True,
        use_auto_relay: bool = True,
        adapters: Sequence[str] = (),
        **kwargs,
    ):
        """Create a server with one or more bloom blocks. See run_server.py for documentation."""

        converted_model_name_or_path = get_compatible_model_repo(converted_model_name_or_path)
        self.converted_model_name_or_path = converted_model_name_or_path

        self.num_handlers = num_handlers
        self.compression = compression
        self.stats_report_interval, self.update_period = stats_report_interval, update_period
        self.prefetch_batches, self.sender_threads = prefetch_batches, sender_threads
        self.revision, self.token = revision, token

        if custom_module_path is not None:
            add_custom_models_from_file(custom_module_path)

        self.block_config = AutoDistributedConfig.from_pretrained(
            converted_model_name_or_path,
            use_auth_token=token,
            revision=revision,
        )

        if dht_prefix is None:
            dht_prefix = self.block_config.dht_prefix
        assert UID_DELIMITER not in dht_prefix and CHAIN_DELIMITER not in dht_prefix, (
            f"DHT prefix should not contain '{UID_DELIMITER}' or '{CHAIN_DELIMITER}'. "
            f"Please specify another --dht_prefix manually when starting a server"
        )
        self.dht_prefix = dht_prefix

        if expiration is None:
            expiration = max(2 * update_period, MAX_DHT_TIME_DISCREPANCY_SECONDS)
        self.expiration = expiration

        self.request_timeout = request_timeout
        self.session_timeout, self.step_timeout = session_timeout, step_timeout

        self.module_uids = [
            f"{self.dht_prefix}{UID_DELIMITER}{block_index}"
            for block_index in range(self.block_config.num_hidden_layers)
        ]

        if reachable_via_relay is None:
            is_reachable = check_direct_reachability(initial_peers=initial_peers, use_relay=False, **kwargs)
            reachable_via_relay = is_reachable is False  # if can't check reachability (returns None), run a full peer
            logger.info(f"This server is accessible {'via relays' if reachable_via_relay else 'directly'}")
        self.dht = DHT(
            initial_peers=initial_peers,
            start=True,
            num_workers=self.block_config.num_hidden_layers,
            use_relay=use_relay,
            use_auto_relay=use_auto_relay,
            client_mode=reachable_via_relay,
            **kwargs,
        )
        self.reachability_protocol = ReachabilityProtocol.attach_to_dht(self.dht) if not reachable_via_relay else None

        visible_maddrs_str = [str(a) for a in self.dht.get_visible_maddrs()]
        if initial_peers == PUBLIC_INITIAL_PEERS:
            logger.info("Connecting to the public swarm")
        else:
            logger.info(f"Connecting to a private swarm, initial peers: {initial_peers}")
        logger.info(f"Running a server on {visible_maddrs_str}")
        self.should_validate_reachability = not skip_reachability_check and initial_peers == PUBLIC_INITIAL_PEERS

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
        self.device = device

        torch_dtype = resolve_block_dtype(self.block_config, DTYPE_MAP[torch_dtype])
        if device.type == "cpu" and torch_dtype == torch.float16:
            raise ValueError(
                f"Type float16 is not supported on CPU. Please use --torch_dtype float32 or --torch_dtype bfloat16"
            )
        if device.type == "mps" and torch_dtype == torch.bfloat16:
            logger.warning(f"Type bfloat16 is not supported on MPS, using float16 instead")
            torch_dtype = torch.float16
        self.torch_dtype = torch_dtype

        if tensor_parallel_devices is None:
            tensor_parallel_devices = (device,)
        self.tensor_parallel_devices = tuple(map(torch.device, tensor_parallel_devices))
        if len(self.tensor_parallel_devices) > 1:
            logger.info(f"Model weights will be split between {', '.join(tensor_parallel_devices)}")
            check_device_balance(self.tensor_parallel_devices)

        if quant_type is None:
            quant_type = QuantType.NF4 if device.type == "cuda" else QuantType.NONE
        self.quant_type = quant_type
        logger.info(f"Model weights are loaded in {get_dtype_name(torch_dtype, quant_type)} format")

        is_multiquery_attn = self.block_config.num_key_value_groups > 1
        if max_batch_size is None:
            max_batch_size = 8192 if is_multiquery_attn else 2048
        if inference_max_length is None:
            inference_max_length = 8192 if is_multiquery_attn else 2048
        self.min_batch_size, self.max_batch_size = min_batch_size, max_batch_size
        self.inference_max_length = inference_max_length
        self.max_chunk_size_bytes = max_chunk_size_bytes
        self.max_alloc_timeout = max_alloc_timeout

        # For attention cache in GPU or RAM
        if attn_cache_tokens is None:
            attn_cache_tokens = 32768 if is_multiquery_attn else 8192
        cache_values_per_block = 2 * self.block_config.hidden_size * attn_cache_tokens
        cache_values_per_block //= self.block_config.num_key_value_groups
        self._cache_bytes_per_block = cache_values_per_block * get_size_in_bytes(self.torch_dtype)

        # For disk cache
        self.cache_dir = cache_dir
        self.max_disk_space = max_disk_space
        self.adapters = adapters

        assert num_blocks is None or block_indices is None, "Please specify num_blocks or block_indices, not both"
        if num_blocks is None and block_indices is None:
            num_blocks = self._choose_num_blocks()
        if num_blocks is not None:
            num_blocks = min(num_blocks, self.block_config.num_hidden_layers)
        if block_indices is not None:
            try:
                first_block_index, last_block_index = block_indices.split(":")
                first_block_index, last_block_index = map(int, map(str.strip, (first_block_index, last_block_index)))
            except Exception as e:
                raise ValueError(f"Failed to parse `--block_indices {block_indices}`, must be start:end (e.g. 0:18)")
            block_indices = range(first_block_index, last_block_index)
            num_blocks = len(block_indices)
        self.strict_block_indices, self.num_blocks = block_indices, num_blocks

        gib = 1024**3
        self.attn_cache_bytes = self._cache_bytes_per_block * num_blocks
        logger.info(f"Attention cache for all blocks will consume up to {self.attn_cache_bytes / gib:.2f} GiB")

        assert isinstance(throughput, float) or throughput in ["auto", "eval", "dry_run"]
        if throughput in ["auto", "eval", "dry_run"]:
            force_eval = throughput in ["eval", "dry_run"]
            throughput_info = get_server_throughput(
                converted_model_name_or_path,
                self.block_config,
                device,
                torch_dtype,
                num_blocks=num_blocks,
                quant_type=quant_type,
                tensor_parallel_devices=self.tensor_parallel_devices,
                reachable_via_relay=reachable_via_relay,
                force_eval=force_eval,
                cache_dir=cache_dir,
            )
            if throughput == "dry_run":
                logger.info("Finished estimating throughput, exiting")
                sys.exit(0)
        else:
            throughput_info = {"throughput": throughput}
        self.server_info = ServerInfo(
            state=ServerState.JOINING,
            public_name=public_name,
            version=petals.__version__,
            adapters=tuple(adapters),
            torch_dtype=str(torch_dtype).replace("torch.", ""),
            quant_type=quant_type.name.lower(),
            using_relay=reachable_via_relay,
            **throughput_info,
        )
        self.model_info = ModelInfo(num_blocks=self.block_config.num_hidden_layers)
        if not os.path.isdir(converted_model_name_or_path):
            self.model_info.repository = "https://huggingface.co/" + converted_model_name_or_path

        self.balance_quality = balance_quality
        self.mean_balance_check_period = mean_balance_check_period
        self.mean_block_selection_delay = mean_block_selection_delay

        self.module_container = None
        self.stop = threading.Event()

    def _choose_num_blocks(self) -> int:
        assert self.device.type in ("cuda", "mps"), (
            "GPU is not available. If you want to run a CPU-only server, please specify --num_blocks. "
            "CPU-only servers in the public swarm are discouraged since they are much slower"
        )
        num_devices = len(self.tensor_parallel_devices) if self.tensor_parallel_devices else 1

        if num_devices > 1:
            assert self.device.type == "cuda", f"Tensor parallelism is not supported on {self.device.type.upper()}"
            memory_per_device = tuple(
                torch.cuda.get_device_properties(device).total_memory for device in self.tensor_parallel_devices
            )
            total_memory = min(memory_per_device) * num_devices
            if max(memory_per_device) / min(memory_per_device) > 1.5:
                raise ValueError(
                    "GPU devices have highly uneven memory, which makes tensor parallelism inefficient. "
                    "Please launch individual servers on each GPU or set --num_blocks manually to "
                    "override this exception."
                )
        elif self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
        else:
            total_memory = psutil.virtual_memory().total

        gib = 1024**3
        # Estimate of GPU memory used in rpc_backward (2 GiB for BLOOM, proportional for other models)
        autograd_memory = 2 * gib * num_devices / 14336 * self.block_config.hidden_size

        block_size = get_block_size(self.block_config, "memory", dtype=self.torch_dtype, quant_type=self.quant_type)
        total_memory_per_block = block_size + self._cache_bytes_per_block
        if self.adapters:
            # Delay import of petals.utils.peft to avoid unnecessary import of bitsandbytes
            from petals.utils.peft import estimate_adapter_memory_per_block

            total_memory_per_block += estimate_adapter_memory_per_block(
                self.block_config,
                self.torch_dtype,
                self.adapters,
                token=self.token,
                cache_dir=self.cache_dir,
                max_disk_space=self.max_disk_space,
            )

        num_blocks = math.floor((total_memory - autograd_memory) / total_memory_per_block)
        assert num_blocks >= 1, "Your GPU does not have enough memory to serve at least one block"

        num_blocks = min(num_blocks, self.block_config.num_hidden_layers)
        logger.info(
            f"Server will fill your GPU memory with {num_blocks} transformer blocks. "
            f"If you want to leave some free GPU memory, please specify a lesser --num_blocks manually"
        )
        return num_blocks

    def run(self):
        while True:
            block_indices = self._choose_blocks()
            self.module_container = ModuleContainer.create(
                dht=self.dht,
                dht_prefix=self.dht_prefix,
                converted_model_name_or_path=self.converted_model_name_or_path,
                block_config=self.block_config,
                attn_cache_bytes=self.attn_cache_bytes,
                server_info=self.server_info,
                model_info=self.model_info,
                block_indices=block_indices,
                num_handlers=self.num_handlers,
                min_batch_size=self.min_batch_size,
                max_batch_size=self.max_batch_size,
                max_chunk_size_bytes=self.max_chunk_size_bytes,
                max_alloc_timeout=self.max_alloc_timeout,
                inference_max_length=self.inference_max_length,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
                max_disk_space=self.max_disk_space,
                device=self.device,
                compression=self.compression,
                stats_report_interval=self.stats_report_interval,
                update_period=self.update_period,
                expiration=self.expiration,
                request_timeout=self.request_timeout,
                session_timeout=self.session_timeout,
                step_timeout=self.step_timeout,
                prefetch_batches=self.prefetch_batches,
                sender_threads=self.sender_threads,
                revision=self.revision,
                token=self.token,
                quant_type=self.quant_type,
                tensor_parallel_devices=self.tensor_parallel_devices,
                should_validate_reachability=self.should_validate_reachability,
                start=True,
            )
            try:
                self.module_container.ready.wait()

                while True:
                    timeout = random.random() * 2 * self.mean_balance_check_period
                    if self.stop.wait(timeout):
                        return

                    if not self.module_container.is_healthy():
                        logger.warning("One of subprocesses crashed, restarting the server")
                        break

                    if self._should_choose_other_blocks():
                        logger.info("Swarm is imbalanced, server will load other blocks")
                        break  # Stop serving this set of modules
            finally:
                self.module_container.shutdown()

            self._clean_memory_and_fds()

    def _clean_memory_and_fds(self):
        self.module_container = None
        gc.collect()  # In particular, this closes unused file descriptors

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

            allocated_vram = torch.cuda.memory_allocated(self.device)
            reserved_vram = torch.cuda.memory_reserved(self.device)
            gib = 1024**3
            logger.info(
                f"Cleaning up, left {allocated_vram / gib:.1f} GiB allocated memory, "
                f"{reserved_vram / gib:.1f} GiB reserved memory"
            )
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def _choose_blocks(self) -> List[int]:
        if self.strict_block_indices is not None:
            return self.strict_block_indices

        # If multiple servers (e.g., launched on the same machine by a script) get to this line at the same time,
        # this delay decreases the probability of a race condition while choosing the best blocks to serve.
        time.sleep(random.random() * 2 * self.mean_block_selection_delay)
        module_infos = get_remote_module_infos(self.dht, self.module_uids, latest=True)
        return block_selection.choose_best_blocks(self.num_blocks, module_infos)

    def _should_choose_other_blocks(self) -> bool:
        if self.strict_block_indices is not None:
            return False

        module_infos = get_remote_module_infos(self.dht, self.module_uids, latest=True)
        return block_selection.should_choose_other_blocks(self.dht.peer_id, module_infos, self.balance_quality)

    def shutdown(self, timeout: Optional[float] = 5):
        self.stop.set()
        if self.module_container is not None and self.module_container.is_alive():
            self.module_container.join(timeout)

        if self.reachability_protocol is not None:
            self.reachability_protocol.shutdown()
        self.dht.shutdown()
        self.dht.join()


class ModuleContainer(threading.Thread):
    """Serves a set of specific Bloom layers for inference, forward, and backward. Announces itself over the DHT."""

    # noinspection PyMethodOverriding
    @classmethod
    def create(
        cls,
        *,
        dht: DHT,
        dht_prefix: str,
        converted_model_name_or_path: str,
        block_config: PretrainedConfig,
        attn_cache_bytes: int,
        server_info: ServerInfo,
        model_info: ModelInfo,
        block_indices: List[int],
        min_batch_size: int,
        max_batch_size: int,
        max_chunk_size_bytes: int,
        max_alloc_timeout: float,
        torch_dtype: torch.dtype,
        cache_dir: str,
        max_disk_space: int,
        device: Union[str, torch.device],
        compression: CompressionType,
        update_period: float,
        expiration: Optional[float],
        revision: Optional[str],
        token: Optional[Union[str, bool]],
        quant_type: QuantType,
        tensor_parallel_devices: Sequence[torch.device],
        should_validate_reachability: bool,
        **kwargs,
    ) -> ModuleContainer:
        module_uids = [f"{dht_prefix}{UID_DELIMITER}{block_index}" for block_index in block_indices]
        memory_cache = MemoryCache(attn_cache_bytes, max_alloc_timeout)

        server_info.state = ServerState.JOINING
        dht_announcer = ModuleAnnouncerThread(
            module_uids,
            dht,
            server_info,
            model_info,
            block_config=block_config,
            memory_cache=memory_cache,
            update_period=update_period,
            expiration=expiration,
            daemon=True,
        )
        dht_announcer.start()
        logger.info(f"Announced that blocks {block_indices} are joining")

        assert len(tensor_parallel_devices) >= 1 and all(isinstance(d, torch.device) for d in tensor_parallel_devices)

        blocks = {}
        try:
            for module_uid, block_index in zip(module_uids, block_indices):
                block = load_pretrained_block(
                    converted_model_name_or_path,
                    block_index,
                    config=block_config,
                    torch_dtype=torch_dtype,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    max_disk_space=max_disk_space,
                )
                block = convert_block(
                    block,
                    block_index,
                    block_config,
                    tensor_parallel_devices,
                    device,
                    quant_type,
                    adapters=server_info.adapters,
                    freeze=True,
                    token=token,
                    cache_dir=cache_dir,
                    max_disk_space=max_disk_space,
                )
                blocks[module_uid] = TransformerBackend(
                    module_uid,
                    block,
                    config=block_config,
                    memory_cache=memory_cache,
                    backend_dtype=torch_dtype,
                    max_chunk_size_bytes=max_chunk_size_bytes,
                    args_schema=(
                        BatchTensorDescriptor(
                            1, 2048, block_config.hidden_size, dtype=torch_dtype, compression=compression
                        ),
                    ),
                    kwargs_schema={},
                    outputs_schema=(
                        BatchTensorDescriptor(
                            1, 2048, block_config.hidden_size, dtype=torch_dtype, compression=compression
                        ),
                    ),
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                )

            merge_inference_pools_inplace(blocks)

            if should_validate_reachability:
                validate_reachability(dht.peer_id)
        except:
            logger.debug("Shutting down backends")
            for backend in blocks.values():
                backend.shutdown()

            dht_announcer.announce(ServerState.OFFLINE)
            logger.info(f"Announced that blocks {module_uids} are offline")
            raise

        return cls(
            dht,
            dht_prefix,
            blocks,
            dht_announcer=dht_announcer,
            server_info=server_info,
            update_period=update_period,
            expiration=expiration,
            **kwargs,
        )

    def __init__(
        self,
        dht: DHT,
        dht_prefix: str,
        module_backends: Dict[str, TransformerBackend],
        *,
        inference_max_length: int,
        num_handlers: int,
        dht_announcer: ModuleAnnouncerThread,
        server_info: ServerInfo,
        update_period: float,
        expiration: Optional[float] = None,
        request_timeout: float,
        session_timeout: float,
        step_timeout: float,
        start: bool,
        **kwargs,
    ):
        super().__init__()

        self.dht, self.module_backends = dht, module_backends
        self.server_info, self.update_period, self.expiration = server_info, update_period, expiration

        handler_event_queues = [mp.Queue() for _ in range(num_handlers)]
        self.conn_handlers = [
            TransformerConnectionHandler(
                dht,
                self.module_backends,
                adapters=server_info.adapters,
                dht_prefix=dht_prefix,
                handler_event_queues=handler_event_queues,
                handler_index=i,
                inference_max_length=inference_max_length,
                request_timeout=request_timeout,
                session_timeout=session_timeout,
                step_timeout=step_timeout,
                quant_type=QuantType[server_info.quant_type.upper()],
            )
            for i in range(num_handlers)
        ]

        self.runtime = RuntimeWithDeduplicatedPools(self.module_backends, device=None, **kwargs)
        # note: We set device=None in runtime to avoid moving all modules to device 0 in runtime.run(). tensor_parallel has already moved it as needed.

        dht_announcer.announce(ServerState.ONLINE)
        self.dht_announcer = dht_announcer

        if start:
            self.run_in_background(await_ready=True)

    def run(self):
        """
        Runs ModuleContainer in the current thread. Initializes dht if necessary, starts connection handlers,
        runs Runtime (self.runtime) to process incoming requests.
        """
        for handler in self.conn_handlers:
            handler.run_in_background()

        self.runtime.run()

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts ModuleContainer in a background thread. if await_ready, this method will wait until the container
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError("ModuleContainer didn't notify .ready in {timeout} seconds")

    @property
    def ready(self) -> mp.synchronize.Event:
        """
        An event (multiprocessing.Event) that is set when the container is ready to process requests.

        Example
        =======
        >>> container.start()
        >>> container.ready.wait(timeout=10)
        >>> print("Container ready" if container.ready.is_set() else "Container didn't start in 10 seconds")
        """
        return self.runtime.ready  # mp.Event that is true if self is ready to process batches

    def is_healthy(self) -> bool:
        return all(handler.is_alive() for handler in self.conn_handlers) and all(
            pool.is_alive() for pool in self.runtime.pools
        )

    def shutdown(self):
        """
        Gracefully terminate the container, process-safe.
        Please note that terminating container otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        self.dht_announcer.announce(ServerState.OFFLINE)
        logger.info(f"Announced that blocks {list(self.module_backends.keys())} are offline")

        self.ready.clear()

        logger.debug("Shutting down connection handlers")
        for handler in self.conn_handlers:
            handler.shutdown()

        logger.debug(f"Shutting down pools")
        for pool in self.runtime.pools:
            if pool.is_alive():
                pool.shutdown()

        logger.debug(f"Shutting down runtime")
        self.runtime.shutdown()

        logger.debug("Shutting down backends")
        for backend in self.module_backends.values():
            backend.shutdown()

        logger.info("Module container shut down successfully")


class ModuleAnnouncerThread(threading.Thread):
    """Periodically announces that this container hosts the specified modules, visible to all DHT peers"""

    def __init__(
        self,
        module_uids: List[str],
        dht: DHT,
        server_info: ServerInfo,
        model_info: ModelInfo,
        *,
        block_config: PretrainedConfig,
        memory_cache: MemoryCache,
        update_period: float,
        expiration: float,
        max_pinged: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.module_uids = module_uids
        self.dht = dht
        self.server_info = server_info
        self.model_info = model_info
        self.memory_cache = memory_cache

        self.bytes_per_token = block_config.hidden_size * get_size_in_bytes(DTYPE_MAP[server_info.torch_dtype])
        self.bytes_per_token //= block_config.num_key_value_groups

        self.update_period = update_period
        self.expiration = expiration
        self.trigger = threading.Event()

        self.max_pinged = max_pinged
        self.dht_prefix = module_uids[0].split(UID_DELIMITER)[0]
        block_indices = [int(uid.split(UID_DELIMITER)[-1]) for uid in module_uids]
        start_block, end_block = min(block_indices), max(block_indices) + 1
        self.next_uids = [f"{self.dht_prefix}{UID_DELIMITER}{i}" for i in range(start_block + 1, end_block + 1)]
        self.ping_aggregator = PingAggregator(self.dht)

    def run(self) -> None:
        while True:
            start_time = time.perf_counter()

            self.server_info.cache_tokens_left = self.memory_cache.bytes_left // self.bytes_per_token
            if self.server_info.state != ServerState.OFFLINE:
                self._ping_next_servers()
                self.server_info.next_pings = {
                    peer_id.to_base58(): rtt for peer_id, rtt in self.ping_aggregator.to_dict().items()
                }
            else:
                self.server_info.next_pings = None  # No need to ping if we're disconnecting

            declare_active_modules(
                self.dht,
                self.module_uids,
                self.server_info,
                expiration_time=get_dht_time() + self.expiration,
            )
            if self.server_info.state == ServerState.OFFLINE:
                break
            if not self.dht_prefix.startswith("_"):  # Not private
                self.dht.store(
                    key="_petals.models",
                    subkey=self.dht_prefix,
                    value=self.model_info.to_dict(),
                    expiration_time=get_dht_time() + self.expiration,
                )

            delay = self.update_period - (time.perf_counter() - start_time)
            if delay < 0:
                logger.warning(
                    f"Declaring blocks to DHT takes more than --update_period, consider increasing it (currently {self.update_period})"
                )
            self.trigger.wait(max(delay, 0))
            self.trigger.clear()

    def announce(self, state: ServerState) -> None:
        self.server_info.state = state
        self.trigger.set()
        if state == ServerState.OFFLINE:
            self.join()

    def _ping_next_servers(self) -> Dict[hivemind.PeerID, float]:
        module_infos = get_remote_module_infos(self.dht, self.next_uids, latest=True)
        middle_servers = {peer_id for info in module_infos[:-1] if info is not None for peer_id in info.servers}
        pinged_servers = set(sample_up_to(middle_servers, self.max_pinged))
        pinged_servers.discard(self.dht.peer_id)
        if module_infos[-1] is not None:
            # Sample servers hosting the block after the last one (most likely continuations) separately
            pinged_servers |= set(sample_up_to(module_infos[-1].servers, self.max_pinged))
        self.ping_aggregator.ping(list(pinged_servers))


class RuntimeWithDeduplicatedPools(Runtime):
    """A version of hivemind.moe.server.runtime.Runtime that allows multiple backends to reuse a task pool"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pools = tuple(set(self.pools))
