from __future__ import annotations

import gc
import math
import multiprocessing as mp
import random
import threading
import time
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from hivemind import DHT, MAX_DHT_TIME_DISCREPANCY_SECONDS, BatchTensorDescriptor, get_dht_time
from hivemind.moe.server.layers import add_custom_models_from_file
from hivemind.moe.server.runtime import Runtime
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger
from transformers import BloomConfig

from petals.bloom.from_pretrained import DTYPE_MAP, load_pretrained_block
from petals.constants import PUBLIC_INITIAL_PEERS
from petals.data_structures import CHAIN_DELIMITER, UID_DELIMITER, ServerState
from petals.dht_utils import declare_active_modules, get_remote_module_infos
from petals.server import block_selection
from petals.server.backend import TransformerBackend
from petals.server.block_utils import get_block_size
from petals.server.handler import TransformerConnectionHandler
from petals.server.memory_cache import MemoryCache
from petals.server.reachability import ReachabilityProtocol, check_direct_reachability, validate_reachability
from petals.server.throughput import get_dtype_name, get_host_throughput
from petals.utils.convert_block import check_device_balance, convert_block
from petals.utils.disk_cache import DEFAULT_CACHE_DIR

logger = get_logger(__file__)


class Server:
    """
    Runs ModuleContainer, periodically checks that the network is balanced,
    restarts the ModuleContainer with other layers if the imbalance is significant
    """

    def __init__(
        self,
        *,
        initial_peers: List[str],
        prefix: Optional[str],
        converted_model_name_or_path: str,
        throughput: Union[float, str],
        num_blocks: Optional[int] = None,
        block_indices: Optional[str] = None,
        num_handlers: int = 8,
        min_batch_size: int = 1,
        max_batch_size: int = 2048,
        inference_max_length: int = 2048,
        torch_dtype: str = "auto",
        revision: str = "main",
        cache_dir: Optional[str] = None,
        max_disk_space: Optional[int] = None,
        attn_cache_size: Optional[int] = None,
        alloc_timeout: float = 60,
        device: Optional[Union[str, torch.device]] = None,
        compression=CompressionType.NONE,
        stats_report_interval: Optional[int] = None,
        custom_module_path=None,
        update_period: float = 150,
        expiration: Optional[float] = None,
        request_timeout: float = 3 * 60,
        session_timeout: float = 30 * 60,
        step_timeout: float = 5 * 60,
        prefetch_batches: int = 1,
        sender_threads: int = 1,
        balance_quality: float = 0.75,
        mean_balance_check_period: float = 120,
        mean_block_selection_delay: float = 2.5,
        use_auth_token: Optional[str] = None,
        load_in_8bit: Optional[bool] = None,
        tensor_parallel_devices: Optional[Sequence[torch.device]] = None,
        skip_reachability_check: bool = False,
        dht_client_mode: Optional[bool] = None,
        use_relay: bool = True,
        use_auto_relay: bool = True,
        **kwargs,
    ):
        """Create a server with one or more bloom blocks. See run_server.py for documentation."""

        self.converted_model_name_or_path = converted_model_name_or_path
        self.num_handlers = num_handlers
        self.min_batch_size, self.max_batch_size = min_batch_size, max_batch_size
        self.inference_max_length = inference_max_length
        self.compression = compression
        self.stats_report_interval, self.update_period = stats_report_interval, update_period
        self.prefetch_batches, self.sender_threads = prefetch_batches, sender_threads
        self.use_auth_token = use_auth_token

        if custom_module_path is not None:
            add_custom_models_from_file(custom_module_path)

        if prefix is None:
            prefix = converted_model_name_or_path
            assert UID_DELIMITER not in prefix and CHAIN_DELIMITER not in prefix, (
                f"Cannot use model name as prefix (contains '{UID_DELIMITER}' or '{CHAIN_DELIMITER}'); "
                f"Please specify --prefix manually when starting a server"
            )
            logger.debug(f"Automatic dht prefix: {prefix}")
        self.prefix = prefix

        if expiration is None:
            expiration = max(2 * update_period, MAX_DHT_TIME_DISCREPANCY_SECONDS)
        self.expiration = expiration

        self.request_timeout = request_timeout
        self.session_timeout, self.step_timeout = session_timeout, step_timeout

        self.block_config = BloomConfig.from_pretrained(
            converted_model_name_or_path,
            use_auth_token=use_auth_token,
            revision=revision,
        )
        self.module_uids = [f"{self.prefix}.{block_index}" for block_index in range(self.block_config.n_layer)]

        if dht_client_mode is None:
            is_reachable = check_direct_reachability(initial_peers=initial_peers, use_relay=False, **kwargs)
            dht_client_mode = is_reachable is False  # if could not check reachability (returns None), run a full peer
            logger.info(f"This server will run DHT in {'client' if dht_client_mode else 'full peer'} mode")
        self.dht = DHT(
            initial_peers=initial_peers,
            start=True,
            num_workers=self.block_config.n_layer,
            use_relay=use_relay,
            use_auto_relay=use_auto_relay,
            client_mode=dht_client_mode,
            **kwargs,
        )
        self.reachability_protocol = ReachabilityProtocol.attach_to_dht(self.dht) if not dht_client_mode else None

        visible_maddrs_str = [str(a) for a in self.dht.get_visible_maddrs()]
        if initial_peers == PUBLIC_INITIAL_PEERS:
            logger.info(f"Connecting to the public swarm, peer_id = {self.dht.peer_id}")
        else:
            logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {initial_peers}")
        self.should_validate_reachability = not skip_reachability_check and initial_peers == PUBLIC_INITIAL_PEERS

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device(device.type, index=0)
        self.device = device

        if isinstance(torch_dtype, str):
            torch_dtype = DTYPE_MAP[torch_dtype]
        assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"
        self.torch_dtype = torch_dtype

        if tensor_parallel_devices is None:
            tensor_parallel_devices = (device,)
        self.tensor_parallel_devices = tuple(map(torch.device, tensor_parallel_devices))
        if len(self.tensor_parallel_devices) > 1:
            logger.info(f"Model weights will be split between {', '.join(tensor_parallel_devices)}")
            check_device_balance(self.tensor_parallel_devices)

        if load_in_8bit is None:
            load_in_8bit = device.type == "cuda"
            if load_in_8bit and len(self.tensor_parallel_devices) > 1:
                load_in_8bit = False
                logger.warning(
                    "Tensor parallelism doesn't work properly with 8-bit weights yet, loading weights in 16-bit. "
                    "You can explicitly set `--load_in_8bit True` to override this"
                )
        self.load_in_8bit = load_in_8bit
        logger.info(f"Model weights will be loaded in {get_dtype_name(torch_dtype, load_in_8bit)} format")

        assert num_blocks is None or block_indices is None, "Please specify num_blocks or block_indices, not both"
        if num_blocks is None and block_indices is None:
            num_blocks = self._choose_num_blocks()
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
        if attn_cache_size is None:
            # Hidden size is 14336 for the bigscience/bloom-petals model. For other models, scale accordingly
            attn_cache_size = 0.5 * gib * num_blocks * self.block_config.hidden_size / 14336
        self.attn_cache_size, self.alloc_timeout = attn_cache_size, alloc_timeout
        logger.info(f"Attention cache for all blocks will consume up to {attn_cache_size / gib:.2f} GiB")

        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR
        self.cache_dir = cache_dir
        self.max_disk_space = max_disk_space

        assert isinstance(throughput, float) or throughput in ["auto", "eval"]
        if throughput in ["auto", "eval"]:
            throughput = get_host_throughput(
                self.block_config,
                device,
                torch_dtype,
                load_in_8bit=load_in_8bit,
                tensor_parallel_devices=self.tensor_parallel_devices,
                force_eval=(throughput == "eval"),
                cache_dir=cache_dir,
            )
        self.throughput = throughput

        self.balance_quality = balance_quality
        self.mean_balance_check_period = mean_balance_check_period
        self.mean_block_selection_delay = mean_block_selection_delay

        self.stop = threading.Event()

    def _choose_num_blocks(self) -> int:
        assert (
            self.converted_model_name_or_path == "bigscience/bloom-petals"
        ), "If you use a model other than bigscience/bloom-petals, please specify --num_blocks manually"
        assert self.device.type == "cuda", (
            "GPU is not available. If you want to run a CPU-only server, please specify --num_blocks. "
            "CPU-only servers in the public swarm are discouraged since they are much slower"
        )
        num_devices = len(self.tensor_parallel_devices) if self.tensor_parallel_devices else 1

        if num_devices > 1:
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
        else:
            total_memory = torch.cuda.get_device_properties(self.device).total_memory

        block_size = get_block_size(self.block_config, "memory", dtype=self.torch_dtype, load_in_8bit=self.load_in_8bit)
        gib = 1024**3
        attn_cache_per_block = 0.5 * gib * num_devices  # TODO: This does not account for manually set --attn_cache_size

        autograd_memory = 2 * gib * num_devices  # gpu memory used for intermediate tensors in rpc_backward
        num_blocks = math.floor((total_memory - autograd_memory) / (block_size + attn_cache_per_block))
        assert num_blocks >= 1, "Your GPU does not have enough memory to serve at least one block"

        logger.info(
            f"Server will fill all your GPU memory with {num_blocks} transformer blocks. "
            f"If you want to leave some free GPU memory, please specify a lesser --num_blocks manually"
        )
        return min(num_blocks, self.block_config.n_layer)

    def run(self):
        while True:
            block_indices = self._choose_blocks()
            self.module_container = ModuleContainer.create(
                dht=self.dht,
                prefix=self.prefix,
                converted_model_name_or_path=self.converted_model_name_or_path,
                block_config=self.block_config,
                attn_cache_size=self.attn_cache_size,
                alloc_timeout=self.alloc_timeout,
                throughput=self.throughput,
                block_indices=block_indices,
                num_handlers=self.num_handlers,
                min_batch_size=self.min_batch_size,
                max_batch_size=self.max_batch_size,
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
                use_auth_token=self.use_auth_token,
                load_in_8bit=self.load_in_8bit,
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
        del self.module_container
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

    def _choose_blocks(self) -> List[int]:
        if self.strict_block_indices is not None:
            return self.strict_block_indices

        # If multiple servers (e.g., launched on the same machine by a script) get to this line at the same time,
        # this delay decreases the probability of a race condition while choosing the best blocks to serve.
        time.sleep(random.random() * 2 * self.mean_block_selection_delay)
        module_infos = get_remote_module_infos(self.dht, self.module_uids, expiration_time=np.inf)
        return block_selection.choose_best_blocks(self.num_blocks, module_infos)

    def _should_choose_other_blocks(self) -> bool:
        if self.strict_block_indices is not None:
            return False

        module_infos = get_remote_module_infos(self.dht, self.module_uids, expiration_time=np.inf)
        return block_selection.should_choose_other_blocks(self.dht.peer_id, module_infos, self.balance_quality)

    def shutdown(self):
        self.stop.set()

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
        prefix: str,
        converted_model_name_or_path: str,
        block_config: BloomConfig,
        attn_cache_size: int,
        alloc_timeout: float,
        throughput: float,
        block_indices: List[int],
        min_batch_size: int,
        max_batch_size: int,
        torch_dtype: torch.dtype,
        cache_dir: str,
        max_disk_space: int,
        device: Union[str, torch.device],
        compression: CompressionType,
        update_period: float,
        expiration: Optional[float],
        use_auth_token: Optional[str],
        load_in_8bit: bool,
        tensor_parallel_devices: Sequence[torch.device],
        should_validate_reachability: bool,
        **kwargs,
    ) -> ModuleContainer:
        module_uids = [f"{prefix}.{block_index}" for block_index in block_indices]
        joining_announcer = ModuleAnnouncerThread(
            module_uids,
            dht,
            ServerState.JOINING,
            throughput=throughput,
            update_period=update_period,
            expiration=expiration,
            daemon=True,
        )
        joining_announcer.start()
        logger.info(f"Announced that blocks {block_indices} are joining")

        assert len(tensor_parallel_devices) >= 1 and all(isinstance(d, torch.device) for d in tensor_parallel_devices)

        memory_cache = MemoryCache(attn_cache_size, alloc_timeout)
        blocks = {}
        try:
            for module_uid, block_index in zip(module_uids, block_indices):
                block = load_pretrained_block(
                    converted_model_name_or_path,
                    block_index,
                    block_config,
                    torch_dtype=torch_dtype,
                    use_auth_token=use_auth_token,
                    cache_dir=cache_dir,
                    max_disk_space=max_disk_space,
                )
                block = convert_block(block, block_config, tensor_parallel_devices, device, load_in_8bit, freeze=True)

                backend_dtype = next(block.parameters()).dtype if torch_dtype == "auto" else torch_dtype
                blocks[module_uid] = TransformerBackend(
                    module_uid,
                    block,
                    config=block_config,
                    memory_cache=memory_cache,
                    backend_dtype=backend_dtype,
                    args_schema=(
                        BatchTensorDescriptor(
                            1, 2048, block_config.hidden_size, dtype=backend_dtype, compression=compression
                        ),
                    ),
                    kwargs_schema={},
                    outputs_schema=(
                        BatchTensorDescriptor(
                            1, 2048, block_config.hidden_size, dtype=backend_dtype, compression=compression
                        ),
                    ),
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                )

            if should_validate_reachability:
                validate_reachability(dht.peer_id)
        except:
            logger.debug("Shutting down backends")
            for backend in blocks.values():
                backend.shutdown()

            joining_announcer.stop.set()
            joining_announcer.join()
            declare_active_modules(
                dht,
                module_uids,
                expiration_time=get_dht_time() + expiration,
                state=ServerState.OFFLINE,
                throughput=throughput,
            )
            logger.info(f"Announced that blocks {module_uids} are offline")
            raise
        else:
            joining_announcer.stop.set()
            joining_announcer.join()

        return cls(
            dht,
            blocks,
            throughput=throughput,
            device=device,
            update_period=update_period,
            expiration=expiration,
            **kwargs,
        )

    def __init__(
        self,
        dht: DHT,
        module_backends: Dict[str, TransformerBackend],
        *,
        inference_max_length: int,
        num_handlers: int,
        throughput: float,
        update_period: float,
        expiration: Optional[float] = None,
        request_timeout: float,
        session_timeout: float,
        step_timeout: float,
        device: Union[str, torch.device],
        start: bool,
        **kwargs,
    ):
        super().__init__()

        self.dht, self.module_backends = dht, module_backends
        self.throughput, self.update_period, self.expiration = throughput, update_period, expiration
        self.conn_handlers = [
            TransformerConnectionHandler(
                dht,
                self.module_backends,
                inference_max_length=inference_max_length,
                request_timeout=request_timeout,
                session_timeout=session_timeout,
                step_timeout=step_timeout,
            )
            for _ in range(num_handlers)
        ]
        self.runtime = Runtime(self.module_backends, device=None, **kwargs)
        # note: We set device=None in runtime to avoid moving all modules to device 0 in runtime.run(). tensor_parallel has already moved it as needed.
        self.online_announcer = ModuleAnnouncerThread(
            list(self.module_backends.keys()),
            dht,
            ServerState.ONLINE,
            throughput=throughput,
            update_period=update_period,
            expiration=expiration,
            daemon=True,
        )
        self.checkpoint_saver = None  # no need to save checkpoints since we do not change model state

        if start:
            self.run_in_background(await_ready=True)

    def run(self):
        """
        Runs ModuleContainer in the current thread. Initializes dht if necessary, starts connection handlers,
        runs Runtime (self.runtime) to process incoming requests.
        """
        if not self.dht.is_alive():
            self.dht.run_in_background(await_ready=True)

        self.online_announcer.start()

        if self.checkpoint_saver is not None:
            self.checkpoint_saver.start()

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
        self.online_announcer.stop.set()
        self.online_announcer.join()

        declare_active_modules(
            self.dht,
            self.module_backends.keys(),
            expiration_time=get_dht_time() + self.expiration,
            state=ServerState.OFFLINE,
            throughput=self.throughput,
        )
        logger.info(f"Announced that blocks {list(self.module_backends.keys())} are offline")

        self.ready.clear()

        for handler in self.conn_handlers:
            handler.shutdown()
        logger.debug("Connection handlers terminated")

        if self.checkpoint_saver is not None:
            self.checkpoint_saver.stop.set()
            self.checkpoint_saver.join()

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
        state: ServerState,
        *,
        throughput: float,
        update_period: float = 30,
        expiration: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.module_uids = module_uids
        self.dht = dht
        self.state = state
        self.throughput = throughput
        self.update_period = update_period
        self.expiration = expiration
        self.stop = threading.Event()

    def run(self) -> None:
        while True:
            declare_active_modules(
                self.dht,
                self.module_uids,
                expiration_time=get_dht_time() + self.expiration,
                state=self.state,
                throughput=self.throughput,
            )
            if self.stop.wait(self.update_period):
                break
