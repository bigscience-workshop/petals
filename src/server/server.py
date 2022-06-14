import threading
from typing import Optional, Dict, Union, Sequence

import torch
from hivemind import Server, DHT
from hivemind.moe.server.dht_handler import DHTHandlerThread
from hivemind.moe.server.layers import add_custom_models_from_file
from hivemind.moe.server.runtime import Runtime
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import use_hivemind_log_handler, get_logger

from src import DistributedBloomConfig
from src.bloom.block import BloomBlock
from src.server.cache import MemoryCache
from src.server.backend import BloomBlockBackend
from src.server.handler import BloomConnectionHandler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class BloomServer(Server):
    """Serves one or more bloom layers for inference, forward and backward; announces oneself to the DHT"""
    def __init__(
            self, dht: DHT, module_backends: Dict[str, BloomBlockBackend], *,
            device: torch.device, num_connection_handlers: int = 8, update_period: float = 30,
            cache_size_bytes: Optional[int] = None, start: bool, **kwargs,
    ):
        threading.Thread.__init__(self)
        self.attention_cache = MemoryCache(device=device, max_size_bytes=cache_size_bytes)

        self.dht, self.module_backends, self.update_period = dht, module_backends, update_period
        self.conn_handlers = [BloomConnectionHandler(dht, self.module_backends) for _ in range(num_connection_handlers)]
        self.runtime = Runtime(self.module_backends, device=device, **kwargs)
        self.dht_handler_thread = DHTHandlerThread(self.experts, dht, update_period=update_period, daemon=True)
        self.checkpoint_saver = None  # no need to save checkpoints since we do not change model state

        if start:
            self.run_in_background(await_ready=True)

    # noinspection PyMethodOverriding
    @classmethod
    def create(
            cls,
            num_blocks: int,
            block_config: str,
            num_handlers: Optional[int] = None,
            min_batch_size: int = 1,
            max_batch_size: int = 4096,
            cache_size_bytes: Optional[int] = None,
            device: Union[str, torch.device] = None,
            initial_peers: Sequence[str] = (),
            compression=CompressionType.NONE,
            stats_report_interval: Optional[int] = None,
            custom_module_path=None,
            update_period: float = 30,
            expiration: Optional[float] = None,
            *,
            start: bool,
            **kwargs,
    ) -> Server:
        """Create a server with one or more bloom blocks. See run_server.py for documentation."""
        if custom_module_path is not None:
            add_custom_models_from_file(custom_module_path)

        dht = DHT(initial_peers=initial_peers, start=True, **kwargs)
        visible_maddrs_str = [str(a) for a in dht.get_visible_maddrs()]
        logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {initial_peers}")

        num_handlers = num_handlers if num_handlers is not None else num_blocks * 8
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(block_config, str):
            block_config = DistributedBloomConfig

        # initialize modules
        module_backends = {}
        for i in range(len(module_backends)):
            module_uid = f"dummy_block.{i}"
            block = BloomBlock(block_config, layer_number=i)
            #TODO run the actual model

            module_backends[module_uid] = BloomBlockBackend(
                name=expert_uid,
                expert=block,
                args_schema=args_schema,
                num_warmup_steps=num_warmup_steps,
                num_total_steps=num_total_steps,
                clip_grad_norm=clip_grad_norm,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
            )

        if checkpoint_dir is not None:
            load_experts(experts, checkpoint_dir)

        return cls(
            dht,
            experts,
            cache_size_bytes=cache_size_bytes,
            num_connection_handlers=num_handlers,
            device=device,
            checkpoint_dir=checkpoint_dir,
            stats_report_interval=stats_report_interval,
            update_period=update_period,
            expiration=expiration,
            start=start,
        )

