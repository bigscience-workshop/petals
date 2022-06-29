import logging
from functools import partial
from typing import Optional, Tuple

import torch
from hivemind import DHT, get_logger, use_hivemind_log_handler
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from torch import nn

from src import DistributedBloomConfig
from src.data_structures import UID_DELIMITER, RemoteModuleInfo
from src.dht_utils import _create_remote_modules_from_infos, _get_remote_module_infos

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteSequential(nn.Sequential):
    """
    A sequence of transformer blocks hosted by the swarm.
    """

    def __init__(self, config: DistributedBloomConfig, dht: DHT, prefix: str, max_retries: int = 3):
        logger.warning(f"{self.__class__.__name__} is in active development; expect adventures")
        if prefix.endswith(UID_DELIMITER):
            logger.warning(
                f"dht_prefix {prefix} already ends with '{UID_DELIMITER}'."
                f"This will cause {self.__class__.__name__} to look for modules under "
                f"{prefix}{UID_DELIMITER}*. Please make sure this is what you intended."
            )

        super().__init__()
        self.config = config
        self.dht = dht
        self.p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p())

        self.prefix = prefix
        self.block_uids = tuple(f"{prefix}{UID_DELIMITER}{i}" for i in range(config.n_layer))
        logger.debug(f"Remote block uids: {self.block_uids}")
        self.block_infos: Tuple[RemoteModuleInfo, ...] = tuple(
            dht.run_coroutine(
                partial(_get_remote_module_infos, uids=self.block_uids, expiration_time=float("inf")),
                return_future=False,
            )
        )

        self.max_retries = max_retries

        assert len(self.block_infos) == len(self.block_uids)
        for uid, info in zip(self.block_uids, self.block_infos):
            assert isinstance(info, (type(None), RemoteModuleInfo)), f"Unexpected dht entry for {uid}: {info}"
            assert info is not None, f"Found no active peers for block {uid}"
            assert isinstance(info.peer_ids, set), f"expected peer_ids to be a set, got {info.peer_ids}"
            assert info.uid == uid, f"The DHT entry for {uid} actually points to {info.uid}"
            assert len(info.peer_ids) > 0, f"Found no active peers for block {uid}"

    def forward(self, inputs: torch.Tensor):
        assert isinstance(inputs, torch.Tensor) and inputs.ndim == 3 and inputs.shape[-1] == self.config.n_embed
        for block_index in range(self.config.n_layer):
            for retry_index in range(self.max_retries):
                try:
                    block = self[block_index]
                    (outputs,) = block(inputs)
                    assert isinstance(outputs, torch.Tensor)
                    assert outputs.shape == inputs.shape, f"Expected {block} output {inputs.shape}, got {outputs.shape}"
                    inputs = outputs
                    break
                except Exception as e:
                    if retry_index == self.max_retries - 1:
                        raise e
                    else:
                        logging.debug(f"Caught {e} when running forward for block {block_index}", exc_info=True)
        return inputs

    def __getitem__(self, block_index: int):
        assert 0 <= block_index < self.config.n_layer
        (module,) = _create_remote_modules_from_infos([self.block_infos[block_index]], self.p2p)
        return module

    def __iter__(self):
        for block_index in range(self.config.n_layer):
            yield self[block_index]
