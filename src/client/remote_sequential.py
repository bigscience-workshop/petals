from __future__ import annotations

import logging
from typing import Optional, Union

import torch
from hivemind import DHT, P2P, get_logger, use_hivemind_log_handler
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from torch import nn

import src
from src.client.inference_session import RemoteSequentialInferenceSession
from src.client.remote_block import RemoteTransformerBlock
from src.client.sequence_manager import RemoteSequenceManager
from src.data_structures import UID_DELIMITER
from src.dht_utils import _create_remote_modules_from_infos

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteSequential(nn.Module):
    """
    A sequence of transformer blocks hosted by the swarm.
    """

    def __init__(
        self,
        config: src.DistributedBloomConfig,
        dht: DHT,
        prefix: str,
        p2p: Optional[P2P] = None,
        sequence_manager: Optional[RemoteSequenceManager] = None,
    ):
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
        self.prefix = prefix
        self.p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p()) if p2p is None else p2p

        block_uids = [f"{prefix}{UID_DELIMITER}{i}" for i in range(config.n_layer)]
        if sequence_manager is None:
            logger.debug(f"Creating new sequence manager for block uids: {block_uids}")
            self.sequence_manager = RemoteSequenceManager(dht, block_uids, self.p2p)
            self.is_subsequence = False
        else:
            assert isinstance(sequence_manager.block_uids, list)
            logger.debug(f"Reusing sequence manager with {len(self.sequence_manager)}")
            self.is_subsequence = self.sequence_manager.block_uids == block_uids

    def forward(self, inputs: torch.Tensor):
        assert isinstance(inputs, torch.Tensor) and inputs.ndim == 3 and inputs.shape[-1] == self.config.n_embed
        for block_index in range(self.config.n_layer):
            for retry_index in range(self.sequence_manager.max_retries):
                try:
                    block = self[block_index]
                    (outputs,) = block(inputs)
                    assert isinstance(outputs, torch.Tensor)
                    assert outputs.shape == inputs.shape, f"Expected {block} output {inputs.shape}, got {outputs.shape}"
                    inputs = outputs
                    break
                except Exception as e:
                    if retry_index == self.sequence_manager.max_retries - 1:
                        raise e
                    else:
                        logging.debug(f"Caught {e} when running forward for block {block_index}", exc_info=True)
        return inputs

    def __getitem__(self, ix: Union[int, slice]) -> Union[RemoteTransformerBlock, RemoteSequential]:
        assert isinstance(ix, (int, slice))
        if isinstance(ix, int):
            assert 0 <= ix < self.config.n_layer
            (module,) = _create_remote_modules_from_infos([self.sequence_manager.block_infos[ix]], self.p2p)
            return module
        else:
            return RemoteSequential(
                self.config,
                self.dht,
                prefix=self.prefix,
                p2p=self.p2p,
                sequence_manager=self.sequence_manager[ix],
            )

    def __iter__(self):
        for block_index in range(self.config.n_layer):
            yield self[block_index]

    def __len__(self):
        return len(self.sequence_manager)

    def inference_session(self) -> RemoteSequentialInferenceSession:
        self.sequence_manager.update_()
        return RemoteSequentialInferenceSession(self.sequence_manager, self.p2p)

    def extra_repr(self) -> str:
        return f"{self.sequence_manager.block_uids[0]}..{self.sequence_manager.block_uids[-1]}"
