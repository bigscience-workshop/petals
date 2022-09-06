from __future__ import annotations

from typing import Optional, Union

import torch
from hivemind import DHT, P2P, get_logger, use_hivemind_log_handler
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from torch import nn

import src
from src.client.inference_session import RemoteSequentialInferenceSession
from src.client.sequence_manager import RemoteSequenceManager
from src.client.sequential_autograd import _RemoteSequentialAutogradFunction
from src.data_structures import UID_DELIMITER
from src.utils.misc import DUMMY

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
        dht_prefix: Optional[str] = None,
        p2p: Optional[P2P] = None,
        sequence_manager: Optional[RemoteSequenceManager] = None,
    ):
        logger.warning(f"{self.__class__.__name__} is in active development; expect adventures")
        super().__init__()
        self.config = config
        self.dht = dht
        self.dht_prefix = dht_prefix or config.dht_prefix
        self.p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p()) if p2p is None else p2p

        num_blocks = self.config.n_layer if sequence_manager is None else len(sequence_manager)
        block_uids = [f"{config.dht_prefix}{UID_DELIMITER}{i}" for i in range(num_blocks)]
        if sequence_manager is None:
            logger.debug(f"Creating new sequence manager for block uids: {block_uids}")
            self.sequence_manager = RemoteSequenceManager(dht, block_uids, self.p2p)
            self.is_subsequence = False
        else:
            logger.debug(f"Reusing sequence manager with {len(sequence_manager)} modules")
            self.sequence_manager = sequence_manager
            assert isinstance(sequence_manager.block_uids, list)
            self.is_subsequence = self.sequence_manager.block_uids != block_uids

    def forward(self, inputs: torch.Tensor, prompts: torch.Tensor = DUMMY):
        outputs = _RemoteSequentialAutogradFunction.apply(inputs, prompts, self.sequence_manager)
        return outputs

    def __getitem__(self, ix: Union[int, slice]) -> RemoteSequential:
        assert isinstance(ix, (int, slice))
        if isinstance(ix, int):
            return RemoteTransformerBlock(
                self.config,
                self.dht,
                dht_prefix=self.dht_prefix,
                p2p=self.p2p,
                sequence_manager=self.sequence_manager[ix],
            )
        else:
            return RemoteSequential(
                self.config,
                self.dht,
                dht_prefix=self.dht_prefix,
                p2p=self.p2p,
                sequence_manager=self.sequence_manager[ix],
            )

    def __iter__(self):
        for block_index in range(len(self)):
            yield self[block_index]

    def __len__(self):
        return len(self.sequence_manager)

    def inference_session(self, **kwargs) -> RemoteSequentialInferenceSession:
        self.sequence_manager.update_()
        return RemoteSequentialInferenceSession(self.sequence_manager, self.p2p, **kwargs)

    def extra_repr(self) -> str:
        return f"modules={self.sequence_manager.block_uids[0]}..{self.sequence_manager.block_uids[-1]}"


class RemoteTransformerBlock(RemoteSequential):
    """Single transformer block hosted by swarm

    This class is deprecated and kept for backward compatibility.
    It will be removed soon in favor of using ``RemoteSequential`` directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self) == 1, "Remote Block is a sequence size 1"

    def extra_repr(self):
        return f"{self.sequence_manager.block_uids[0]}"
