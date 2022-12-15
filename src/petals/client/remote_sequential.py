from __future__ import annotations

from typing import Optional, Union

import torch
from hivemind import DHT, P2P, get_logger
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from torch import nn

import petals.client
from petals.client.inference_session import InferenceSession
from petals.client.routing.sequence_manager import RemoteSequenceManager
from petals.client.sequential_autograd import _RemoteSequentialAutogradFunction
from petals.data_structures import UID_DELIMITER
from petals.utils.misc import DUMMY

logger = get_logger(__file__)


class RemoteSequential(nn.Module):
    """
    A sequence of transformer blocks hosted by the swarm.
    """

    def __init__(
        self,
        config: petals.client.DistributedBloomConfig,
        dht: DHT,
        dht_prefix: Optional[str] = None,
        p2p: Optional[P2P] = None,
        sequence_manager: Optional[RemoteSequenceManager] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.dht = dht
        self.dht_prefix = dht_prefix or config.dht_prefix
        self.p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p()) if p2p is None else p2p

        num_blocks = self.config.n_layer if sequence_manager is None else len(sequence_manager)
        block_uids = tuple(f"{config.dht_prefix}{UID_DELIMITER}{i}" for i in range(num_blocks))
        if sequence_manager is None:
            logger.debug(f"Creating new sequence manager for block uids: {block_uids}")
            self.sequence_manager = RemoteSequenceManager(dht, block_uids, self.p2p, start=True, **kwargs)
            self.is_subsequence = False
        else:
            logger.debug(f"Reusing sequence manager with {len(sequence_manager)} modules")
            if kwargs:
                logger.warning(f"Parameters {kwargs} are ignored because sequence_manager is explicitly provided")
            self.sequence_manager = sequence_manager
            assert isinstance(sequence_manager.sequence_info.block_uids, tuple)
            self.is_subsequence = self.sequence_manager.sequence_info.block_uids != block_uids

    def forward(self, inputs: torch.Tensor, prompts: torch.Tensor = DUMMY):
        assert inputs.ndim == 3, "inputs must be a tensor of shape [batch_size, seq_length, hidden_size]"
        assert inputs.shape[1] <= 2048, "The sequence length is capped at 2048 tokens in this version"
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

    def inference_session(self, **kwargs) -> InferenceSession:
        return InferenceSession(self.sequence_manager, self.p2p, **kwargs)

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
