from __future__ import annotations

from typing import Optional, Union

import torch
from hivemind import DHT, get_logger
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from torch import nn

import petals.client
from petals.client.inference_session import InferenceSession
from petals.client.routing.sequence_manager import RemoteSequenceManager
from petals.client.sequential_autograd import _RemoteSequentialAutogradFunction
from petals.data_structures import UID_DELIMITER
from petals.utils.misc import DUMMY

logger = get_logger(__name__)


class RemoteSequential(nn.Module):
    """
    A sequence of transformer blocks hosted by the swarm.
    """

    def __init__(
        self,
        config: petals.client.DistributedBloomConfig,
        dht: DHT,
        dht_prefix: Optional[str] = None,
        sequence_manager: Optional[RemoteSequenceManager] = None,
    ):
        super().__init__()
        self.config = config
        self.dht = dht
        self.dht_prefix = dht_prefix or config.dht_prefix

        num_blocks = self.config.n_layer if sequence_manager is None else len(sequence_manager)
        block_uids = tuple(f"{config.dht_prefix}{UID_DELIMITER}{i}" for i in range(num_blocks))

        if sequence_manager is None:
            sequence_manager = RemoteSequenceManager(dht, block_uids, config)
        self.sequence_manager = sequence_manager

    def forward(self, inputs: torch.Tensor, prompts: torch.Tensor = DUMMY):
        assert inputs.ndim == 3, "inputs must be a tensor of shape [batch_size, seq_length, hidden_size]"
        assert inputs.shape[1] <= 2048, "The sequence length is capped at 2048 tokens in this version"
        outputs = _RemoteSequentialAutogradFunction.apply(inputs, prompts, self.sequence_manager)
        return outputs

    def __getitem__(self, ix: Union[int, slice]) -> RemoteSequential:
        return RemoteSequential(
            self.config,
            self.dht,
            dht_prefix=self.dht_prefix,
            sequence_manager=self.sequence_manager[ix],
        )

    def __iter__(self):
        for block_index in range(len(self)):
            yield self[block_index]

    def __len__(self):
        return len(self.sequence_manager)

    def inference_session(self, **kwargs) -> InferenceSession:
        return InferenceSession(self.sequence_manager, **kwargs)

    def extra_repr(self) -> str:
        return f"modules={self.sequence_manager.block_uids[0]}..{self.sequence_manager.block_uids[-1]}"
