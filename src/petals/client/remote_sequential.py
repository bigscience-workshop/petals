from __future__ import annotations

from typing import Optional, Union

import torch
from hivemind import DHT, get_logger
from torch import nn

from petals.client.inference_session import InferenceSession
from petals.client.routing.sequence_manager import RemoteSequenceManager, SequenceManagerConfig
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
        config: SequenceManagerConfig,
        *,
        sequence_manager: Optional[RemoteSequenceManager] = None,
        dht: Optional[DHT] = None,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        assert sequence_manager is None or (
            dht is None and start_block is None and end_block is None
        ), "`dht`, `start_block`, and `end_block` have no effect when you provide a custom `sequence_manager`"
        if sequence_manager is None:
            if start_block is None:
                start_block = 0
            if end_block is None:
                end_block = self.config.num_hidden_layers
            block_uids = tuple(f"{config.dht_prefix}{UID_DELIMITER}{i}" for i in range(start_block, end_block))
            sequence_manager = RemoteSequenceManager(config, block_uids, dht=dht, **kwargs)
        self.sequence_manager = sequence_manager

    def forward(self, inputs: torch.Tensor, prompts: torch.Tensor = DUMMY):
        assert inputs.ndim == 3, "inputs must be a tensor of shape [batch_size, seq_length, hidden_size]"
        assert inputs.shape[1] <= 2048, "The sequence length is capped at 2048 tokens in this version"
        outputs = _RemoteSequentialAutogradFunction.apply(inputs, prompts, self.sequence_manager)
        return outputs

    def __getitem__(self, ix: Union[int, slice]) -> RemoteSequential:
        return RemoteSequential(
            self.config,
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
