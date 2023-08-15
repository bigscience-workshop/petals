from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional, Union

import torch
from hivemind import DHT, get_logger
from torch import nn

from petals.client.config import ClientConfig
from petals.client.inference_session import InferenceSession
from petals.client.routing import RemoteSequenceManager
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
        config: ClientConfig,
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

        self._thread_local = threading.local()
        self._thread_local.active_session = None

    def forward(self, inputs: torch.Tensor, prompts: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        assert inputs.ndim == 3, "inputs must be a tensor of shape [batch_size, seq_length, hidden_size]"
        if self._thread_local.active_session is None:
            assert all(v is None for v in kwargs.values()), f"Extra kwargs are not supported in forward: {kwargs}"
            return _RemoteSequentialAutogradFunction.apply(inputs, prompts, self.sequence_manager)
        else:
            return self._thread_local.active_session.step(inputs, prompts, **kwargs)

    @property
    def active_session(self) -> Optional[InferenceSession]:
        return self._thread_local.active_session

    @contextmanager
    def use_session(self, session: Optional[InferenceSession]) -> InferenceSession:
        """Inside this context, forward() will use the specified InferenceSession."""

        try:
            prev_session = self._thread_local.active_session
            self._thread_local.active_session = session
            yield session
        finally:
            self._thread_local.active_session = prev_session

    @contextmanager
    def inference_session(self, **kwargs) -> InferenceSession:
        """Inside this context, forward() will use a new InferenceSession created with given parameters."""

        with self.use_session(InferenceSession(self.sequence_manager, **kwargs)) as session:
            yield session

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

    def extra_repr(self) -> str:
        return f"modules={self.sequence_manager.block_uids[0]}..{self.sequence_manager.block_uids[-1]}"
