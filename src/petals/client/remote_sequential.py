from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional, Union

import torch
from hivemind import DHT, get_logger
from torch import nn

from petals.client.config import ClientConfig
from petals.client.inference_session import InferenceSession
from petals.client.routing import RemoteSequenceManager
from petals.client.sequential_autograd import _RemoteSequentialAutogradFunction
from petals.data_structures import UID_DELIMITER

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

        self._active_session = ContextVar("active_session", default=None)

    def forward(self, inputs: torch.Tensor, prompts: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        assert inputs.ndim == 3, "inputs must be a tensor of shape [batch_size, seq_length, hidden_size]"
        if self.active_session is None:
            assert all(v is None for v in kwargs.values()), f"Extra kwargs are not supported in forward: {kwargs}"
            return _RemoteSequentialAutogradFunction.apply(inputs, prompts, self.sequence_manager)
        else:
            return self.active_session.step(inputs, prompts, **kwargs)

    @property
    def active_session(self) -> Optional[InferenceSession]:
        """
        If called inside `with model.inference_session(...):` or `with model.use_session(...):`,
        returns an active InferenceSession. Otherwise, returns None.
        """

        return self._active_session.get()

    @property
    def position(self) -> int:
        """Returns the prefix length (in tokens) in the active inference session or zero if no session is active."""

        return self.active_session.position if self.active_session is not None else 0

    @contextmanager
    def use_session(self, session: Optional[InferenceSession]) -> InferenceSession:
        """Inside this context, forward() will use an _existing_ InferenceSession provided as the argument."""

        token = self._active_session.set(session)
        try:
            yield session
        finally:
            self._active_session.reset(token)

    @contextmanager
    def inference_session(self, **kwargs) -> InferenceSession:
        """
        Inside this context, forward() will use a _new_ InferenceSession created with given parameters.

        :param max_length: Maximal expected length of inference results. Servers use this parameter
                           to calculate the size of attention caches allocated to this client.
        """

        with InferenceSession(self.sequence_manager, **kwargs) as session, self.use_session(session):
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
