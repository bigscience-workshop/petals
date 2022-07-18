from __future__ import annotations

import contextlib
import logging
import random
from typing import Optional, Union

import torch
from hivemind import DHT, P2P, get_logger, use_hivemind_log_handler
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.moe.expert_uid import ExpertInfo
from torch import nn

import src
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
        max_retries: int = 3,
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
        self.max_retries = max_retries
        self.p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p()) if p2p is None else p2p

        block_uids = [f"{prefix}{UID_DELIMITER}{i}" for i in range(config.n_layer)]
        if sequence_manager is None:
            logger.debug(f"Creating new sequence manager for block uids: {block_uids}")
            self.sequence_manager = RemoteSequenceManager(dht, block_uids)
            self.is_subsequence = False
        else:
            assert isinstance(sequence_manager.block_uids, list)
            logger.debug(f"Reusing sequence manager with {len(self.sequence_manager)}")
            self.is_subsequence = self.sequence_manager.block_uids == block_uids

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
                max_retries=self.max_retries,
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


class RemoteSequentialInferenceSession:
    """An interface to a multi-step *inference* session for a sequence of remote transformer blocks"""

    def __init__(self, sequence_manager: RemoteSequenceManager, p2p: P2P):
        self.sequence_manager = sequence_manager
        self.p2p = p2p
        self.closed = False
        self.stack = contextlib.ExitStack()
        self.active_sessions = []

    def __enter__(self):
        assert not self.closed
        self.stack.__enter__()
        # TODO(yozh) replace this code with a fault-tolerant chain that can be reconstructed if some peers fail
        current_block = 0
        while current_block != len(self.sequence_manager):
            candidate_spans = self.sequence_manager.spans_containing_block[current_block]
            chosen_span = random.choice(candidate_spans)  # TODO this is a temporary code
            assert chosen_span.start <= current_block < chosen_span.end

            # TODO begin throwaway prototype code
            remote = RemoteTransformerBlock(self.sequence_manager.block_infos[current_block], self.p2p)
            _ = remote.info  # TODO fix
            span_uids = self.sequence_manager.block_uids[current_block: chosen_span.end]
            remote._info = ExpertInfo(" ".join(span_uids), chosen_span.peer_id)
            self.active_sessions.append(remote.inference_session())
            self.stack.enter_context(self.active_sessions[-1])
            current_block = chosen_span.end
            # TODO end throwaway prototype code

        return self

    def step(self, inputs: torch.Tensor):
        assert not self.closed
        for session in self.active_sessions:
            outputs = session.step(inputs)
            assert outputs.shape == inputs.shape, f"expected {inputs.shape}, got {outputs.shape}"
            inputs = outputs
        return inputs

    def close(self, *exc_details):
        """Finish a given inference session, close the underlying connection"""
        if not self.closed:
            self.stack.__exit__(*exc_details or (None, None, None))
            self.active_sessions.clear()
            self.closed = True

    def __exit__(self, *exc_details):
        self.close(*exc_details)

    def __del__(self):
        self.close()
