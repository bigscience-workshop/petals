from __future__ import annotations

import contextlib
import logging
import random

import torch
from hivemind import DHT, P2P, get_logger, use_hivemind_log_handler
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.moe.expert_uid import ExpertInfo
from torch import nn

from src import DistributedBloomConfig, RemoteTransformerBlock
from src.client.remote_sequence_info import RemoteSequenceInfo
from src.data_structures import UID_DELIMITER
from src.dht_utils import _create_remote_modules_from_infos


use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteSequential(nn.Module):
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
        self.prefix = prefix
        self.max_retries = max_retries
        self.p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p())

        block_uids = tuple(f"{prefix}{UID_DELIMITER}{i}" for i in range(config.n_layer))
        logger.debug(f"Remote block uids: {block_uids}")
        self.remote_sequence_info = RemoteSequenceInfo(dht, block_uids)

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
        (module,) = _create_remote_modules_from_infos([self.remote_sequence_info.block_infos[block_index]], self.p2p)
        return module

    def __iter__(self):
        for block_index in range(self.config.n_layer):
            yield self[block_index]

    def __len__(self):
        return len(self.remote_sequence_info)

    def inference_session(self) -> RemoteSequentialInferenceSession:
        self.remote_sequence_info.update_()
        return RemoteSequentialInferenceSession(self.remote_sequence_info, self.p2p)


class RemoteSequentialInferenceSession:
    """An interface to a multi-step *inference* session for a sequence of remote transformer blocks"""

    def __init__(self, remote_sequence_info: RemoteSequenceInfo, p2p: P2P):
        self.remote_sequence_info = remote_sequence_info
        self.p2p = p2p
        self.closed = False
        self.stack = contextlib.ExitStack()
        self.active_sessions = []

    def __enter__(self):
        assert not self.closed
        self.stack.__enter__()
        # TODO(yozh) replace this code with a fault-tolerant chain that can be reconstructed if some peers fail
        current_block = 0
        while current_block != len(self.remote_sequence_info):
            candidate_spans = self.remote_sequence_info.spans_containing_block[current_block]
            chosen_span = random.choice(candidate_spans)  # TODO this is a temporary code
            assert chosen_span.start <= current_block < chosen_span.end

            # TODO begin throwaway prototype code
            remote = RemoteTransformerBlock(self.remote_sequence_info.block_infos[current_block], self.p2p)
            _=remote.info #TODO fix
            span_uids = self.remote_sequence_info.block_uids[current_block: chosen_span.end]
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
