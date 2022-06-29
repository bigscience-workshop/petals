from __future__ import annotations

import dataclasses
import logging
import threading
from functools import partial
from typing import Optional, Tuple, NamedTuple, List, Sequence

import torch
from hivemind import DHT, get_logger, use_hivemind_log_handler, PeerID
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.proto import runtime_pb2
from torch import nn

from src import DistributedBloomConfig
from src.data_structures import UID_DELIMITER, RemoteModuleInfo, ModuleUID
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
        self.prefix = prefix
        self.max_retries = max_retries
        self.p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p())

        self.block_uids = tuple(f"{prefix}{UID_DELIMITER}{i}" for i in range(config.n_layer))
        logger.debug(f"Remote block uids: {self.block_uids}")
        self.remote_model_info = RemoteModelInfo(dht, self.block_uids)

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

    def inference_session(self) -> RemoteSequentialInferenceSession:
        self.remote_model_info.update_()
        return RemoteExpertWorker.run_coroutine(RemoteSequentialInferenceSession._create(self))


Span = NamedTuple('Span', [('start', int), ('end', Optional[int]), ('peer_id', PeerID)])


@dataclasses.dataclass(frozen=False, init=False)
class RemoteModelInfo:
    """Stores meta-information about which peers host which blocks - and prepare to form sessions"""
    dht: DHT
    block_uids: Tuple[ModuleUID, ...]
    block_infos: List[Optional[RemoteModuleInfo], ...]
    spans_by_priority: List[Span]  # sorted from best to worst
    spans_containing_block: Tuple[List[Span], ...]
    lock_changes: threading.Lock

    def __init__(self, dht: DHT, block_uids: Sequence[ModuleUID]):
        self.dht = dht
        self.block_uids = block_uids
        self.block_infos: List[Optional[RemoteModuleInfo], ...] = [None] * len(self.block_uids)
        self.spans_by_priority = []
        self.spans_containing_block = tuple(list() for _ in range(len(self.block_uids)))
        self.lock_changes = threading.Lock()
        self.update_()

        for uid, info in zip(self.block_uids, self.block_infos):
            assert info is not None, f"Found no remote peers for block {uid}"
        assert self.spans_by_priority and self.spans_containing_block

    def update_(self):
        with self.lock_changes:
            self.update_block_infos_()
            self.spans_by_priority, self.spans_containing_block = self.compute_spans(self.block_infos)

    def update_block_infos_(self):
        new_block_infos: Sequence[RemoteModuleInfo] = self.dht.run_coroutine(
            partial(_get_remote_module_infos, uids=self.block_uids, expiration_time=float("inf")),
            return_future=False)
        assert len(new_block_infos) == len(self.block_uids)
        for block_index, (uid, info) in enumerate(zip(self.block_uids, new_block_infos)):
            if info is None:
                logger.warning(f"Found no block info for block {uid}")
            if not isinstance(info, RemoteModuleInfo):
                logger.warning(f"Unexpected dht entry type for {uid}: {info}")
            if not info.peer_ids:
                logger.warning(f"Found no active peers for block {uid}")
            if info.uid != uid:
                logger.warning(f"The DHT entry for {uid} actually points to {info.uid}")
            if not isinstance(info.peer_ids, set):
                logger.warning(f"Expected peer_ids for {uid} to be a set, got {type(info.peer_ids)}")
            self.block_infos[block_index] = info

    @staticmethod
    def compute_spans(block_infos: Sequence[RemoteModuleInfo]):
        closed_spans = []
        active_spans = {}
        for block_index, info in enumerate(block_infos):
            for peer_id in info.peer_ids:
                if peer_id not in active_spans:
                    active_spans[peer_id] = Span(start=block_index, end=block_index + 1, peer_id=peer_id)
                else:  # peer_id in active_spans
                    active_spans[peer_id] = active_spans[peer_id]._replace(end=block_index + 1)

            for peer_id in list(active_spans.keys()):
                if peer_id not in info.peer_ids or block_index == len(block_infos) - 1:
                    closed_spans.append(active_spans.pop(peer_id))
        assert not active_spans

        closed_spans.sort(key=lambda span: span.end - span.start, reverse=True)

        spans_containing_block = tuple(list() for _ in range(len(block_infos)))
        for span in closed_spans:
            for block_index in range(span.start, span.end):
                spans_containing_block[block_index].append(span)

        return closed_spans, spans_containing_block


class RemoteSequentialInferenceSession:
    pass
#     """An interface to a multi-step *inference* session for a sequence of remote modules"""
# 
#     def __init__(self, block):
#         self.closed = False
# 
#     @classmethod
#     async def _create(cls, remote_sequential: RemoteSequential, **kwargs) -> RemoteSequentialInferenceSession:
#         """Create a new session for a sequence of modules. This code is meant to be run inside RemoteExpertWorker"""
# 
#         remote_sequential.
#         return cls(remote_module.uid, remote_module.info, inputs_queue, outputs_stream)
# 
#     def step(self, new_hidden_states: torch.Tensor):
#         """Inference step: send a chunk of input tensors and receive a chunk of outputs"""
#         if self.closed:
#             raise Exception("Session is closed, cannot perform step")
#         # serialize inputs and put them into the queue
#         inputs = (new_hidden_states,)
#         outputs_serialized = RemoteExpertWorker.run_coroutine(
#             self._step(
#                 runtime_pb2.ExpertRequest(
#                     uid=self.uid,
#                     tensors=[
#                         serialize_torch_tensor(tensor, proto.compression)
#                         for tensor, proto in zip(inputs, nested_flatten(self.info["forward_schema"]))
#                     ],
#                 )
#             )
#         )
#         outputs = list(map(deserialize_torch_tensor, outputs_serialized.tensors))
#         assert outputs[0].shape == inputs[0].shape, f"expected outputs[0] to be hidden states but got {outputs[0]}"
#         return outputs[0]
# 
#     async def _step(self, inputs_serialized: runtime_pb2.ExpertRequest) -> runtime_pb2.ExpertResponse:
#         """Inference step on serialized data. This code is meant to be run inside RemoteExpertWorker"""
#         await self._inputs_queue.put(inputs_serialized)
#         return await anext(self._outputs_stream)
# 
#     def close(self):
#         """Finish a given inference session, close the underlying connection"""
#         if self._outputs_stream is None:
#             return  # already closed
#         RemoteExpertWorker.run_coroutine(self._aclose_stream())
#         self._outputs_stream = self._inputs_queue = None
#         self.closed = True
# 
#     async def _aclose_stream(self):
#         """Close the inference session. This code is meant to be run inside RemoteExpertWorker"""
#         if self._outputs_stream is None:
#             return  # already closed
#         await self._inputs_queue.put(runtime_pb2.ExpertRequest())  # empty request will trigger end of session
#         try:
#             await anext(self._outputs_stream)
#         except StopAsyncIteration:
#             pass
# 
#     def __del__(self):
#         self.close()
# 
#     def __enter__(self):
#         assert not self.closed
#         return self
# 
#     def __exit__(self, *exc_details):
#         self.close()
