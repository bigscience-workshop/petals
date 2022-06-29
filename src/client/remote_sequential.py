from __future__ import annotations

import logging
import random

import torch
from hivemind import DHT, get_logger, use_hivemind_log_handler
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from torch import nn

from src import DistributedBloomConfig
from src.client.remote_sequence_info import RemoteSequenceInfo
from src.data_structures import UID_DELIMITER
from src.dht_utils import _create_remote_modules_from_infos


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
        self.remote_model_info = RemoteSequenceInfo(dht, self.block_uids)

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
        return RemoteSequentialInferenceSession(self.remote_model_info)




class RemoteSequentialInferenceSession:
    """An interface to a multi-step *inference* session for a sequence of remote transformer blocks"""

    def __init__(self, remote_sequence_info: RemoteSequenceInfo):
        self.remote_sequence_info = remote_sequence_info
        self.closed = False

        # TODO(yozh) replace this code with a fault-tolerant chain that can be reconstructed if some peers fail
        current_final_block = 0
        self.active_chain = []

        while current_final_block != len(remote_sequence_info):
            candidate_spans = remote_sequence_info.spans_containing_block[current_final_block]
            chosen_span = random.choice(candidate_spans)  # TODO this is a temporary code
            assert chosen_span.start <= current_final_block < chosen_span.end

            self.active_chain.append((current_final_block, chosen_span.end, chosen_span))
            current_final_block = chosen_span.end



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
