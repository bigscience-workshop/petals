"""
A PyTorch autograd function that runs forward/backward on a sequence of remote servers in a fault-tolerant manner
"""
import asyncio
import itertools
from collections import deque
from typing import List, Optional, Sequence, Tuple

import torch
from hivemind import MSGPackSerializer
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.utils.logging import get_logger

from petals.client.remote_forward_backward import run_remote_backward, run_remote_forward
from petals.client.routing import RemoteSequenceManager, maybe_log_traceback
from petals.data_structures import CHAIN_DELIMITER, RemoteSpanInfo
from petals.server.handler import TransformerConnectionHandler
from petals.utils.misc import DUMMY, is_dummy
from petals.utils.packaging import pack_args_kwargs

logger = get_logger(__name__)

MAX_TOKENS_IN_BATCH = 1024


async def sequential_forward(
    inputs: torch.Tensor,
    prompts: torch.Tensor,
    sequence_manager: RemoteSequenceManager,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> Tuple[torch.Tensor, Sequence[torch.Tensor], Sequence[RemoteSpanInfo]]:
    """
    Constructs a routing path from <start_index> to <end_index>.
    Performs chained forward for each subsequence of blocks on the path.
    If some subsequence fails, reconstructs the remaining path and tries to finish the forward.
    """

    assert isinstance(inputs, torch.Tensor) and inputs.ndim == 3, f"{type(inputs)}: {inputs.ndim}"

    inputs_device = inputs.device
    inputs_dtype = inputs.dtype
    inputs = inputs.cpu()
    prompts = prompts.cpu()

    end_index = end_index if end_index is not None else len(sequence_manager.block_uids)
    assert start_index >= 0 and end_index <= len(sequence_manager.block_uids)
    assert is_dummy(prompts) or len(prompts) == len(
        sequence_manager.block_uids
    )  # should be n_layers - 1 but add extra prompts for convenience

    sequences = deque()
    intermediate_inputs = []
    done_sequences = []

    block_idx = start_index
    while block_idx < end_index:
        for attempt_no in itertools.count():
            logger.debug(f"Forward: block {block_idx}, attempt {attempt_no}")
            span = None
            try:
                if not sequences or attempt_no >= 1:
                    sequences = deque(sequence_manager.make_sequence(block_idx, end_index, mode="max_throughput"))
                    # make_sequence() could return a longer sequence
                    sequences[-1].end = min(sequences[-1].end, end_index)
                    logger.debug(f"Found path from block {block_idx} to {end_index} via {len(sequences)} servers")

                span = sequences.popleft()

                stub = TransformerConnectionHandler.get_stub(sequence_manager.state.p2p, span.peer_id)
                flat_tensors, args_structure = pack_args_kwargs(inputs, prompts[span.start : span.end])

                span_uids = CHAIN_DELIMITER.join(sequence_manager.block_uids[span.start : span.end])
                metadata = sequence_manager.get_request_metadata(
                    "rpc_forward", args_structure, span_uids, *flat_tensors
                )
                (outputs,) = await run_remote_forward(
                    span_uids,
                    stub,
                    sequence_manager.rpc_info,
                    *flat_tensors,
                    config=sequence_manager.config,
                    metadata=MSGPackSerializer.dumps(metadata),
                )

                assert isinstance(outputs, torch.Tensor)
                assert outputs.shape == inputs.shape, f"Expected output {inputs.shape}, got {outputs.shape}"

                # Save intermediate inputs and subsequences if the forward is already done for them
                intermediate_inputs.append(inputs)
                done_sequences.append(span)

                inputs = outputs
                block_idx = span.end
                sequence_manager.on_request_success(span.peer_id)
                break
            except Exception as e:
                sequence_manager.on_request_failure(span.peer_id if span is not None else None)
                if attempt_no + 1 == sequence_manager.config.max_retries:
                    raise
                delay = sequence_manager.get_retry_delay(attempt_no)
                logger.warning(
                    f"Caught exception when running forward via {span} (retry in {delay:.0f} sec): {repr(e)}"
                )
                maybe_log_traceback(e)
                await asyncio.sleep(delay)

    outputs = inputs.to(device=inputs_device, dtype=inputs_dtype)
    intermediate_inputs = [tensor.to(device=inputs_device, dtype=inputs_dtype) for tensor in intermediate_inputs]
    return outputs, intermediate_inputs, done_sequences


async def sequential_backward(
    grad_outputs: Sequence[torch.Tensor],
    intermediate_inputs: List[torch.Tensor],
    prompts: torch.Tensor,
    forward_sequences: List[RemoteSpanInfo],
    sequence_manager: RemoteSequenceManager,
) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:
    """
    Performs chained backward for each forward subsequence.
    If some subsequence fails, reconstructs the particular sub-path and recovers the backward.
    """
    assert len(intermediate_inputs) == len(forward_sequences)

    grad_outputs_device = grad_outputs[0].device if grad_outputs else None
    grad_outputs_dtype = grad_outputs[0].dtype if grad_outputs else None
    prompts_device = prompts.device
    prompts_dtype = prompts.dtype

    grad_outputs = [tensor.cpu() for tensor in grad_outputs]
    intermediate_inputs = [tensor.cpu() for tensor in intermediate_inputs]
    prompts = prompts.cpu()

    grad_prompts_reversed = []
    while len(forward_sequences) > 0 and len(intermediate_inputs) > 0:
        inputs = intermediate_inputs.pop()
        span = forward_sequences.pop()
        for attempt_no in itertools.count():
            logger.debug(f"Backward: block {span.end - 1}, attempt {attempt_no}")
            try:
                if attempt_no >= 1:
                    _, backup_inputs, backup_sequences = await sequential_forward(
                        inputs, prompts, sequence_manager, start_index=span.start, end_index=span.end
                    )
                    assert len(backup_inputs) == len(backup_sequences)
                    assert backup_sequences[0].start == span.start
                    assert backup_sequences[-1].end == span.end

                    intermediate_inputs.extend(backup_inputs)
                    forward_sequences.extend(backup_sequences)
                    inputs = intermediate_inputs.pop()
                    span = forward_sequences.pop()

                grad_outputs_cpu = [grad.cpu() for grad in grad_outputs]
                flat_tensors, args_structure = pack_args_kwargs(
                    inputs, *grad_outputs_cpu, prompts[span.start : span.end]
                )

                span_uids = CHAIN_DELIMITER.join(sequence_manager.block_uids[span.start : span.end])
                stub = TransformerConnectionHandler.get_stub(sequence_manager.state.p2p, span.peer_id)
                metadata = sequence_manager.get_request_metadata(
                    "rpc_backward", args_structure, span_uids, *flat_tensors, peer_id=span.peer_id
                )
                grad_outputs, *span_grad_prompts = await run_remote_backward(
                    span_uids,
                    stub,
                    sequence_manager.rpc_info,
                    *flat_tensors,
                    config=sequence_manager.config,
                    metadata=MSGPackSerializer.dumps(metadata),
                )
                grad_outputs = [grad_outputs]
                grad_prompts_reversed.extend(span_grad_prompts)
                sequence_manager.on_request_success(span.peer_id)
                break
            except Exception as e:
                sequence_manager.on_request_failure(span.peer_id if span is not None else None)
                if attempt_no + 1 == sequence_manager.config.max_retries:
                    raise
                delay = sequence_manager.get_retry_delay(attempt_no)
                logger.warning(
                    f"Caught exception when running backward via {span} (retry in {delay:.0f} sec): {repr(e)}"
                )
                maybe_log_traceback(e)
                await asyncio.sleep(delay)

    # For now, we do not support mixed dummy and grad prompts
    # Concat in num_layer dimension
    grad_prompts = torch.cat(grad_prompts_reversed[::-1], dim=0) if grad_prompts_reversed else None

    if grad_outputs_dtype is not None:
        grad_outputs = [tensor.to(device=grad_outputs_device, dtype=grad_outputs_dtype) for tensor in grad_outputs]
    if grad_prompts is not None:
        grad_prompts = grad_prompts.to(device=prompts_device, dtype=prompts_dtype)
    return grad_outputs, grad_prompts


async def _gather_forward(input_batches, prompt_batches, sequence_manager):
    """Wrapper for asyncio.gather to perform parallel sequential forwards"""
    return await asyncio.gather(
        *[
            sequential_forward(input_batch, prompt_batch, sequence_manager)
            for input_batch, prompt_batch in zip(input_batches, prompt_batches)
        ]
    )


async def _gather_backward(
    grad_output_batches, intermediate_input_batches, prompt_batches, forward_sequences, sequence_manager
):
    """Wrapper for asyncio.gather to perform parallel sequential backwards"""
    return await asyncio.gather(
        *[
            sequential_backward((grad_output,), input_batch, prompt_batch, spans, sequence_manager)
            for grad_output, input_batch, prompt_batch, spans in zip(
                grad_output_batches, intermediate_input_batches, prompt_batches, forward_sequences
            )
        ]
    )


class _RemoteSequentialAutogradFunction(torch.autograd.Function):
    """
    PyTorch autograd function that provides forward and backward calls for the entire sequence of remote transformer blocks.
    This function splits input data into batches with <MAX_TOKENS_IN_BATCH> and performs efficient parallel processing.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, prompts: torch.Tensor, sequence_manager: RemoteSequenceManager):
        batch_size = max(MAX_TOKENS_IN_BATCH // inputs.shape[1], 1)
        input_batches: Sequence[torch.Tensor] = inputs.detach().split(batch_size)
        if prompts is None or is_dummy(prompts):
            prompt_batches = [DUMMY] * len(input_batches)
        else:
            prompt_batches: Sequence[torch.Tensor] = prompts.detach().split(batch_size, dim=1)

        sequence_manager.rpc_info  # lazy init
        outputs = RemoteExpertWorker.run_coroutine(_gather_forward(input_batches, prompt_batches, sequence_manager))
        assert len(outputs) == len(input_batches)

        output_batches = [output[0] for output in outputs]
        intemediate_input_batches = [output[1] for output in outputs]
        sequences_for_batches = [output[2] for output in outputs]

        ctx.prompt_batches = prompt_batches
        ctx.sequence_manager = sequence_manager
        ctx.intemediate_input_batches = intemediate_input_batches
        ctx.sequences_for_batches = sequences_for_batches
        return torch.cat(output_batches, dim=0)

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        intermediate_input_batches: List[Sequence[torch.Tensor]] = ctx.intemediate_input_batches
        forward_sequences: List[Sequence[RemoteSpanInfo]] = ctx.sequences_for_batches
        ctx.sequence_manager.rpc_info  # lazy init

        batch_size = max(MAX_TOKENS_IN_BATCH // grad_outputs.shape[1], 1)
        grad_output_batches: Sequence[torch.Tensor] = grad_outputs.split(batch_size)
        assert len(intermediate_input_batches) == len(grad_output_batches) == len(forward_sequences)

        outputs = RemoteExpertWorker.run_coroutine(
            _gather_backward(
                grad_output_batches,
                intermediate_input_batches,
                ctx.prompt_batches,
                forward_sequences,
                ctx.sequence_manager,
            )
        )
        grad_input_batches = [output[0][0] for output in outputs]
        grad_prompt_batches = [output[1] for output in outputs]

        grad_inputs = torch.cat(grad_input_batches, dim=0)
        dummy_grad_prompts = [grad_prompt is None for grad_prompt in grad_prompt_batches]
        grad_prompts = torch.cat(grad_prompt_batches, dim=1) if not any(dummy_grad_prompts) else None
        return (grad_inputs, grad_prompts, None)
