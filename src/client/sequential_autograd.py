import asyncio
import logging
from typing import List, Optional, Sequence, Tuple

import torch
from hivemind import serialize_torch_tensor
from hivemind.moe.client.expert import expert_backward, expert_forward
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import StubBase
from hivemind.utils.nested import nested_compare, nested_flatten, nested_pack

from src.client.sequence_manager import RemoteSequenceManager
from src.data_structures import CHAIN_DELIMITER, ModuleUID, RemoteSpanInfo, RPCInfo
from src.server.handler import TransformerConnectionHandler
from src.utils.misc import DUMMY, is_dummy

MAX_TOKENS_IN_BATCH = 1024


async def run_expert_forward(
    uid: ModuleUID, stub: StubBase, rpc_info: RPCInfo, *inputs: torch.Tensor, **kwargs
) -> Tuple[torch.Tensor, ...]:
    """
    Serializes input tensors and calls "expert_forward".
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L198
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """

    # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
    # detach to avoid pickling the computation graph
    assert len(kwargs) == len(rpc_info["keyword_names"]), f"Keyword args should be {rpc_info['keyword_names']}"
    kwargs = {key: kwargs[key] for key in rpc_info["keyword_names"]}

    # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors
    forward_inputs = (inputs, kwargs)

    # Modify forward_schema to support prompts
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    # TODO: rm this assert when support arbitrary number of input tensors
    assert len(args_schema) == 1 and len(inputs) == 2
    forward_schema_with_prompts = (tuple(args_schema * len(inputs)), kwargs_schema)

    if not nested_compare(forward_inputs, forward_schema_with_prompts):
        raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")

    forward_inputs = nested_flatten(forward_inputs)
    inputs = tuple(tensor.cpu().detach() for tensor in forward_inputs)

    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(inputs, nested_flatten(forward_schema_with_prompts))
        )
    )

    deserialized_outputs = await expert_forward(uid, inputs, serialized_tensors, stub)
    flat_outputs = tuple(deserialized_outputs)
    return nested_pack(flat_outputs, structure=rpc_info["outputs_schema"])


async def run_expert_backward(
    uid: ModuleUID,
    stub: StubBase,
    rpc_info: RPCInfo,
    inputs: List[torch.Tensor],
    grad_outputs: List[torch.Tensor],
) -> Sequence[torch.Tensor]:
    """
    Serializes grad outputs and calls "expert_backward".
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L221
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """

    grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
    inputs_and_grad_outputs = tuple(nested_flatten((inputs, grad_outputs_cpu)))

    # Modify forward_schema to support prompts
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    # TODO: rm this assert when support arbitrary number of input tensors
    assert len(args_schema) == 1 and len(inputs) == 2
    forward_schema_with_prompts = (tuple(args_schema * len(inputs)), kwargs_schema)

    backward_schema = tuple(nested_flatten((forward_schema_with_prompts, rpc_info["outputs_schema"])))

    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
        )
    )

    deserialized_grad_inputs = await expert_backward(uid, inputs_and_grad_outputs, serialized_tensors, stub)
    return deserialized_grad_inputs


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

    assert isinstance(inputs, torch.Tensor) and inputs.ndim == 3

    end_index = end_index if end_index is not None else len(sequence_manager.block_uids)
    assert start_index >= 0 and end_index <= len(sequence_manager.block_uids)
    assert is_dummy(prompts) or len(prompts) == len(
        sequence_manager.block_uids
    )  # should be n_layers - 1 but add extra prompts for convenience

    sequences = sequence_manager.make_sequence(start_index, end_index)
    intermediate_inputs = []
    done_sequences = []

    while len(sequences) > 0:
        while True:
            try:
                span = sequences.pop(0)
                span_uids: str = CHAIN_DELIMITER.join(sequence_manager.block_uids[span.start : span.end])
                stub = TransformerConnectionHandler.get_stub(sequence_manager.p2p, span.peer_id)
                inputs_and_prompts = [inputs, prompts[span.start : span.end]]

                (outputs,) = await run_expert_forward(span_uids, stub, sequence_manager.rpc_info, *inputs_and_prompts)

                assert isinstance(outputs, torch.Tensor)
                assert outputs.shape == inputs.shape, f"Expected output {inputs.shape}, got {outputs.shape}"

                # Save intermediate inputs and subsequences if the forward is already done for them
                intermediate_inputs.append(inputs)
                done_sequences.append(span)

                inputs = outputs
                break
            except Exception as e:
                logging.warn(f"Caught {e} when running forward for chain {span.start}-{span.end}", exc_info=True)
                backup_sequences = sequence_manager.make_sequence(span.start)
                assert backup_sequences[0].start == span.start
                sequences = backup_sequences

    return outputs, intermediate_inputs, done_sequences


async def sequential_backward(
    grad_outputs: Sequence[torch.Tensor],
    intermediate_inputs: Sequence[torch.Tensor],
    prompts: Sequence[torch.Tensor],
    forward_sequences: Sequence[RemoteSpanInfo],
    sequence_manager: RemoteSequenceManager,
) -> Sequence[torch.Tensor]:
    """
    Performs chained backward for each forward subsequence.
    If some subsequence fails, reconstructs the particular sub-path and recovers the backward.
    """
    assert len(intermediate_inputs) == len(forward_sequences)

    grad_prompts = []
    while len(forward_sequences) > 0 and len(intermediate_inputs) > 0:
        while True:
            try:
                inputs = intermediate_inputs.pop(-1)
                span = forward_sequences.pop(-1)

                span_uids: str = CHAIN_DELIMITER.join(sequence_manager.block_uids[span.start : span.end])
                stub = TransformerConnectionHandler.get_stub(sequence_manager.p2p, span.peer_id)

                inputs_and_prompts = [inputs, prompts[span.start : span.end]]
                grad_outputs, span_grad_prompts = await run_expert_backward(
                    span_uids, stub, sequence_manager.rpc_info, inputs_and_prompts, grad_outputs
                )
                grad_outputs = [grad_outputs]
                grad_prompts.append(span_grad_prompts)
                break
            except Exception as e:
                logging.warning(f"Caught {e} when running backward for chain {span.start}-{span.end}", exc_info=True)
                _, backup_intermediate_inputs, backup_forward_sequences = await sequential_forward(
                    inputs, prompts, sequence_manager, start_index=span.start, end_index=span.end
                )
                assert len(intermediate_inputs) == len(forward_sequences)
                assert backup_forward_sequences[0].start == span.start
                assert backup_forward_sequences[-1].end == span.end

                forward_sequences.extend(backup_forward_sequences)
                intermediate_inputs.extend(backup_intermediate_inputs)

    dummy_grad_prompts = [is_dummy(grad_prompt) for grad_prompt in grad_prompts]
    # For now, we do not support mixed dummy and grad prompts
    # Concat in num_layer dimension
    grad_prompts = torch.cat(grad_prompts, dim=0) if not any(dummy_grad_prompts) else None
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
        if is_dummy(prompts):
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
