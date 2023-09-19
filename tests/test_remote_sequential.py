import pytest
import torch
import torch.nn.functional as F
from hivemind import DHT, BatchTensorDescriptor, get_logger
from hivemind.proto import runtime_pb2

from petals import AutoDistributedConfig
from petals.client import RemoteSequenceManager, RemoteSequential
from petals.data_structures import UID_DELIMITER
from petals.server.from_pretrained import load_pretrained_block
from test_utils import *

logger = get_logger(__name__)


@pytest.mark.forked
def test_remote_sequential():
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    dht = DHT(initial_peers=config.initial_peers, client_mode=True, start=True)
    test_inputs = torch.randn(1, 5, config.hidden_size, requires_grad=True)
    grad_proj = torch.randn(1, 5, config.hidden_size)

    sequential = RemoteSequential(config, dht=dht)

    full_outputs = sequential(test_inputs)
    (full_outputs * grad_proj).sum().backward()
    assert test_inputs.grad is not None
    full_grad = test_inputs.grad.clone()
    test_inputs.grad.data.zero_()

    first_half = sequential[: config.num_hidden_layers // 2]
    second_half = sequential[config.num_hidden_layers // 2 :]
    assert len(first_half) + len(second_half) == len(sequential)
    assert abs(len(first_half) - len(second_half)) == config.num_hidden_layers % 2
    for m in sequential, first_half, second_half:
        assert isinstance(repr(m), str)

    hidden = first_half(test_inputs)
    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape == test_inputs.shape
    assert hidden.requires_grad
    second_half_outputs = second_half(hidden)
    assert torch.allclose(second_half_outputs, full_outputs, atol=1e-3)

    (second_half_outputs * grad_proj).sum().backward()
    assert torch.allclose(test_inputs.grad, full_grad, atol=3e-2)

    # test RemoteSequential with lossy compression
    block_uids = [f"{config.dht_prefix}{UID_DELIMITER}{i}" for i in range(config.num_hidden_layers)]
    lossy_sequential = RemoteSequential(
        config, sequence_manager=DummyCustomSequenceManager(config, block_uids, dht=dht)
    )

    test_inputs.grad = None
    approx_outputs = lossy_sequential(test_inputs)
    (approx_outputs * grad_proj).sum().backward()

    assert not torch.allclose(approx_outputs, full_outputs, rtol=0, atol=1e-4), "compression was not used"
    assert not torch.allclose(test_inputs.grad, full_grad, rtol=0, atol=1e-3), "compression was not used"
    assert abs(approx_outputs - full_outputs).mean() < 0.01
    absmax = abs(full_grad).max()
    assert abs(test_inputs.grad / absmax - full_grad / absmax).mean() < 0.05


class DummyCustomSequenceManager(RemoteSequenceManager):
    """A sequence manager that compresses inputs/outputs during forward and backward pass."""

    @property
    def rpc_info(self):
        rpc_info = super().rpc_info
        dims = (2048, 1024)
        compressed_input_schema = BatchTensorDescriptor(dims, compression=runtime_pb2.CompressionType.FLOAT16)
        rpc_info["forward_schema"] = (compressed_input_schema,), dict()  # (args, kwargs)
        return rpc_info

    def get_request_metadata(self, protocol: str, *args, **kwargs):
        metadata = super().get_request_metadata(protocol, *args, **kwargs)
        if protocol == "rpc_forward":
            metadata["output_compression"] = (runtime_pb2.CompressionType.FLOAT16,)
        elif protocol == "rpc_backward":
            metadata["output_compression"] = (runtime_pb2.CompressionType.FLOAT16,)
            # FIXME: Initially, we used CompressionType.BLOCKWISE_8BIT for rpc_backward() here.
            # This is currently broken since hivemind==1.1.8 is not compatible with bitsandbytes==0.39.1.
            # Please revert to BLOCKWISE_8BIT once this is fixed: https://github.com/learning-at-home/hivemind/issues/572
        return metadata


@pytest.mark.forked
def test_remote_sequential_prompts(batch_size=2, seq_len=5, pre_seq_len=3):
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    remote_sequential = RemoteSequential(config)

    inputs = F.normalize(torch.randn(batch_size, seq_len, config.hidden_size), dim=-1)
    output_proj = F.normalize(torch.randn(batch_size, seq_len + pre_seq_len, config.hidden_size), dim=-1)
    input_prompts = F.normalize(torch.randn(batch_size, pre_seq_len, config.hidden_size, requires_grad=True), dim=-1)
    intermediate_prompts = torch.randn(
        config.num_hidden_layers, batch_size, pre_seq_len, config.hidden_size, requires_grad=True
    )

    input_prompts = input_prompts.detach().requires_grad_(True)
    intermediate_prompts = intermediate_prompts.detach().requires_grad_(True)

    inputs_with_prompts = torch.cat([inputs, input_prompts], dim=1)
    assert inputs_with_prompts.shape == (batch_size, seq_len + pre_seq_len, config.hidden_size)

    outputs = remote_sequential(inputs_with_prompts, prompts=intermediate_prompts)

    (outputs * output_proj).sum().backward()
    assert intermediate_prompts.grad is not None

    input_prompts_ref = input_prompts.clone().detach().requires_grad_(True)
    intermediate_prompts_ref = intermediate_prompts.clone().detach().requires_grad_(True)

    assert input_prompts_ref.grad is None
    assert intermediate_prompts_ref.grad is None

    outputs_ref = torch.cat([inputs, input_prompts_ref], dim=1)
    for block_index in range(config.num_hidden_layers):
        block_prompt = intermediate_prompts_ref[block_index]
        outputs_ref[:, : block_prompt.shape[1]] += block_prompt

        block = load_pretrained_block(MODEL_NAME, block_index=block_index, torch_dtype=torch.float32)
        (outputs_ref,) = block(outputs_ref)

    assert torch.allclose(outputs_ref, outputs, atol=1e-3)

    (outputs_ref * output_proj).sum().backward()
    assert input_prompts_ref.grad is not None
    assert torch.allclose(input_prompts_ref.grad, input_prompts.grad, atol=3e-2)
    assert intermediate_prompts_ref.grad is not None
    assert torch.allclose(intermediate_prompts_ref.grad, intermediate_prompts.grad, atol=1e-2)
