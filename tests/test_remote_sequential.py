import pytest
import torch
from hivemind import DHT, get_logger, use_hivemind_log_handler
from test_utils import *

from src import RemoteSequential
from src.bloom.from_pretrained import load_pretrained_block
from src.client.remote_model import DistributedBloomConfig

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


@pytest.mark.forked
def test_remote_sequential():
    config = DistributedBloomConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    dht = DHT(initial_peers=config.initial_peers, client_mode=True, start=True)
    test_inputs = torch.randn(1, 5, config.hidden_size, requires_grad=True)
    grad_proj = torch.randn(1, 5, config.hidden_size)

    sequential = RemoteSequential(config, dht)

    full_outputs = sequential(test_inputs)
    (full_outputs * grad_proj).sum().backward()
    assert test_inputs.grad is not None
    full_grad = test_inputs.grad.clone()
    test_inputs.grad.data.zero_()

    first_half = sequential[: config.n_layer // 2]
    second_half = sequential[config.n_layer // 2 :]
    assert len(first_half) + len(second_half) == len(sequential)
    assert abs(len(first_half) - len(second_half)) == config.n_layer % 2
    for m in sequential, first_half, second_half:
        assert isinstance(repr(m), str)

    hidden = first_half(test_inputs)
    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape == test_inputs.shape
    assert hidden.requires_grad
    second_half_outputs = second_half(hidden)
    assert torch.allclose(second_half_outputs, full_outputs)

    (second_half_outputs * grad_proj).sum().backward()
    assert torch.allclose(test_inputs.grad, full_grad)


@pytest.mark.forked
def test_remote_sequential_prompts(batch_size=2, seq_len=5, pre_seq_len=3):
    config = DistributedBloomConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    dht = DHT(initial_peers=config.initial_peers, client_mode=True, start=True)
    remote_sequential = RemoteSequential(config, dht)

    inputs = torch.randn(batch_size, seq_len, config.hidden_size)
    output_proj = torch.randn(batch_size, seq_len + pre_seq_len, config.hidden_size)
    input_prompts = torch.randn(batch_size, pre_seq_len, config.hidden_size, requires_grad=True)
    intermediate_prompts = torch.randn(config.n_layer, batch_size, pre_seq_len, config.hidden_size, requires_grad=True)

    input_prompts = input_prompts.detach().requires_grad_(True)
    intermediate_prompts = intermediate_prompts.detach().requires_grad_(True)
    with torch.no_grad():
        intermediate_prompts[...] = torch.randn_like(intermediate_prompts)

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
    for block_index in range(config.n_layer):
        block_prompt = intermediate_prompts_ref[block_index]
        outputs_ref[:, : block_prompt.shape[1]] += block_prompt

        block = load_pretrained_block(MODEL_NAME, block_index=block_index, torch_dtype=torch.float32)
        (outputs_ref,) = block(outputs_ref)

    assert torch.allclose(outputs_ref, outputs)

    (outputs_ref * output_proj).sum().backward()
    assert input_prompts_ref.grad is not None
    assert torch.allclose(input_prompts_ref.grad, input_prompts.grad)
    assert intermediate_prompts_ref.grad is not None
    assert torch.allclose(intermediate_prompts_ref.grad, intermediate_prompts.grad)
