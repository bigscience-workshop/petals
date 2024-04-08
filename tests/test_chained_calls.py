######
# Warning:torch this test is a work in progress. It will be modified soon.
# - if you want more stable tests, see test_block_exact_match
# - if you want to figure out chained inference, ask yozh


import pytest
import torch

from petals import AutoDistributedConfig
from petals.client.remote_sequential import RemoteSequential
from petals.server.from_pretrained import load_pretrained_block
from petals.utils.misc import DUMMY_KEY_PAST
from test_utils import *


@pytest.mark.forked
def test_forward_backward_exact_match(atol_forward=1e-4, atol_backward=1e-4, seq_length=1):
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    remote_blocks = RemoteSequential(config, start_block=3, end_block=6)
    assert isinstance(remote_blocks, RemoteSequential)

    ref_blocks = [
        load_pretrained_block(MODEL_NAME, 3, torch_dtype=torch.float32),
        load_pretrained_block(MODEL_NAME, 4, torch_dtype=torch.float32),
        load_pretrained_block(MODEL_NAME, 5, torch_dtype=torch.float32),
    ]
    inputs = torch.randn(1, seq_length, config.hidden_size, requires_grad=True)
    outputs_rpc = remote_blocks.forward(inputs)
    outputs_rpc.sum().backward()
    grads_rpc = inputs.grad

    inputs.grad = None
    hidden_states = inputs
    for ref_block in ref_blocks:
        hidden_states = ref_block.forward(hidden_states)[0]
    outputs_ref = hidden_states
    outputs_ref.sum().backward()
    grads_ref = inputs.grad

    assert torch.allclose(outputs_ref, outputs_rpc, rtol=0, atol=atol_forward)
    assert torch.allclose(grads_ref, grads_rpc, rtol=0, atol=atol_backward)


@pytest.mark.forked
def test_chained_inference_exact_match(atol_inference=1e-4):
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    remote_blocks = RemoteSequential(config, start_block=3, end_block=5)

    inputs = torch.randn(1, 8, config.hidden_size)

    outputs_inference = []
    with remote_blocks.inference_session(max_length=inputs.shape[1]) as sess:
        for i in range(inputs.shape[1]):
            outputs_inference.append(sess.step(inputs[:, i : i + 1, :]))
    outputs_inference = torch.cat(outputs_inference, dim=1)

    dtype = torch.float32
    ref_blocks = [
        load_pretrained_block(MODEL_NAME, 3, torch_dtype=dtype),
        load_pretrained_block(MODEL_NAME, 4, torch_dtype=dtype),
    ]
    outputs_ref = []
    cache = (DUMMY_KEY_PAST.to(dtype), DUMMY_KEY_PAST.to(dtype))
    caches = [cache, cache]
    for i in range(inputs.shape[1]):
        new_caches = []
        hidden_states = inputs[:, i : i + 1, :]
        for ref_block, cache in zip(ref_blocks, caches):
            with torch.no_grad():
                hidden_states, new_cache = ref_block.forward(hidden_states, use_cache=True, layer_past=cache)
                new_caches.append(new_cache)

        outputs_ref.append(hidden_states)
        caches = new_caches
    outputs_ref = torch.cat(outputs_ref, dim=1)
    assert torch.allclose(outputs_ref, outputs_inference, rtol=0, atol=atol_inference)
