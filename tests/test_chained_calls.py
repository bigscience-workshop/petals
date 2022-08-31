######
# Warning:torch this test is a work in progress. It will be modified soon.
# - if you want more stable tests, see test_block_exact_match
# - if you want to figure out chained inference, ask yozh


import hivemind
import pytest
import torch
import transformers
from hivemind.moe.expert_uid import UID_DELIMITER, ExpertInfo
from test_utils import *

import src
from src.bloom.from_pretrained import load_pretrained_block
from src.client.remote_sequential import RemoteTransformerBlock
from src.data_structures import UID_DELIMITER
from src.dht_utils import get_remote_module


@pytest.mark.forked
def test_forward_backward_exact_match(atol_forward=1e-4, atol_backward=1e-4, seq_length=1):
    dht = hivemind.DHT(initial_peers=INITIAL_PEERS, client_mode=True, start=True)
    config = src.DistributedBloomConfig.from_pretrained(MODEL_NAME)
    remote_block = get_remote_module(dht, f"{MODEL_NAME}{UID_DELIMITER}0", config)
    assert remote_block is not None, f"Could not find {MODEL_NAME}{UID_DELIMITER}0 in DHT"
    assert isinstance(remote_block, RemoteTransformerBlock)

    ref_blocks = [
        load_pretrained_block(MODEL_NAME, 3, torch_dtype=torch.float32),
        load_pretrained_block(MODEL_NAME, 4, torch_dtype=torch.float32),
        load_pretrained_block(MODEL_NAME, 5, torch_dtype=torch.float32),
    ]
    inputs = torch.randn(1, seq_length, config.hidden_size, requires_grad=True)
    outputs_rpc = remote_block.forward(inputs)[0]
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
    dht = hivemind.DHT(initial_peers=INITIAL_PEERS, client_mode=True, start=True)
    config = src.DistributedBloomConfig.from_pretrained(MODEL_NAME)
    remote_block = get_remote_module(dht, f"{MODEL_NAME}{UID_DELIMITER}0", config)
    assert remote_block is not None, f"Could not find {MODEL_NAME}{UID_DELIMITER}0 in DHT"
    assert isinstance(remote_block, RemoteTransformerBlock)

    inputs = torch.randn(1, 8, config.hidden_size)

    outputs_inference = []
    with remote_block.inference_session(max_length=inputs.shape[1]) as sess:
        for i in range(inputs.shape[1]):
            outputs_inference.append(sess.step(inputs[:, i : i + 1, :]))
    outputs_inference = torch.cat(outputs_inference, dim=1)

    ref_blocks = [
        load_pretrained_block(MODEL_NAME, 3, torch_dtype=torch.float32),
        load_pretrained_block(MODEL_NAME, 4, torch_dtype=torch.float32),
    ]
    outputs_ref = []
    caches = [None, None]
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
