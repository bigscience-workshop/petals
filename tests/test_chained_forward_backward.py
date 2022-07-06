######
# Warning:torch this test is a work in progress. It will be modified soon.
# - if you want more stable tests, see test_block_exact_match
# - if you want to figure out chained inference, ask yozh

import os

import hivemind
import torch
from hivemind.moe.expert_uid import ExpertInfo

from src.bloom.from_pretrained import load_pretrained_block
from src.client.remote_block import RemoteTransformerBlock
from src.dht_utils import get_remote_module

INITIAL_PEERS = os.environ.get("INITIAL_PEERS")
if not INITIAL_PEERS:
    raise RuntimeError("Must specify INITIAL_PEERS environment variable with one or more peer ids")
INITIAL_PEERS = INITIAL_PEERS.split()


BLOCK_UID = os.environ.get("BLOCK_UID")
if not BLOCK_UID:
    raise RuntimeError("Must specify BLOCK_UID as an index of a transformer block to be tested")

REF_NAME = os.environ.get("REF_NAME", "bigscience/test-bloomd-6b3")


# seq_length > 128: rpc_forward_stream & rpc_backward_stream
# seq_length <= 128: rpc_forward & rpc_backward
def test_forward_backward_exact_match(atol_forward=1e-4, atol_backward=1e-4, seq_length=1):
    dht = hivemind.DHT(initial_peers=INITIAL_PEERS, client_mode=True, start=True)
    (remote_block,) = get_remote_module(dht, BLOCK_UID)
    assert remote_block is not None, f"Could not find {BLOCK_UID} in DHT"
    assert isinstance(remote_block, RemoteTransformerBlock)

    _ = remote_block.info  # lazy-init info now, because otherwise we will _break_ info init by chaning _info
    remote_block._info = ExpertInfo("bloom6b3.3 bloom6b3.4 bloom6b3.5", remote_block._info.peer_id)

    ref_blocks = [
        load_pretrained_block(REF_NAME, 3, torch_dtype=torch.float32),
        load_pretrained_block(REF_NAME, 4, torch_dtype=torch.float32),
        load_pretrained_block(REF_NAME, 5, torch_dtype=torch.float32),
    ]
    inputs = torch.randn(1, seq_length, 4096, requires_grad=True)
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
