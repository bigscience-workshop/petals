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
REF_INDEX = int(os.environ.get("REF_INDEX", BLOCK_UID[-1].split(".")[-1]))


def test_remote_block_exact_match(atol_inference=1e-4):
    dht = hivemind.DHT(initial_peers=INITIAL_PEERS, client_mode=True, start=True)
    remote_block = get_remote_module(dht, BLOCK_UID)
    assert remote_block is not None, f"Could not find {BLOCK_UID} in DHT"
    assert isinstance(remote_block, RemoteTransformerBlock)

    _ = remote_block.info  # lazy-init info now, because otherwise we will _break_ info init by chaning _info
    remote_block._info = ExpertInfo("bloom6b3.3 bloom6b3.4", remote_block._info.peer_id)

    inputs = torch.randn(1, 8, 4096)

    outputs_inference = []
    with remote_block.begin_inference_session() as sess:
        for i in range(inputs.shape[1]):
            outputs_inference.append(sess.step(inputs[:, i : i + 1, :]))
    outputs_inference = torch.cat(outputs_inference, dim=1)

    ref_blocks = [
        load_pretrained_block(REF_NAME, 3, torch_dtype=torch.float32),
        load_pretrained_block(REF_NAME, 4, torch_dtype=torch.float32),
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
