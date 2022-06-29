# Note: this code is being actively modified by justheuristic. If you want to change anything about it, please warn me.
import os

import hivemind
import torch

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


def test_remote_block_exact_match(atol_forward=1e-5, atol_inference=1e-3):
    dht = hivemind.DHT(initial_peers=INITIAL_PEERS, client_mode=True, start=True)
    remote_block = get_remote_module(dht, BLOCK_UID)
    assert remote_block is not None, f"Could not find {BLOCK_UID} in DHT"
    assert isinstance(remote_block, RemoteTransformerBlock)

    inputs = torch.randn(1, 8, 4096)
    (outputs_forward,) = remote_block(inputs)

    outputs_inference = []
    with remote_block.inference_session() as sess:
        for i in range(inputs.shape[1]):
            outputs_inference.append(sess.step(inputs[:, i : i + 1, :]))
    outputs_inference = torch.cat(outputs_inference, dim=1)

    ref_block = load_pretrained_block(REF_NAME, REF_INDEX, torch_dtype=torch.float32)
    (outputs_local,) = ref_block(inputs)

    assert torch.allclose(outputs_local, outputs_forward, rtol=0, atol=atol_forward)
    assert torch.allclose(outputs_local, outputs_inference, rtol=0, atol=atol_inference)
