# Note: this code is being actively modified by justheuristic. If you want to change anything about it, please warn me.
import os
import random

import hivemind
import pytest
import torch
import transformers

from src.bloom.from_pretrained import load_pretrained_block
from src.client.remote_block import RemoteTransformerBlock
from src.data_structures import UID_DELIMITER
from src.dht_utils import get_remote_module

INITIAL_PEERS = os.environ.get("INITIAL_PEERS")
if not INITIAL_PEERS:
    raise RuntimeError("Must specify INITIAL_PEERS environment variable with one or more peer ids")
INITIAL_PEERS = INITIAL_PEERS.split()

MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    raise RuntimeError("Must specify MODEL_NAME as a name of a model to be tested")


def test_remote_block_exact_match(atol_forward=1e-5, atol_inference=1e-3):
    dht = hivemind.DHT(initial_peers=INITIAL_PEERS, client_mode=True, start=True)
    config = transformers.AutoConfig.from_pretrained(MODEL_NAME)

    for block_index in random.sample(range(config.n_layer), 3):
        block_uid = f"{MODEL_NAME}{UID_DELIMITER}{block_index}"
        remote_block = get_remote_module(dht, block_uid)
        assert remote_block is not None, f"Could not find {block_uid} in DHT"
        assert isinstance(remote_block, RemoteTransformerBlock)

        inputs = torch.randn(1, 8, config.hidden_size)
        (outputs_forward,) = remote_block(inputs)

        outputs_inference = []
        with remote_block.inference_session() as sess:
            for i in range(inputs.shape[1]):
                outputs_inference.append(sess.step(inputs[:, i : i + 1, :]))
        outputs_inference = torch.cat(outputs_inference, dim=1)

        ref_block = load_pretrained_block(MODEL_NAME, block_index, torch_dtype=torch.float32)
        (outputs_local,) = ref_block(inputs)

        assert torch.allclose(outputs_local, outputs_forward, rtol=0, atol=atol_forward)
        assert torch.allclose(outputs_local, outputs_inference, rtol=0, atol=atol_inference)
