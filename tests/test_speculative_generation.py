import random

import pytest
import torch

from petals import AutoDistributedConfig, RemoteSequential
from petals.server.block_functions import MAX_SHORT_INFERENCE_TOKENS
from petals.server.from_pretrained import load_pretrained_block
from test_utils import *


@pytest.mark.forked
def test_remote_block_with_cache_invalidation_exact_match(atol_forward=1e-4, atol_inference=1e-3):
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    remote_sequential = RemoteSequential(config)

    block_index = random.randint(0, config.num_hidden_layers - 1)
    remote_block = remote_sequential[block_index]

    inputs = torch.randn(1, MAX_SHORT_INFERENCE_TOKENS - 50, config.hidden_size)
    short_inputs = torch.randn(1, MAX_SHORT_INFERENCE_TOKENS - 50, config.hidden_size)
    short_inputs[:, :2, :] = inputs[:, :2, :]

    initial_outputs_inference = None
    secondary_outputs_inference = None
    with torch.inference_mode():
        with remote_block.inference_session(max_length=inputs.shape[1]) as sess:
            initial_outputs_inference = sess.step(inputs)
            secondary_outputs_inference = sess.step(short_inputs[:, 2:, :], last_validated_position=2)
            result = torch.cat([initial_outputs_inference[:, :2, :], secondary_outputs_inference], dim=1)

    ref_block = load_pretrained_block(MODEL_NAME, block_index, torch_dtype=torch.float32)
    (outputs_local,) = ref_block(short_inputs)

    assert torch.allclose(outputs_local, result, rtol=0, atol=atol_inference)
