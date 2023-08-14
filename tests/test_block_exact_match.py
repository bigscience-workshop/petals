import random

import pytest
import torch

from petals import AutoDistributedConfig, RemoteSequential
from petals.server.block_functions import MAX_SHORT_INFERENCE_TOKENS
from petals.server.from_pretrained import load_pretrained_block
from test_utils import *


@pytest.mark.forked
def test_remote_block_exact_match(atol_forward=1e-4, atol_inference=1e-3):
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    remote_sequential = RemoteSequential(config)

    block_index = random.randint(0, config.num_hidden_layers - 1)
    remote_block = remote_sequential[block_index]

    inputs = torch.randn(1, MAX_SHORT_INFERENCE_TOKENS + 8, config.hidden_size)
    outputs_forward = remote_block(inputs)

    outputs_inference = []
    with torch.inference_mode():
        with remote_block.inference_session(max_length=inputs.shape[1]) as sess:
            # Test long inference (unmerged inference pools)
            outputs_inference.append(sess.step(inputs[:, : MAX_SHORT_INFERENCE_TOKENS + 1, :]))

            # Test short inference (merged inference pools)
            for i in range(MAX_SHORT_INFERENCE_TOKENS + 1, inputs.shape[1]):
                outputs_inference.append(sess.step(inputs[:, i : i + 1, :]))

            # test that max length is respected
            with pytest.raises(ValueError, match=r"Maximum length exceeded") as exc_info:
                sess.step(inputs[:, -1:, :])
            assert "Maximum length exceeded" in repr(exc_info.value)
    outputs_inference = torch.cat(outputs_inference, dim=1)

    ref_block = load_pretrained_block(MODEL_NAME, block_index, torch_dtype=torch.float32)
    (outputs_local,) = ref_block(inputs)

    assert torch.allclose(outputs_local, outputs_forward, rtol=0, atol=atol_forward)
    assert torch.allclose(outputs_local, outputs_inference, rtol=0, atol=atol_inference)
