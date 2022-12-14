import random

import hivemind
import pytest
import torch
from test_utils import *

from petals.bloom.from_pretrained import load_pretrained_block
from petals.client import DistributedBloomConfig
from petals.client.remote_sequential import RemoteTransformerBlock
from petals.data_structures import UID_DELIMITER
from petals.dht_utils import get_remote_module


@pytest.mark.forked
def test_remote_block_exact_match(atol_forward=1e-4, atol_inference=1e-3):
    dht = hivemind.DHT(initial_peers=INITIAL_PEERS, client_mode=True, start=True)
    config = DistributedBloomConfig.from_pretrained(MODEL_NAME)

    for block_index in random.sample(range(config.n_layer), 3):
        remote_block = get_remote_module(dht, f"{MODEL_NAME}{UID_DELIMITER}{block_index}", config)
        assert isinstance(remote_block, RemoteTransformerBlock)

        inputs = torch.randn(1, 8, config.hidden_size)
        outputs_forward = remote_block(inputs)

        outputs_inference = []
        with remote_block.inference_session(max_length=inputs.shape[1]) as sess:
            for i in range(inputs.shape[1]):
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
