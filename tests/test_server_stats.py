import time

import hivemind
import pytest
import torch

from petals import AutoDistributedConfig, RemoteSequential
from petals.server.handler import CACHE_TOKENS_AVAILABLE
from test_utils import *


@pytest.mark.forked
def test_server_info(block_from: int = 2, block_to: int = 5, max_length: int = 100, max_length2: int = 50):
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME)
    config.allowed_servers = ["QmNV5G3hq2UmAck2htEgsqrmPFBff5goFZAdmKDcZLBZLX"]  # PeerID from server2.id

    dht = hivemind.DHT(initial_peers=INITIAL_PEERS, client_mode=True, start=True)
    blocks1 = RemoteSequential(config, dht=dht, start_block=block_from, end_block=block_to)
    blocks2 = RemoteSequential(config, dht=dht, start_block=block_to - 1, end_block=block_to)

    info_before = blocks1.sequence_manager.rpc_info

    with blocks1.inference_session(max_length=max_length) as sess:
        sess.step(torch.randn(1, 1, config.hidden_size))
        blocks1.sequence_manager.state.rpc_info = None  # invalidate cache
        info_inside = blocks1.sequence_manager.rpc_info

        with blocks2.inference_session(max_length=max_length2) as sess2:
            sess2.step(torch.randn(1, 1, config.hidden_size))
            blocks2.sequence_manager.state.rpc_info = None  # invalidate cache
            info_inside2 = blocks2.sequence_manager.rpc_info

    time.sleep(0.1)
    blocks1.sequence_manager.state.rpc_info = None  # invalidate cache
    info_after = blocks1.sequence_manager.rpc_info

    assert info_before[CACHE_TOKENS_AVAILABLE] == info_after[CACHE_TOKENS_AVAILABLE]
    assert info_before[CACHE_TOKENS_AVAILABLE] - info_inside[CACHE_TOKENS_AVAILABLE] == max_length * len(blocks1)
    assert info_inside[CACHE_TOKENS_AVAILABLE] - info_inside2[CACHE_TOKENS_AVAILABLE] == max_length2 * len(blocks2)
