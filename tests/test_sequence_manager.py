import threading
import time

import pytest
import torch
from hivemind import DHT, get_logger

from petals import AutoDistributedConfig
from petals.client import RemoteSequenceManager, RemoteSequential
from petals.data_structures import UID_DELIMITER
from test_utils import *

logger = get_logger(__name__)


@pytest.mark.forked
@pytest.mark.parametrize("mode", ["max_throughput", "min_latency"])
def test_sequence_manager_basics(mode: str):
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    dht = DHT(initial_peers=config.initial_peers, client_mode=True, start=True)
    sequential = RemoteSequential(config, dht=dht)
    shutdown_evt = threading.Event()

    # test RemoteSequential with lossy compression
    block_uids = [f"{config.dht_prefix}{UID_DELIMITER}{i}" for i in range(config.num_hidden_layers)]
    sequential = RemoteSequential(
        config,
        sequence_manager=RemoteSequenceManagerWithChecks(config, block_uids, dht=dht, _was_shut_down=shutdown_evt),
    )

    sequence = sequential.sequence_manager.make_sequence(mode=mode)
    assert all(sequence[i].peer_id != sequence[i + 1].peer_id for i in range(len(sequence) - 1))

    assert sequential.sequence_manager.is_alive()
    assert sequential.sequence_manager._thread.ready.is_set()
    assert not shutdown_evt.is_set()
    sequential(torch.randn(1, 2, config.hidden_size))

    sequential.sequence_manager.shutdown()
    del sequential
    time.sleep(1)

    assert shutdown_evt.is_set()


class RemoteSequenceManagerWithChecks(RemoteSequenceManager):
    """A sequence manager that signals if it was shut down"""

    def __init__(self, *args, _was_shut_down: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self._was_shut_down = _was_shut_down

    def shutdown(self):
        super().shutdown()
        assert not self.is_alive()
        self._was_shut_down.set()
