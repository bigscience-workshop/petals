import pytest
import torch
from test_utils import MODEL_NAME, INITIAL_PEERS

from petals.client import DistributedBloomConfig
from petals.server.throughput import measure_compute_rps


@pytest.mark.forked
def test_throughput_basic():
    config = DistributedBloomConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    throughput = measure_compute_rps(
        config, device=torch.device("cpu"), dtype=torch.bfloat16, load_in_8bit=False, n_steps=10
    )
    assert isinstance(throughput, float) and throughput > 0
