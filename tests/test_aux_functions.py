import pytest
import torch
from test_utils import MODEL_NAME

from petals.client import DistributedBloomConfig
from petals.server.throughput import measure_compute_rps, measure_network_rps


@pytest.mark.forked
def test_throughput_basic():
    config = DistributedBloomConfig.from_pretrained(MODEL_NAME)
    compute_rps = measure_compute_rps(
        config, device=torch.device("cpu"), dtype=torch.bfloat16, load_in_8bit=False, n_steps=10
    )
    assert isinstance(compute_rps, float) and compute_rps > 0
    network_rps = measure_network_rps(config)
    assert isinstance(network_rps, float) and network_rps > 0
