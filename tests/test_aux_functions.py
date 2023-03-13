import pytest
import torch

from petals.client import DistributedBloomConfig
from petals.server.throughput import measure_compute_rps
from test_utils import MODEL_NAME


@pytest.mark.forked
@pytest.mark.parametrize("tensor_parallel", [False, True])
def test_compute_throughput(tensor_parallel: bool):
    config = DistributedBloomConfig.from_pretrained(MODEL_NAME)
    tensor_parallel_devices = ("cpu", "cpu") if tensor_parallel else ()
    compute_rps = measure_compute_rps(
        config,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        load_in_8bit=False,
        tensor_parallel_devices=tensor_parallel_devices,
        n_steps=10,
    )
    assert isinstance(compute_rps, float) and compute_rps > 0
