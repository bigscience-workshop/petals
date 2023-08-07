import subprocess
import sys

import pytest
import torch

from petals import AutoDistributedConfig
from petals.server.throughput import measure_compute_rps
from petals.utils.convert_block import QuantType
from test_utils import MODEL_NAME


def test_bnb_not_imported_when_unnecessary():
    """
    We avoid importing bitsandbytes when it's not used,
    since bitsandbytes doesn't always find correct CUDA libs and may raise exceptions because of that.

    If this test fails, please change your code to import bitsandbytes and/or petals.utils.peft
    in the function's/method's code when it's actually needed instead of importing them in the beginning of the file.
    This won't slow down the code - importing a module for the 2nd time doesn't rerun module code.
    """

    subprocess.check_call([sys.executable, "-c", "import petals, sys; assert 'bitsandbytes' not in sys.modules"])


@pytest.mark.forked
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("n_tokens", [1, 16])
@pytest.mark.parametrize("tensor_parallel", [False, True])
def test_compute_throughput(inference: bool, n_tokens: int, tensor_parallel: bool):
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME)
    tensor_parallel_devices = ("cpu", "cpu") if tensor_parallel else ()
    compute_rps = measure_compute_rps(
        config,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        quant_type=QuantType.NONE,
        tensor_parallel_devices=tensor_parallel_devices,
        n_tokens=n_tokens,
        n_steps=5,
        inference=inference,
    )
    assert isinstance(compute_rps, float) and compute_rps > 0
