import torch
from hivemind import DHT, get_logger, use_hivemind_log_handler
from test_utils import *

from src import RemoteSequential
from src.client.remote_model import DistributedBloomConfig

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


def test_remote_sequential():
    config = DistributedBloomConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    dht = DHT(initial_peers=config.initial_peers, client_mode=True, start=True)
    test_inputs = torch.randn(1, 5, config.hidden_size, requires_grad=True)
    grad_proj = torch.randn(1, 5, config.hidden_size)

    sequential = RemoteSequential(config, dht)

    full_outputs = sequential(test_inputs)
    (full_outputs * grad_proj).sum().backward()
    assert test_inputs.grad is not None
    full_grad = test_inputs.grad.clone()
    test_inputs.grad.data.zero_()

    first_half = sequential[: config.n_layer // 2]
    second_half = sequential[config.n_layer // 2 :]
    assert len(first_half) + len(second_half) == len(sequential)
    assert abs(len(first_half) - len(second_half)) == config.n_layer % 2
    for m in sequential, first_half, second_half:
        assert isinstance(repr(m), str)

    hidden = first_half(test_inputs)
    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape == test_inputs.shape
    assert hidden.requires_grad
    second_half_outputs = second_half(hidden)
    assert torch.allclose(second_half_outputs, full_outputs)

    (second_half_outputs * grad_proj).sum().backward()
    assert torch.allclose(test_inputs.grad, full_grad)
