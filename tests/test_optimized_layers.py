from petals.models.falcon.block import WrappedFalconBlock
from petals.server.from_pretrained import load_pretrained_block
from petals.utils.auto_config import AutoDistributedConfig
from petals.server.block_utils import resolve_block_dtype
from petals.utils.convert_block import QuantType, convert_block
import torch


def test_falcon():
    config = AutoDistributedConfig.from_pretrained("tiiuae/falcon-rw-1b")
    config.alibi = False
    config.new_decoder_architecture = True

    device = "cuda:0"
    tensor_parallel_devices = (device,)
    dtype = torch.bfloat16
    quant_type = QuantType.NONE

    block = config.block_class(config).to(dtype)
    block = convert_block(block, 0, config, tensor_parallel_devices, device, quant_type=quant_type, freeze=True)

    unopt_block = WrappedFalconBlock(config).to(dtype)
    unopt_block = convert_block(
        unopt_block, 0, config, tensor_parallel_devices, device, quant_type=quant_type, freeze=True
    )

    unopt_block.load_state_dict(block.state_dict())

    for _ in range(3):
        dummy_input = torch.randn(1, 1, config.hidden_size, device="cuda", dtype=dtype)
        block_output = block(dummy_input)
        unopt_block_output = unopt_block(dummy_input)
        assert torch.allclose(block_output[0], unopt_block_output[0], atol=1e-6, rtol=0)
