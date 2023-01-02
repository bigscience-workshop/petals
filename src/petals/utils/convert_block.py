"""
Tools for converting transformer blocks, applying quantization and/or tensor parallelism
"""
import re
from typing import Sequence

import bitsandbytes as bnb
import tensor_parallel as tp
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from tensor_parallel.slicing_configs import get_bloom_config
from transformers import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention

from petals.bloom.block import WrappedBloomBlock
from petals.utils.linear8bitlt_patch import CustomLinear8bitLt

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


def replace_8bit_linear(model: nn.Module, threshold=6.0):
    """
    A helper function to convert all `torch.nn.Linear` modules to `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `GPT3.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes-cudaXXX` with `XXX` is your CUDA version (e.g., 11.6 = 116)
    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` and 'score' that should
    be kept as a `torch.nn.Linear` module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        threshold (`float`, *optional*):
            `int8_threshold` for outlier detection as described in the formentioned paper. This parameters is set to
            `6.0` as described by the paper.
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold)

        if isinstance(module, torch.nn.Linear) and n not in ["lm_head", "score"]:
            model._modules[n] = CustomLinear8bitLt(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            )
            model._modules[n].weight = weight_8bit = bnb.nn.Int8Params(
                module.weight.data, requires_grad=False, has_fp16_weights=False
            ).to(module.weight.dtype)
            if weight_8bit.device.type == "cuda" and weight_8bit.data.dtype != torch.int8:
                # by default, bnb performs quantization implicitly, when module moves to cuda; if weight is already
                weight_8bit.cuda(device=weight_8bit.device)  # on cuda, we need to trigger quantization explicitly
            model._modules[n].bias = module.bias
    return model


def make_tensor_parallel(
    block: WrappedBloomBlock, model_config: BloomConfig, devices: Sequence[torch.device], output_device: torch.device
):
    assert isinstance(block, (WrappedBloomBlock, CustomLinear8bitLt))
    tp_config = get_bloom_config(model_config, devices)
    del tp_config.state_rules[re.compile(".*word_embeddings.weight$")]
    tp_block = tp.TensorParallel(block, devices, config=tp_config, output_device=output_device)
    if len(devices) == 1:
        tp_block.to(devices[0])
    total_heads = 0
    for tp_shard in tp_block.module_shards:
        for submodule in tp_shard.modules():
            if isinstance(submodule, BloomAttention):
                total_heads += submodule.num_heads
    assert total_heads == model_config.n_head
    return tp_block


def check_device_balance(devices: Sequence[torch.device]):
    if not all(device.type == "cuda" for device in devices):
        logger.warning("Running tensor parallelism on non-GPU devices; proceed at your own risk")
        return
    unique_device_capabilities = set(map(torch.cuda.get_device_capability, devices))
    if len(unique_device_capabilities) > 1:
        logger.warning(
            f"Found GPUs with uneven capabilities: {unique_device_capabilities}. "
            f"Using GPUs with different performance will cause the server to wait for the slowest GPU."
        )

    memory_per_device = tuple(torch.cuda.get_device_properties(device).total_memory for device in devices)
    used_memory = min(memory_per_device) * len(memory_per_device)
    wasted_memory_rate = (sum(memory_per_device) - used_memory) / sum(memory_per_device)
    if wasted_memory_rate > 0.05:
        logger.warning(
            f"GPU devices have highly uneven memory, {wasted_memory_rate * 100:.2f}% memory is wasted. "
            f"Consider running high-memory GPUs in a separate server."
        )
