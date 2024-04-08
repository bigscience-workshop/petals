"""
Tools for converting transformer blocks, applying quantization and/or tensor parallelism
"""
import re
from enum import Enum
from typing import Optional, Sequence

import tensor_parallel as tp
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from tensor_parallel.slicing_configs import get_bloom_config
from transformers import PretrainedConfig

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


class QuantType(Enum):
    NONE = 0
    INT8 = 1  # 8-bit as in the LLM.int8() paper
    NF4 = 2  # 4-bit as in the QLoRA paper


def convert_block(
    block: nn.Module,
    block_index: int,
    config: PretrainedConfig,
    tensor_parallel_devices: Sequence[torch.device],
    output_device: torch.device,
    quant_type: QuantType,
    freeze: bool = True,
    adapters: Optional[Sequence[str]] = None,
    **kwargs,
) -> tp.TensorParallel:
    """
    Optimize a transformer block for use in a Petals server, apply tensor parallelism and/or LLM.8bit quantization

    :note: some optimizations will modify the input block in-place!
    :param block: a single transformer block, either pre-trained or newly initialized
    :param config: HF transformers config for the full model
    :param tensor_parallel_devices: if specified, use tensor parallelism to split the model between these devices
    :note: if there is only a single device, model wil still be wrapped with TensorParallel (for uniformity)
    :param output_device: if tensor_parallel_devices is True, output
    :param quant_type: quantization type
    :param freeze: if True (default), make all module parameters non-trainable
    :return: a module that acts like the original block, but runs with all specified optimizations

    """
    if freeze:
        block.requires_grad_(False)

    block = make_tensor_parallel(block, config, tensor_parallel_devices, output_device=output_device)

    if quant_type != QuantType.NONE:
        block = quantize_module(block, quant_type=quant_type)

    for shard, device in zip(block.module_shards, block.devices):
        shard.to(device)

    if adapters:
        from petals.utils.peft import add_adapter_to_block, create_lora_adapter, load_peft

        create_lora_adapter(block, quant_type=quant_type)
        for adapter_name in adapters:
            adapter_config, adapter_state_dict = load_peft(
                adapter_name,
                block_idx=block_index,
                **kwargs,
            )
            add_adapter_to_block(block, block_index, adapter_name, adapter_config, adapter_state_dict)

    return block


def quantize_module(model: nn.Module, *, quant_type: QuantType) -> nn.Module:
    # Import bitsandbytes only when necessary, so Petals runs on platforms not supported by bitsandbytes
    import bitsandbytes as bnb

    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            quantize_module(module, quant_type=quant_type)

        if isinstance(module, torch.nn.Linear) and n not in ["lm_head", "score"]:
            assert module.weight.device.type == "cpu", f"expected linear layers on CPU, got {module.weight.device}"
            if quant_type == QuantType.INT8:
                model._modules[n] = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,  # Default from the LLM.int8() paper
                )
                model._modules[n].weight = bnb.nn.Int8Params(
                    module.weight.data, requires_grad=False, has_fp16_weights=False
                ).to(module.weight.dtype)
            elif quant_type == QuantType.NF4:
                compress_statistics = True
                model._modules[n] = bnb.nn.LinearNF4(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    compress_statistics=compress_statistics,
                )
                model._modules[n].weight = bnb.nn.Params4bit(
                    module.weight.data,
                    requires_grad=False,
                    quant_type="nf4",
                    blocksize=64,
                    compress_statistics=compress_statistics,
                ).to(module.weight.dtype)
            else:
                raise ValueError(f"Unsupported quant_type='{quant_type}'")
            model._modules[n].bias = module.bias
    return model


def make_tensor_parallel(
    block: nn.Module, model_config: PretrainedConfig, devices: Sequence[torch.device], output_device: torch.device
) -> nn.Module:
    if model_config.model_type == "bloom":
        tp_config = get_bloom_config(model_config, devices)
        del tp_config.state_rules[re.compile(".*word_embeddings.weight$")]
    else:
        if len(devices) > 1:
            logger.warning("Tensor parallelism is not tested for models other than BLOOM yet, proceed with caution")
        tp_config = None
    tp_block = tp.TensorParallel(block, devices, config=tp_config, output_device=output_device, delay_init=True)
    total_heads = 0
    for tp_shard in tp_block.module_shards:
        for submodule in tp_shard.modules():
            if isinstance(submodule, model_config.attn_class):
                total_heads += submodule.num_heads
    assert total_heads == model_config.num_attention_heads
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
