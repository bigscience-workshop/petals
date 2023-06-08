from dataclasses import dataclass, field
from typing import Dict, Type

import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomAttention
from transformers.models.llama.modeling_llama import LlamaAttention

from petals.bloom.block import WrappedBloomBlock
from petals.llama.block import WrappedLlamaBlock


@dataclass(frozen=True)
class ModelSpec:
    block_class: Type[nn.Module]
    attn_class: Type[nn.Module]  # An nn.Module with attention, expected to have the .num_heads attribute
    block_prefix: str
    config_map: Dict[str, str] = field(
        default_factory=dict
    )  # How to translate config keys into the bigscience/bloom format


MODEL_SPECS = {
    "bloom": ModelSpec(block_class=WrappedBloomBlock, attn_class=BloomAttention, block_prefix="h"),
    "llama": ModelSpec(
        block_class=WrappedLlamaBlock,
        attn_class=LlamaAttention,
        block_prefix="model.layers",
        config_map={
            "n_head": "num_attention_heads",
            "n_layer": "num_hidden_layers",
        },
    ),
}