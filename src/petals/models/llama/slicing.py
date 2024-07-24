"""
Optimized configs for selected models. These configs are not necessary, but they can improve performance in some
cases, e.g. training with very small batches or inference with long sequences.

NB: some of these configs get fairly complicated in order to squeeze a bit of extra performance. When developing your
  own config, you can get most of the performance benefits by using auto config -- and maybe splitting MLP layers.
"""
from functools import partial
from itertools import chain
from typing import Callable, Dict, Sequence

import torch
from transformers import PretrainedConfig, LlamaConfig

from tensor_parallel.communications import CollectiveOperation
from tensor_parallel.slicer_wrapper import Config
from tensor_parallel.tensor_parallel import PerDeviceTensors

ConfigGetter = Callable[[PretrainedConfig, Sequence[torch.device]], Config]

def get_llama_config(model_config: LlamaConfig, devices: Sequence[torch.device]) -> Config:
    assert model_config.model_type == "llama", f"Trying to pass {model_config.model_type} as llama config"
    
    world_size = len(devices)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    num_kv = model_config.num_key_value_heads
    q_per_kv = model_config.num_attention_heads // model_config.num_key_value_heads

    gather_kv_across_ranks = CollectiveOperation(
        world_size=world_size, 
        func=lambda *kvs: [PerDeviceTensors(*chain(*(x or [None] for x in kvs)))] * world_size
    )

    select_kv_for_rank = lambda kvs, rank: (kvs[2 * rank], kvs[2 * rank + 1]) if kvs else None

    config = Config(
        state_rules={
            # LlamaAttention
            r".*self_attn\.q_proj\.weight$": partial(split_heads, dim=0, head_dim=q_per_kv * head_dim, world_size=world_size),
            r".*self_attn\.k_proj\.weight$": partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
            r".*self_attn\.v_proj\.weight$": partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
            r".*self_attn\.o_proj\.weight$": partial(split_heads, dim=1, head_dim=q_per_kv * head_dim, world_size=world_size),
            # LlamaFeedForward
            r".*mlp\.gate_proj\.weight$": "split 0",
            r".*mlp\.down_proj\.weight$": "split 1",
            r".*mlp\.up_proj\.weight$": "split 0",
            # LlamaModel
            #r".*embed_tokens.weight$": "split 1",
            #r".*lm_head\.weight$": "split 0",
        },
        input_rules={
            r".*self_attn$": {"past_key_value": select_kv_for_rank},
        },
        output_rules={
            r".*self_attn$": {0: "sum", 2: gather_kv_across_ranks},
            r".*mlp$": {0: "sum"},
            r".*embed_tokens$": {0: "gather -1"},
            r".*lm_head$": {0: "gather -1"},
        },
        attr_rules={
            r".*self_attn$": {
                "hidden_size": partial(split_inner_dim, num_heads=num_kv, world_size=world_size),
                "num_heads": partial(split_num_heads, world_size=world_size),
                "num_key_value_heads": partial(split_num_heads, world_size=world_size),
            }
        },
        #attr_rules={
        #    r".*self_attn$": {
        #        "hidden_size": partial(split_inner_dim, num_heads=num_kv, world_size=world_size),
        #        "num_heads": lambda n, rank: q_per_kv * split_num_heads(n // q_per_kv, rank=rank, world_size=world_size),
        #    }
        #},
    )

    return config



def split_heads(tensor: torch.Tensor, *, dim: int, head_dim: int, rank: int, world_size: int, optional: bool = False):
    """Split a tensor along dim such that each part size is divisible by head_dim"""
    if tensor is None and optional:
        return None
    assert tensor.shape[dim] % head_dim == 0, tensor.shape
    if dim < 0:
        dim = (tensor.ndim + dim) % tensor.ndim
    shape = list(tensor.shape)
    shape[dim] //= head_dim
    shape.insert(dim + 1, head_dim)
    tensor_part = tensor.reshape(shape).tensor_split(world_size, dim=dim)[rank].flatten(dim, dim + 1)
    return tensor_part


def split_num_heads(num_heads: int, *, rank: int, world_size: int):
    return torch.empty(num_heads, device="meta").tensor_split(world_size)[rank].numel()

def split_inner_dim(inner_dim: int, *, rank: int, num_heads: int, world_size: int):
    return split_num_heads(num_heads=num_heads, rank=rank, world_size=world_size) * (inner_dim // num_heads)
