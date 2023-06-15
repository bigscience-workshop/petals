import os
from typing import Optional, Union

from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from petals.client.modeling_utils import LMHeadConfig
from petals.client.routing.sequence_manager import SequenceManagerConfig
from petals.llama.block import WrappedLlamaBlock
from petals.utils.auto_config import AutoDistributedConfig


class DistributedLlamaConfig(LlamaConfig, SequenceManagerConfig, LMHeadConfig):
    pre_seq_len: int = 0  # a number of tokens for prompt tuning.
    tuning_mode: Optional[str] = None  # fine-tuning regime, one of [None, "ptune", "deep_ptune"]

    block_class = WrappedLlamaBlock
    attn_class = LlamaAttention
    block_prefix = "model.layers"

    @property
    def n_head(self) -> int:  # For BLOOM compatibility
        return self.num_attention_heads

    @property
    def n_layer(self) -> int:  # For BLOOM compatibility
        return self.num_hidden_layers

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike, None], *args, **kwargs):
        config = super().from_pretrained(model_name_or_path, *args, **kwargs)
        if config.dht_prefix is None and model_name_or_path is not None and not os.path.isdir(model_name_or_path):
            config.dht_prefix = str(model_name_or_path)
        return config


AutoDistributedConfig.register(DistributedLlamaConfig)
