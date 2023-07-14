import os
from typing import Optional, Union

from hivemind import get_logger
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.client.routing.sequence_manager import SequenceManagerConfig
from petals.models.llama.block import WrappedLlamaBlock

logger = get_logger(__name__)


class DistributedLlamaConfig(LlamaConfig, SequenceManagerConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedLlamaBlock
    attn_class = LlamaAttention
    block_prefix = "model.layers"

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        logger.info(
            "LLaMA is available solely for non-commercial research purposes. "
            "Make sure you follow the terms of use: https://bit.ly/llama-license"
        )

        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {dht_prefix}")
        return super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
