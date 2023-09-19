import os
from typing import Optional, Union

from hivemind import get_logger
from transformers.models.falcon import FalconConfig
from transformers.models.falcon.modeling_falcon import FalconAttention

from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.models.falcon.block import WrappedFalconBlock
from petals.utils.auto_config import DefaultRevisionMixin

logger = get_logger(__name__)


class DistributedFalconConfig(DefaultRevisionMixin, FalconConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedFalconBlock
    attn_class = FalconAttention
    block_prefix = "transformer.h"

    @property
    def num_key_value_groups(self) -> int:
        if self.new_decoder_architecture:
            return self.num_attention_heads // self.num_kv_heads
        if self.multi_query:
            return self.num_attention_heads
        return 1

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        if "180B" in model_name_or_path.upper():
            logger.info("Make sure you follow the Falcon-180B license: https://bit.ly/falcon-180b-license")

        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        if config.pad_token_id is None:
            config.pad_token_id = 0
        return result
