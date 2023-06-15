import os
from typing import Optional, Union

from transformers.models.bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention

from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.client.routing.sequence_manager import SequenceManagerConfig
from petals.models.bloom.block import WrappedBloomBlock
from petals.utils.auto_config import AutoDistributedConfig


class DistributedBloomConfig(BloomConfig, SequenceManagerConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedBloomBlock
    attn_class = BloomAttention
    block_prefix = "h"

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        if dht_prefix is None and model_name_or_path is not None and not os.path.isdir(model_name_or_path):
            dht_prefix = str(model_name_or_path)
        return super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)


AutoDistributedConfig.register(DistributedBloomConfig)
