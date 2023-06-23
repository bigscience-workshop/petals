import os
from typing import Optional, Union

from hivemind import get_logger
from transformers.models.bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention

from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.client.routing.sequence_manager import SequenceManagerConfig
from petals.models.bloom.block import WrappedBloomBlock
from petals.utils.auto_config import AutoDistributedConfig
from petals.utils.version import get_compatible_model_repo

logger = get_logger(__name__)


class DistributedBloomConfig(BloomConfig, SequenceManagerConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedBloomBlock
    attn_class = BloomAttention
    block_prefix = "h"

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            # We need "-petals" for backward compatibility with Petals < 1.2.0
            dht_prefix = str(model_name_or_path) + "-petals"
            logger.info(f"Using DHT prefix: {dht_prefix}")
        return super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
