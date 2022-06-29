# this code is in active development, interfaces may change
import os
from typing import Optional, Union

import hivemind
from hivemind import DHT, get_logger, use_hivemind_log_handler

from src.bloom import BloomForCausalLM, DistributedBloomConfig
from src.bloom.from_pretrained import CLIENT_BRANCH, _load_state_dict
from src.client.remote_sequential import RemoteSequential
from src.data_structures import UID_DELIMITER

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class DistributedBloomForCausalLM(BloomForCausalLM):
    """BloomForCausalLM, but all transformer layers are hosted by the swarm"""

    def __init__(self, config: DistributedBloomConfig, dht: DHT, prefix: str):
        n_layer, config.n_layer = config.n_layer, 0  # temporarily set n_layer to 0 to prevent layer initialization
        super().__init__(config)
        assert len(self.transformer.h) == 0
        config.n_layer = n_layer
        self.transformer.h = RemoteSequential(config, dht, prefix)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        assert 'initial_peers' in kwargs
        dht = hivemind.DHT(
            initial_peers=kwargs.pop('initial_peers'), client_mode=kwargs.pop('client_mode', True),
            start=True)

        if 'prefix' not in kwargs:
            logger.warning(f"No prefix specified; setting prefix to {pretrained_model_name_or_path}")
            assert UID_DELIMITER not in pretrained_model_name_or_path, \
                f"Cannot infer prefix automatically from {pretrained_model_name_or_path}; please specify prefix=..."
        prefix = kwargs.pop("prefix", pretrained_model_name_or_path)

        config = DistributedBloomConfig.from_pretrained(pretrained_model_name_or_path, revision=CLIENT_BRANCH, **kwargs)
        model = cls(config, dht, prefix)
        model.load_state_dict(_load_state_dict(
            pretrained_model_name_or_path, use_auth_token=kwargs.get('use_auth_token')
        ), strict=True)
        return model



