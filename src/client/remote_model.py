# this code is in active development, interfaces may change
from typing import Optional

from hivemind import DHT, get_logger, use_hivemind_log_handler

from src.bloom import BloomForCausalLM, DistributedBloomConfig
from src.client.remote_sequential import RemoteSequential
from src.data_structures import UID_DELIMITER

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class DistributedBloomForCausalLM(BloomForCausalLM):
    """BloomForCausalLM, but all transformer layers are hosted by the swarm"""

    def __init__(self, config: DistributedBloomConfig, dht: DHT, prefix: Optional[str] = None):
        n_layer, config.n_layer = config.n_layer, 0  # temporarily set n_layer to 0 to prevent layer initialization
        super().__init__(config)
        assert len(self.transformer.h) == 0
        config.n_layer = n_layer
        self.transformer.h = RemoteSequential(config, dht, prefix)
