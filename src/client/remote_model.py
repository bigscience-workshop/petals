# this code is in active development, interfaces may change

from hivemind import DHT, use_hivemind_log_handler, get_logger

from src.bloom import DistributedBloomConfig, BloomForCausalLM
from src.client.remote_sequential import RemoteSequential
from src.data_structures import UID_DELIMITER

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class DistributedBloomForCausalLM(BloomForCausalLM):
    """BloomForCausalLM, but all transformer layers are hosted by the swarm"""
    def __init__(self, config: DistributedBloomConfig, dht: DHT, prefix: str):
        logger.warning(f"{self.__class__.__name__} is in active development; expect adventures")
        if prefix.endswith(UID_DELIMITER):
            logger.warning(f"dht_prefix {prefix} already ends with '{UID_DELIMITER}'."
                           f"This will cause {self.__class__.__name__} to look for modules under "
                           f"{prefix}{UID_DELIMITER}*. Please make sure this is what you intended.")

        n_layer, config.n_layer = config.n_layer, 0  # temporarily set n_layer to 0 to prevent layer initialization
        super().__init__(config)
        assert len(self.transformer.h) == 0
        config.n_layer = n_layer
        self.transformer.h = RemoteSequential(config, dht, prefix)


