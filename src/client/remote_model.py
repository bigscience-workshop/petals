# this code is in active development, interfaces may change
import os
from typing import Optional, Tuple

import hivemind
from hivemind import get_logger, use_hivemind_log_handler

from src.bloom.model import BloomConfig, BloomForCausalLM, BloomModel, BloomPreTrainedModel, LMHead
from src.client.remote_sequential import RemoteSequential
from src.data_structures import UID_DELIMITER

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class DistributedBloomConfig(BloomConfig):
    """
    A bloom config that contains information about DHT peers.
    To create a distributed model, one must provide dht_prefix and either initial_peers or dht.
    """

    initial_peers: Tuple[str, ...] = ()  # a list of initial peers for hivemind DHT
    dht_prefix: str  # a prefix for all dht keys that correspond to this model (usually equal to model name)
    dht: Optional[hivemind.DHT] = None  # a running DHT instance, e.g. when using the same DHT for multiple models
    chunk_size_for_efficient_fp16_on_cpu: int = 10000  # a chunk size for a LM head for efficient half-precision on CPU


class DistributedBloomModel(BloomModel):
    """BloomModel, but all transformer layers are hosted by the swarm"""

    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig):
        assert config.dht_prefix, "Could not find dht_prefix in config, please create model with dht_prefix=..."
        assert config.initial_peers or config.dht, "Please specify initial_peers=list(...) or dht=hivemind.DHT(...)"

        n_layer, config.n_layer = config.n_layer, 0  # temporarily set n_layer to 0 to prevent layer initialization
        super().__init__(config)
        assert len(self.h) == 0
        config.n_layer = n_layer

        dht = (
            config.dht
            if config.dht is not None
            else hivemind.DHT(initial_peers=config.initial_peers, client_mode=True, start=True)
        )
        assert isinstance(dht, hivemind.DHT) and dht.is_alive(), "dht must be a running hivemind.DHT instance"
        self.h = RemoteSequential(config, dht, config.dht_prefix)

        # Forbid accumulate grads for embeddings and layernorm
        self.set_requires_grad(False)

    def set_requires_grad(self, value):
        for p in self.parameters():
            p.requires_grad = value


class DistributedBloomForCausalLM(BloomForCausalLM):
    """DistributedBloomForCausalLM, but all transformer layers are hosted by the swarm"""

    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig):
        BloomPreTrainedModel.__init__(self, config)
        self.transformer = DistributedBloomModel(config)
        self.lm_head = LMHead(config, self.transformer.word_embeddings)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.word_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.word_embeddings.weight = new_embeddings.weight
