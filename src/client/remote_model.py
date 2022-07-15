# this code is in active development, interfaces may change
import os
from typing import Optional, Tuple

import hivemind
import torch
import torch.nn as nn
from hivemind import get_logger, use_hivemind_log_handler

from src.bloom.model import (
    BloomConfig,
    BloomForCausalLM,
    BloomForSequenceClassification,
    BloomModel,
    BloomPreTrainedModel,
    LMHead,
)
from src.client.remote_sequential import RemoteSequential

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
    num_prefix_tokens: int = 0  # a number of tokens for prompt tuning.


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

    def forward(self, *args, use_cache=None, **kwargs):
        if use_cache:
            raise ValueError("Distributed forward does not support use_cache; for efficient cache-aware generation, "
                             "please use model.transformer.inference_session() or model.generate(...)")
        return super().forward(*args, use_cache=False, **kwargs)


class DistributedBloomPrefix(DistributedBloomModel):
    """DistributedBloomModel with prefix tokens for prompt tuning"""

    def __init__(self, config):
        super().__init__(config)
        assert config.num_prefix_tokens > 0, "The number of prefix tokens must be > 0"
        self.prefix_length = config.num_prefix_tokens

        self.prompt_embeddings = nn.Embedding(self.prefix_length, config.hidden_size)
        self.prefix_tokens = torch.arange(self.prefix_length).long()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        prefix_tokens = prefix_tokens.to(self.word_embeddings.weight.device)
        prompts = self.prompt_embeddings(prefix_tokens)
        return prompts

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        position_ids=None,
        head_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert (
            input_ids is None or inputs_embeds is None
        ), "You cannot specify both input_ids and inputs_embeds at the same time"
        assert input_ids is not None or inputs_embeds is not None, "You must specify either input_ids or inputs_embeds"

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        batch_size = inputs_embeds.shape[0]

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, self.prefix_length, device=attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        prompts = self.get_prompt(batch_size)
        inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)

        transformer_outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Remove prefix
        last_hidden_state = transformer_outputs[0][:, self.prefix_length :]
        transformer_outputs["last_hidden_state"] = last_hidden_state
        return transformer_outputs


class DistributedBloomForCausalLM(BloomForCausalLM):
    """ Similar to BloomForCausalLM, but all transformer layers are hosted by the swarm"""

    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig):
        BloomPreTrainedModel.__init__(self, config)
        if config.num_prefix_tokens > 0:
            self.transformer = DistributedBloomPrefix(config)
        else:
            self.transformer = DistributedBloomModel(config)
        self.lm_head = LMHead(config, self.transformer.word_embeddings)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.word_embeddings

    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return None
        return self.lm_head

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        assert isinstance(new_embeddings, nn.Embedding)
        self.transformer.word_embeddings = self.lm_head.word_embeddings = new_embeddings
        assert self.lm_head.bias is None or len(self.lm_head.bias) == new_embeddings.num_embeddings

    def set_output_embeddings(self, new_lm_head: nn.Linear):
        with torch.no_grad():
            self.lm_head.word_embeddings.weight[...] = new_lm_head.weight
            self.lm_head.bias[...] = new_lm_head.bias


class DistributedBloomForSequenceClassification(BloomForSequenceClassification):
    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig):
        super().__init__(config)
        if config.num_prefix_tokens > 0:
            self.transformer = DistributedBloomPrefix(config)
        else:
            self.transformer = DistributedBloomModel(config)

        # Initialize weights and apply final processing
        self.post_init()
