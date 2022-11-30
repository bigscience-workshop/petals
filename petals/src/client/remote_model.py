# this code is in active development, interfaces may change
from typing import List, Optional

import hivemind
import torch
import torch.nn as nn
from hivemind import get_logger, use_hivemind_log_handler
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from petals.bloom.model import (
    BloomConfig,
    BloomForCausalLM,
    BloomForSequenceClassification,
    BloomModel,
    BloomPreTrainedModel,
    LMHead,
)
from petals.client.remote_generation import RemoteGenerationMixin
from petals.client.remote_sequential import RemoteSequential
from petals.constants import PUBLIC_INITIAL_PEERS
from petals.utils.misc import DUMMY

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class DistributedBloomConfig(BloomConfig):
    """
    A bloom config that contains information about DHT peers.
    To create a distributed model, one must provide dht_prefix and either initial_peers or dht.
    """

    initial_peers: List[str] = PUBLIC_INITIAL_PEERS  # a list of initial peers for hivemind DHT
    dht_prefix: str  # a prefix for all dht keys that correspond to this model (usually equal to model name)
    dht: Optional[hivemind.DHT] = None  # a running DHT instance, e.g. when using the same DHT for multiple models
    chunk_size_for_efficient_fp16_on_cpu: int = 10000  # a chunk size for a LM head for efficient half-precision on CPU
    pre_seq_len: int = 0  # a number of tokens for prompt tuning.
    tuning_mode: Optional[str] = None  # One of the finetune options: [None, 'shallow_ptune', 'deep_ptune', 'adapters']


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

        if config.tuning_mode and "ptune" in config.tuning_mode:
            assert config.pre_seq_len > 0, "The number of prefix tokens must be > 0"
            self.pre_seq_len = config.pre_seq_len
            self.prompt_embeddings = nn.Embedding(self.pre_seq_len, config.hidden_size)
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()

            if config.tuning_mode == "deep_ptune":
                self.intermediate_prompt_embeddings = nn.Embedding(
                    self.pre_seq_len,
                    config.num_hidden_layers * config.hidden_size
                    # ^-- TODO: should be num_hidden_layers - 1
                )
                self.intermediate_prompt_embeddings.weight.data.zero_()
        elif config.tuning_mode:
            raise NotImplementedError(f"{self.tuning_mode} mode is not supported for now")

    def set_requires_grad(self, value):
        for p in self.parameters():
            p.requires_grad = value

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        prefix_tokens = prefix_tokens.to(self.word_embeddings.weight.device)
        prompts = self.prompt_embeddings(prefix_tokens)

        if self.config.tuning_mode == "deep_ptune":
            intermediate_prompts = self.intermediate_prompt_embeddings(prefix_tokens)
            intermediate_prompts = intermediate_prompts.view(
                batch_size, self.pre_seq_len, len(self.h), self.config.hidden_size  # TODO: should be len(self.h) - 1
            )
            intermediate_prompts = intermediate_prompts.permute([2, 0, 1, 3])
        else:
            intermediate_prompts = DUMMY
        return prompts, intermediate_prompts

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        assert attention_mask is None, "DistributedBloomModel does not support attention masks right now"

        for k, v in kwargs.items():
            if not (v is None or v is False):
                logger.debug(f"Extra keyword arguments are not yet supported (got {k} = {v})")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.config.tuning_mode and "ptune" in self.config.tuning_mode:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.config.tuning_mode and "ptune" in self.config.tuning_mode:
            hidden_states = self.h(hidden_states, prompts=intermediate_prompts)
        else:
            hidden_states = self.h(hidden_states)

        # Remove prefix
        if self.config.tuning_mode and "ptune" in self.config.tuning_mode:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class DistributedBloomForCausalLM(RemoteGenerationMixin, BloomForCausalLM):
    """DistributedBloomForCausalLM, but all transformer layers are hosted by the swarm"""

    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig):
        BloomPreTrainedModel.__init__(self, config)
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
        BloomPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.transformer = DistributedBloomModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
