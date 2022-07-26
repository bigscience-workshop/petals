"""
PyTorch BLOOM model that implements several memory-efficient modes.
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
from typing import Tuple, Union, Optional

import torch
import torch.utils.checkpoint
from torch import nn

from src.bloom.model import BloomModel
use_hivemind_log_handler("in_root_logger")
logger = logging.get_logger(__file__)


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = False
        self.embedding = nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class BloomPrefixV2(BloomModel):
    """DistributedBloomModel with prefix tokens for prompt tuning"""

    def __init__(self, config):
        super().__init__(config)
        assert config.pre_seq_len > 0, "The number of prefix tokens must be > 0"
        assert config.prompt_tuning_mode == 'deep'

        self.pre_seq_len = config.pre_seq_len
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        
        self.prefix_encoder = PrefixEncoder(config)
        self.hidden_size = config.hidden_size 
        # self.dropout = torch.nn.Dropout(0.0)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            len(self.h) * 2, 
            self.n_head,
            self.hidden_size // self.n_head
        )
        # past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 1, 3, 4]).split(2)
        return past_key_values

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

        if position_ids is not None:
            logger.warning("position_ids are ignored in this bloom implementation")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        batch_size = inputs_embeds.shape[0]

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, self.prefix_length, device=attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        past_key_values = self.get_prompt(batch_size=batch_size)

        transformer_outputs = super().forward(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            head_mask=head_mask, 
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return transformer_outputs
        