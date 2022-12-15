"""
PyTorch BLOOM model that implements several memory-efficient modes.
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from hivemind import get_logger
from torch import nn
from transformers import BloomConfig

logger = get_logger(__file__)


class LMHead(nn.Module):
    """
    The modified language modeling head which does not create extra tensor for the linear layer with weights tied to the input
    embeddings. Thus, it reduces initial memory consumption which might be crucial for large dictionaries.
    In addition, it provides an effcient way to deal with half-precision word embeddings on CPU.
    """

    def __init__(self, config: BloomConfig, word_embeddings: nn.Embedding):
        super().__init__()
        self.word_embeddings = word_embeddings
        self.chunk_size = config.chunk_size_for_efficient_fp16_on_cpu

    @property
    def in_features(self) -> int:
        return self.word_embeddings.num_embeddings

    @property
    def out_features(self) -> int:
        return self.word_embeddings.embedding_dim

    @property
    def weight(self):
        return self.word_embeddings.weight

    @property
    def bias(self):
        return None

    def forward(self, hidden_states):
        word_embeddings = self.word_embeddings.weight

        # We use 'chunked_forward' only when embeddings are in half-precision on CPU.
        if word_embeddings.dtype in [torch.float16, torch.bfloat16] and word_embeddings.device.type == "cpu":
            lm_logits = self.chunked_forward(hidden_states)
        else:
            # Switch dtype in case word_embeddings are fp16/bf16
            hidden_states = hidden_states.to(word_embeddings.dtype)
            lm_logits = F.linear(hidden_states, word_embeddings)
        return lm_logits

    def chunked_forward(self, hidden_states):
        """Splits word embeddings on chunks and iteratively casts them into fp32 to perform matmul more efficiently on CPU.
        chunk_size: provides trade-off between efficiency and extra memory consumption.
        """
        assert self.chunk_size > 0, "Chunk size for chunked forward must be positive"

        word_embeddings = self.word_embeddings.weight
        num_embeddings = self.word_embeddings.num_embeddings

        hidden_states = hidden_states.float()
        output = torch.empty(*hidden_states.shape[:-1], num_embeddings)

        for i in range(0, num_embeddings, self.chunk_size):
            chunk = word_embeddings[i : i + self.chunk_size].float()
            output[..., i : i + self.chunk_size] = F.linear(hidden_states, chunk)
        return output
