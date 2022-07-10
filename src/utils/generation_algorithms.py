import torch

from abc import ABC
from typing import Tuple

TokenIds = torch.Tensor
HypoIds = torch.Tensor


class DecodingAlgorithm(ABC):
    """
    An abstract class for decoding algorithms. Describe base function of those algorithms: they have to select new tokens and provide the corresponding hypothesis.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, logits: torch.Tensor) -> Tuple[TokenIds, HypoIds]:
        """
        :param logits: A tensor of shape (batch_size, seq_lenth, vocab_size)
        :return: A tuple of selected token ids and corresponding hypothesis. The shape of the token ids is (batch_size, seq_length) and the shape of the hypothesis is (batch_size)
        """
        pass


class GreedyAlgorithm(DecodingAlgorithm):
    """
    The simpliest algorithm for decoding. It selects the most probable token.
    """
    def __call__(self, logits: torch.Tensor) -> Tuple[TokenIds, HypoIds]:
        """
        Returns the most propable token. The second return object always are range of integers from 0 to batch_size - 1.
        """
        return logits.max(-1)[1].unsqueeze(1), torch.arange(logits.size(0))


class SamplingAlgorithm(DecodingAlgorithm):
    def sample(self, logits: torch.Tensor, indices_to_remove: torch.Tensor) -> Tuple[TokenIds, HypoIds]:
        """
        :param logits: A tensor of shape (batch_size * num_hypos, vocab_size)
        :param indices_to_remove: A bool tensor of shape (batch_size * num_hypos, vocab_size)
        :return: A tuple of selected token ids and corresponding hypothesis. The shape of the token ids is (batch_size, seq_length) and the shape of the hypothesis is (batch_size). 
        """
        logits[indices_to_remove] = -float("Inf")
        probs = torch.softmax(logits / self.temperature, -1)
        return torch.multinomial(probs, num_samples=1), torch.arange(logits.size(0))


class TopKAlgorithm(SamplingAlgorithm):
    # TODO: Add NumHypos, maxBatchSize
    def __init__(self, top_k: int, temperature: float = 1.0) -> None:
        self.top_k = top_k
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> Tuple[TokenIds, HypoIds]:
        indices_to_remove = logits < torch.topk(logits, self.top_k, dim=-1)[0][..., -1, None]
        return self.sample(logits, indices_to_remove)


class NucleusAlgorithm(SamplingAlgorithm):
    def __init__(self, top_p: float, temperature: float = 1.0) -> None:
        self.top_p = top_p
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> Tuple[TokenIds, HypoIds]:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits / self.temperature, -1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        return self.sample(logits, indices_to_remove)


# TODO: In generate function we need to check usage of top_k or sampling algorithm
