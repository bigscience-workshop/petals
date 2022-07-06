import torch

from abc import ABC
from typing import Tuple

TokenIds = torch.Tensor
BatchIds = torch.Tensor


class DecodingAlgorithm(ABC):
    def __init__(self) -> None:
        pass

    def __call__(self, logits: torch.Tensor) -> Tuple[TokenIds, BatchIds]:
        pass


class GreedyAlgorithm(DecodingAlgorithm):
    def __call__(self, logits: torch.Tensor) -> Tuple[TokenIds, BatchIds]:
        return logits.max(-1)[1], torch.arange(logits.size(0))


class TopKAlgorithm(DecodingAlgorithm):
    # TODO: Add NumHypos, maxBatchSize
    def __init__(self, top_k: int, temperature: float = 1.0) -> None:
        self.top_k = top_k
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> Tuple[TokenIds, BatchIds]:
        logits = logits[:, -1]
        indices_to_remove = logits < torch.topk(logits, self.top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")
        probs = torch.softmax(logits / self.temperature, -1)
        return torch.multinomial(probs, num_samples=1), torch.arange(logits.size(0))


class NucleusAlgorithm(DecodingAlgorithm):
    def __init__(self, top_p: float, temperature: float = 1.0) -> None:
        self.top_p = top_p
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> Tuple[TokenIds, BatchIds]:
        logits = logits[:, -1]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits / self.temperature, -1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("Inf")
        probs = torch.softmax(logits / self.temperature, -1)
        return torch.multinomial(probs, num_samples=1), torch.arange(logits.size(0))


# TODO: In generate function we need to check usage of top_k or sampling algorithm
