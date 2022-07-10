import torch

from abc import ABC


class ABConstraint(ABC):
    def __init__(self) -> None:
        pass

    def update(self, token_id: torch.Tensor, is_started: torch.Tensor) -> None:
        pass

    def consume_prefix(self, prefix: torch.Tensor) -> None:
        pass

    def calculate_transation(self, logits: torch.Tensor) -> torch.Tensor:
        pass
    
    
class MaxNewTokensConstraint(ABConstraint):
    def __init__(self, max_new_tokens: int, eos_token_id: int, min_logits: float = -100000) -> None:
        self.max_new_tokens = max_new_tokens
        self.current_generated_tokens = 0
        self.eos_token_id = eos_token_id
        self.min_logits = min_logits
    
    def update(self, token_id: torch.Tensor, is_started: torch.Tensor) -> None:
        self.current_generated_tokens += 1

    def calculate_transation(self, logits: torch.Tensor) -> torch.Tensor:
        if self.current_generated_tokens > self.max_new_tokens:
            mask = torch.zeros_like(logits)
            mask[..., self.eos_token_id] = 1
            logits += self.min_logits * (1 - mask)
        return logits
