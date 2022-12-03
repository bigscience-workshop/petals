from abc import ABC

import torch


class ABCBloomConstraint(ABC):
    """
    Base class of all kind of decoding constraints. It can be used to implement a new constraint.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, tokens_id: torch.Tensor, logits: torch.Tensor, hypo_ids: torch.Tensor) -> torch.Tensor:
        """
        This method is called by the decoding algorithm to apply the constraint. It changes and returns new logits.
        :param tokens_id: The token id of the last chosen token.
        :param logits: The logits from the Bloom model.
        :param hypo_ids: The hypothesis ids of the last tokens.
        """
        pass


class EosConstraint(ABCBloomConstraint):
    """
    This constrained repeats EOS token if it was generated on the previous step.
    Args:
        prefix: The prefix of the sequence.
        eos_token_id: The id of the end of sentence token.
        pad_token_id: The id of the padding token.
        min_logits: The minimum logits that can be generated. Default: -1e6.
    """

    def __init__(self, prefix: torch.Tensor, eos_token_id: int, pad_token_id: int, min_logits: float = -1e8) -> None:
        self.eos_token_id = eos_token_id
        self.min_logits = min_logits
        self.past_tokens = None

        self.wait_until_starting = (prefix == pad_token_id).sum(1).unsqueeze(1)

    def __call__(self, tokens_id: torch.Tensor, logits: torch.Tensor, hypo_ids: torch.Tensor) -> torch.Tensor:
        if self.past_tokens is not None:
            mask = (self.wait_until_starting < 0) & (self.past_tokens == self.eos_token_id)
            logits += self.min_logits * mask
            logits[mask[:, 0], self.eos_token_id] = 0

        if tokens_id is not None:
            self.past_tokens = tokens_id
            self.wait_until_starting -= 1

        return logits
