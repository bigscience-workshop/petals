from collections import defaultdict
from typing import Sequence

import torch
from hivemind import DHT
from torch import nn

from src import DistributedBloomConfig
from src.server.backend import MAX_LENGTH


class RemoteInferenceChain(nn.Module):
    """An auxiliary class that manages distributed inference in a chain of one or more remote transformer modules"""

    def __init__(self, dht: DHT, config: DistributedBloomConfig, block_names: Sequence[str]):
        super().__init__()
        self.dht = dht
        self.config, self.block_names = config, block_names
        self.block_caches = {name: torch.zeros(1, MAX_LENGTH, config.hidden_size) for name in block_names}
        self.current_position = 0

    def step(self, hidden_states: torch.Tensor):
        pass


# plan:
# - run inference STUB from a jupyter notebook
# - extend to run actual inference
# - extend to run multiple layers at a time
