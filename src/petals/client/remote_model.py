import dataclasses
from typing import Optional, Sequence, Union

from petals.client.routing.sequence_manager import SequenceManagerConfig
from petals.constants import PUBLIC_INITIAL_PEERS


@dataclasses.dataclass
class DistributedPretrainedConfig(SequenceManagerConfig):
    initial_peers: Sequence[str] = tuple(PUBLIC_INITIAL_PEERS)  # a list of initial peers for hivemind DHT
    dht_prefix: Optional[str] = None  # a prefix for all dht keys that correspond to this model (default: model name)
    daemon_startup_timeout: int = 60  # timeout for the libp2p daemon connecting to initial peers

    pre_seq_len: int = 0  # a number of tokens for prompt tuning.
    tuning_mode: Optional[str] = None  # fine-tuning regime, one of [None, "ptune", "deep_ptune"]

    # This settings matter for running the client with dtype bfloat16 on CPU.
    # If the CPU doesn't support AVX512, chunked_forward() significantly speeds up computations.
    use_chunked_forward: Union[str, bool] = "auto"
    chunked_forward_step: int = 16384
