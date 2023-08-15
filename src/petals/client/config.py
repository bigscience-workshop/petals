import dataclasses
from typing import Optional, Sequence, Union

from hivemind import PeerID

from petals.constants import PUBLIC_INITIAL_PEERS


@dataclasses.dataclass
class ClientConfig:
    initial_peers: Sequence[str] = tuple(PUBLIC_INITIAL_PEERS)  # a list of initial peers for hivemind DHT
    dht_prefix: Optional[str] = None  # a prefix for all dht keys that correspond to this model (default: model name)
    daemon_startup_timeout: int = 60  # timeout for the libp2p daemon connecting to initial peers

    show_route: Union[str, bool] = "inference"  # show chosen route through servers. one of [False, "inference", True]
    allowed_servers: Optional[Sequence[Union[PeerID, str]]] = None  # if defined, send requests only to these servers
    blocked_servers: Optional[Sequence[Union[PeerID, str]]] = None  # if defined, do not use these servers
    use_server_to_server: bool = True  # Use direct server-to-server communication

    connect_timeout: float = 5  # timeout for opening a connection
    request_timeout: float = 3 * 60  # timeout for forward/backward/inference requests
    update_period: float = 60  # refresh DHT information once in this many seconds

    max_retries: Optional[int] = None  # max number retries before the client raises an exception (default: inf)
    min_backoff: float = 1  # after a repeated failure, sleep for this many seconds times 2 ** (num_failures - 1)
    max_backoff: float = 60  # limit maximal sleep time between retries to this value
    ban_timeout: float = 15  # when a remote peer fails to respond, prevent routing to that peer for this many seconds
    active_adapter: Optional[str] = None  # name of active LoRA adapter (usually, Hugging Face repo)

    max_pinged: int = 3  # max servers to ping from each sequence side, per update
    ping_timeout: float = 2  # max time to wait for pings, per update
