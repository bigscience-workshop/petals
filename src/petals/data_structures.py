import dataclasses
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import pydantic
from hivemind import PeerID
from hivemind.moe.expert_uid import ExpertUID

ModuleUID = str
UID_DELIMITER = "."  # delimits parts of one module uid, e.g. "bloom.transformer.h.4.self_attention"
CHAIN_DELIMITER = " "  # delimits multiple uids in a sequence, e.g. "bloom.layer3 bloom.layer4"


def parse_uid(uid: ModuleUID) -> Tuple[str, int]:
    assert CHAIN_DELIMITER not in uid, "parse_uid() does not support chained UIDs"
    dht_prefix, index = uid.split(UID_DELIMITER)
    return dht_prefix, int(index)


@pydantic.dataclasses.dataclass
class ModelInfo:
    num_blocks: pydantic.conint(ge=1, strict=True)
    repository: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, source: dict):
        return cls(**source)


class ServerState(Enum):
    OFFLINE = 0
    JOINING = 1
    ONLINE = 2


RPS = pydantic.confloat(ge=0, allow_inf_nan=False, strict=True)


@pydantic.dataclasses.dataclass
class ServerInfo:
    state: ServerState
    throughput: RPS

    start_block: Optional[pydantic.conint(ge=0, strict=True)] = None
    end_block: Optional[pydantic.conint(ge=0, strict=True)] = None

    public_name: Optional[str] = None
    version: Optional[str] = None

    network_rps: Optional[RPS] = None
    forward_rps: Optional[RPS] = None
    inference_rps: Optional[RPS] = None

    adapters: Sequence[str] = ()
    torch_dtype: Optional[str] = None
    quant_type: Optional[str] = None
    using_relay: Optional[bool] = None
    cache_tokens_left: Optional[pydantic.conint(ge=0, strict=True)] = None
    next_pings: Optional[Dict[str, pydantic.confloat(ge=0, strict=True)]] = None

    def to_tuple(self) -> Tuple[int, float, dict]:
        extra_info = dataclasses.asdict(self)
        del extra_info["state"], extra_info["throughput"]
        return (self.state.value, self.throughput, extra_info)

    @classmethod
    def from_tuple(cls, source: tuple):
        state, throughput = source[:2]
        extra_info = source[2] if len(source) > 2 else {}
        # pydantic will validate existing fields and ignore extra ones
        return cls(state=ServerState(state), throughput=throughput, **extra_info)


@dataclasses.dataclass
class RemoteModuleInfo:
    """A remote module that is served by one or more servers"""

    uid: ModuleUID
    servers: Dict[PeerID, ServerInfo]


@dataclasses.dataclass
class RemoteSpanInfo:
    """A chain of remote blocks served by one specific remote peer"""

    peer_id: PeerID
    start: int
    end: int
    server_info: ServerInfo

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def state(self) -> ServerState:
        return self.server_info.state

    @property
    def throughput(self) -> float:
        return self.server_info.throughput


RPCInfo = Dict[str, Any]

Handle = int


@dataclasses.dataclass(frozen=True)
class InferenceMetadata:
    uid: ExpertUID
    prefix_length: int
    cache_handles: Tuple[Handle, ...]
    active_adapter: Optional[str]
