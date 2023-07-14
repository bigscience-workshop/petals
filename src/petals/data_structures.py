from __future__ import annotations

import math

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from hivemind import PeerID
from hivemind.moe.expert_uid import ExpertUID

from petals.server.memory_cache import Handle

ModuleUID = str
UID_DELIMITER = "."  # delimits parts of one module uid, e.g. "bloom.transformer.h.4.self_attention"
CHAIN_DELIMITER = " "  # delimits multiple uids in a sequence, e.g. "bloom.layer3 bloom.layer4"


class ServerState(Enum):
    OFFLINE = 0
    JOINING = 1
    ONLINE = 2


@dataclass
class ServerInfo:
    state: ServerState
    throughput: float

    adapters: Tuple[str] = ()
    version: Optional[str] = None
    using_relay: Optional[bool] = None
    cache_tokens_left: Optional[int] = None

    def to_tuple(self) -> Tuple[int, float, dict]:
        extra_info = dataclasses.asdict(self)
        del extra_info["state"], extra_info["throughput"]
        return (self.state.value, self.throughput, extra_info)

    @classmethod
    def from_tuple(cls, info: tuple):
        state, throughput = info[:2]
        extra_info = info[2] if len(info) > 2 else {}

        if not (
            isinstance(state, int) and isinstance(throughput, float) and math.isfinite(throughput) and throughput >= 0.0
        ):
            # FIXME: Use proper validation for all fields with pydantic 1.0
            raise ValueError(f"Invalid server info: {info}")

        return cls(ServerState(info["state"]), info["throughput"], **extra_info)


@dataclass
class RemoteModuleInfo:
    """A remote module that is served by one or more servers"""

    uid: ModuleUID
    servers: Dict[PeerID, ServerInfo]


@dataclass
class RemoteSpanInfo:
    """A chain of remote blocks served by one specific remote peer"""

    peer_id: PeerID
    start: int
    end: int
    throughput: float

    @property
    def length(self):
        return self.end - self.start


RPCInfo = Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class InferenceMetadata:
    uid: ExpertUID
    prefix_length: int
    cache_handles: Tuple[Handle, ...]
    active_adapter: Optional[str]
