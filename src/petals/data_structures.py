from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from hivemind import PeerID

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


@dataclass
class RemoteModuleInfo:
    """A remote module that is served by one or more servers"""

    uid: ModuleUID
    servers: Dict[PeerID, ServerInfo]


@dataclass
class RemoteSpanInfo:
    """A chain of remote blocks served by one specific remote peer"""

    start: int
    end: int
    peer_id: PeerID


RPCInfo = Dict[str, Any]
