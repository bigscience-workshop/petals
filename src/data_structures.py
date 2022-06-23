from typing import Collection, NamedTuple

from hivemind import PeerID

ModuleUID = str
UID_DELIMITER = "."  # delimits parts of one module uid, e.g. "bloom.transformer.h.4.self_attention"
CHAIN_DELIMITER = " "  # delimits multiple uids in a sequence, e.g. "bloom.layer3 bloom.layer4"
RemoteModuleInfo = NamedTuple("RemoteModuleInfo", [("uid", ModuleUID), ("peer_ids", Collection[PeerID])])
