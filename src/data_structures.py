from typing import NamedTuple, Collection

from hivemind import PeerID


ModuleUID = str
UID_DELIMITER = '.'
RemoteModuleInfo = NamedTuple("RemoteModuleInfo", [("uid", ModuleUID), ("peer_ids", Collection[PeerID])])
