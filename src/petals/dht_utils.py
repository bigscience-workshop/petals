"""
Utilities for declaring and retrieving active model layers using a shared DHT.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

from hivemind.dht import DHT, DHTNode, DHTValue
from hivemind.p2p import PeerID
from hivemind.utils import DHTExpiration, MPFuture, get_dht_time, get_logger

from petals.data_structures import CHAIN_DELIMITER, UID_DELIMITER, ModuleUID, RemoteModuleInfo, ServerInfo

logger = get_logger(__name__)


def declare_active_modules(
    dht: DHT,
    uids: Sequence[ModuleUID],
    server_info: ServerInfo,
    expiration_time: DHTExpiration,
    wait: bool = True,
) -> Union[Dict[ModuleUID, bool], MPFuture[Dict[ModuleUID, bool]]]:
    """
    Declare that your node serves the specified modules; update timestamps if declared previously

    :param uids: a list of module ids to declare
    :param wait: if True, awaits for declaration to finish, otherwise runs in background
    :param throughput: specify your performance in terms of compute throughput
    :param expiration_time: declared modules will be visible for this many seconds
    :returns: if wait, returns store status for every key (True = store succeeded, False = store rejected)
    """
    if isinstance(uids, str):
        uids = [uids]
    if not isinstance(uids, list):
        uids = list(uids)
    for uid in uids:
        assert isinstance(uid, ModuleUID) and UID_DELIMITER in uid and CHAIN_DELIMITER not in uid

    return dht.run_coroutine(
        partial(_declare_active_modules, uids=uids, server_info=server_info, expiration_time=expiration_time),
        return_future=not wait,
    )


async def _declare_active_modules(
    dht: DHT,
    node: DHTNode,
    uids: List[ModuleUID],
    server_info: ServerInfo,
    expiration_time: DHTExpiration,
) -> Dict[ModuleUID, bool]:
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    return await node.store_many(
        keys=uids,
        subkeys=[dht.peer_id.to_base58()] * len(uids),
        values=[server_info.to_tuple()] * len(uids),
        expiration_time=expiration_time,
        num_workers=num_workers,
    )


def get_remote_module_infos(
    dht: DHT,
    uids: Sequence[ModuleUID],
    expiration_time: Optional[DHTExpiration] = None,
    active_adapter: Optional[str] = None,
    *,
    latest: bool = False,
    return_future: bool = False,
) -> Union[List[Optional[RemoteModuleInfo]], MPFuture]:
    return dht.run_coroutine(
        partial(
            _get_remote_module_infos,
            uids=uids,
            active_adapter=active_adapter,
            expiration_time=expiration_time,
            latest=latest,
        ),
        return_future=return_future,
    )


async def _get_remote_module_infos(
    dht: DHT,
    node: DHTNode,
    uids: List[ModuleUID],
    active_adapter: Optional[str],
    expiration_time: Optional[DHTExpiration],
    latest: bool,
) -> List[Optional[RemoteModuleInfo]]:
    if latest:
        assert expiration_time is None, "You should define either `expiration_time` or `latest`, not both"
        expiration_time = math.inf
    elif expiration_time is None:
        expiration_time = get_dht_time()
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    found: Dict[ModuleUID, DHTValue] = await node.get_many(uids, expiration_time, num_workers=num_workers)

    modules: List[Optional[RemoteModuleInfo]] = [None] * len(uids)
    for i, uid in enumerate(uids):
        metadata = found[uid]
        if metadata is None or not isinstance(metadata.value, dict):
            if metadata is not None:
                logger.warning(f"Incorrect metadata for {uid}: {metadata}")
            continue
        servers = {}
        for peer_id, server_info in metadata.value.items():
            try:
                peer_id = PeerID.from_base58(peer_id)
                server_info = ServerInfo.from_tuple(server_info.value)

                if active_adapter and active_adapter not in server_info.adapters:
                    logger.debug(f"Skipped server {peer_id} since it does not have adapter {active_adapter}")
                    continue

                servers[peer_id] = server_info
            except (TypeError, ValueError) as e:
                logger.warning(f"Incorrect peer entry for uid={uid}, peer_id={peer_id}: {e}")
        if servers:
            modules[i] = RemoteModuleInfo(uid, servers)
    return modules
