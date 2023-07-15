import asyncio
import math
import threading
import time
from typing import Dict, Sequence

import hivemind
from hivemind.proto import dht_pb2
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


rtt_lock = threading.Lock()
rtt_cache = hivemind.TimedStorage()


async def ping(
    peer_id: hivemind.PeerID,
    _dht: hivemind.DHT,
    node: hivemind.dht.DHTNode,
    *,
    wait_timeout: float = 1,
    expiration: float = 600,
    use_cache: bool = True,
) -> float:
    with rtt_lock:
        if use_cache and peer_id in rtt_cache:
            return rtt_cache.get(peer_id).value

    try:
        ping_request = dht_pb2.PingRequest(peer=node.protocol.node_info)
        start_time = time.perf_counter()
        await node.protocol.get_stub(peer_id).rpc_ping(ping_request, timeout=wait_timeout)
        round_trip_time = time.perf_counter() - start_time
    except Exception:
        logger.debug(f"Failed to ping {peer_id}:", exc_info=True)
        round_trip_time = math.inf

    with rtt_lock:
        rtt_cache.store(peer_id, round_trip_time, hivemind.get_dht_time() + expiration)
    return round_trip_time


async def ping_parallel(peer_ids: Sequence[hivemind.PeerID], *args, **kwargs) -> Dict[hivemind.PeerID, float]:
    rpc_infos = await asyncio.gather(*[ping(peer_id, *args, **kwargs) for peer_id in peer_ids])
    return dict(zip(peer_ids, rpc_infos))
