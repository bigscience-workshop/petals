import asyncio
import math
import threading
import time
from functools import partial
from typing import Dict, Sequence

import hivemind
from hivemind.proto import dht_pb2
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


async def ping(
    peer_id: hivemind.PeerID,
    _dht: hivemind.DHT,
    node: hivemind.dht.DHTNode,
    *,
    wait_timeout: float = 5,
) -> float:
    try:
        ping_request = dht_pb2.PingRequest(peer=node.protocol.node_info)
        start_time = time.perf_counter()
        await node.protocol.get_stub(peer_id).rpc_ping(ping_request, timeout=wait_timeout)
        return time.perf_counter() - start_time
    except Exception as e:
        if str(e) == "protocol not supported":  # Happens on servers with client-mode DHT (e.g., reachable via relays)
            return time.perf_counter() - start_time

        logger.debug(f"Failed to ping {peer_id}:", exc_info=True)
        return math.inf


async def ping_parallel(peer_ids: Sequence[hivemind.PeerID], *args, **kwargs) -> Dict[hivemind.PeerID, float]:
    rpc_infos = await asyncio.gather(*[ping(peer_id, *args, **kwargs) for peer_id in peer_ids])
    return dict(zip(peer_ids, rpc_infos))


class PingAggregator:
    def __init__(self, dht: hivemind.DHT, *, ema_alpha: float = 0.2, expiration: float = 300):
        self.dht = dht
        self.ema_alpha = ema_alpha
        self.expiration = expiration
        self.ping_emas = hivemind.TimedStorage()
        self.lock = threading.Lock()

    def ping(self, peer_ids: Sequence[hivemind.PeerID], **kwargs) -> None:
        current_rtts = self.dht.run_coroutine(partial(ping_parallel, peer_ids, **kwargs))
        logger.debug(f"Current RTTs: {current_rtts}")

        with self.lock:
            expiration = hivemind.get_dht_time() + self.expiration
            for peer_id, rtt in current_rtts.items():
                prev_rtt = self.ping_emas.get(peer_id)
                if prev_rtt is not None and prev_rtt.value != math.inf:
                    rtt = self.ema_alpha * rtt + (1 - self.ema_alpha) * prev_rtt.value  # Exponential smoothing
                self.ping_emas.store(peer_id, rtt, expiration)

    def to_dict(self) -> Dict[hivemind.PeerID, float]:
        with self.lock, self.ping_emas.freeze():
            smoothed_rtts = {peer_id: rtt.value for peer_id, rtt in self.ping_emas.items()}
            logger.debug(f"Smothed RTTs: {smoothed_rtts}")
            return smoothed_rtts
