import asyncio
import math
import threading
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

import requests
from hivemind.dht import DHT, DHTNode
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import P2P, P2PContext, PeerID, ServicerBase
from hivemind.proto import dht_pb2
from hivemind.utils import get_logger

from petals.constants import REACHABILITY_API_URL

logger = get_logger(__name__)


def validate_reachability(peer_id, wait_time: float = 7 * 60, retry_delay: float = 15) -> None:
    """verify that your peer is reachable from a (centralized) validator, whether directly or through a relay"""
    for attempt_no in range(math.floor(wait_time / retry_delay) + 1):
        try:
            r = requests.get(f"{REACHABILITY_API_URL}/api/v1/is_reachable/{peer_id}", timeout=10)
            r.raise_for_status()
            response = r.json()

            if response["success"]:
                logger.info("Server is reachable from the Internet. It will appear at http://health.petals.ml soon")
                return

            if attempt_no == 0:
                # Usually, libp2p manages to set up relays before we finish loading blocks.
                # In other cases, we may need to wait for up to `wait_time` seconds before it's done.
                logger.info("Detected a NAT or a firewall, connecting to libp2p relays. This takes a few minutes")
            time.sleep(retry_delay)
        except Exception as e:
            logger.warning(f"Skipping reachability check because health.petals.ml is down: {repr(e)}")
            return

    raise RuntimeError(
        f"Server has not become reachable from the Internet:\n\n"
        f"{response['message']}\n\n"
        f"You need to fix your port forwarding and/or firewall settings. How to do that:\n\n"
        f"    1. Choose a specific port for the Petals server, for example, 31337.\n"
        f"    2. Ensure that this port is accessible from the Internet and not blocked by your firewall.\n"
        f"    3. Add these arguments to explicitly announce your IP address and port to other peers:\n"
        f"        python -m petals.cli.run_server ... --public_ip {response['your_ip']} --port 31337\n"
        f"    4. If it does not help, ask for help in our Discord: https://discord.gg/Wuk8BnrEPH\n"
    )


def check_direct_reachability(max_peers: int = 5, threshold: float = 0.5, **kwargs) -> Optional[bool]:
    """test if your peer is accessible by others in the swarm with the specified network options in **kwargs"""

    async def _check_direct_reachability():
        target_dht = await DHTNode.create(client_mode=True, **kwargs)
        try:
            protocol = ReachabilityProtocol(target_dht.protocol.p2p)
            async with protocol.serve():
                successes = requests = 0
                for remote_peer in list(target_dht.protocol.routing_table.peer_id_to_uid.keys()):
                    probe_available = await protocol.call_check(remote_peer=remote_peer, check_peer=target_dht.peer_id)
                    if probe_available is None:
                        continue  # remote peer failed to check probe
                    successes += probe_available
                    requests += 1
                    if requests >= max_peers:
                        break

            logger.debug(f"Direct reachability: observed {successes} successes out of {requests} requests")
            return (successes / requests) >= threshold if requests > 0 else None
        finally:
            await target_dht.shutdown()

    return RemoteExpertWorker.run_coroutine(_check_direct_reachability())


PROBE_P2P_ARGS = dict(dht_mode="client", use_relay=False, auto_nat=False, nat_port_map=False, no_listen=True)


class ReachabilityProtocol(ServicerBase):
    """Mini protocol to test if a locally running peer is accessible by other devices in the swarm"""

    def __init__(self, p2p: P2P, *, probe: Optional[P2P] = None, wait_timeout: float = 5.0):
        probe = probe if probe is not None else p2p
        self.p2p, self.probe, self.wait_timeout = p2p, probe, wait_timeout

    async def call_check(self, remote_peer: PeerID, *, check_peer: PeerID) -> Optional[bool]:
        """Returns True if remote_peer can reach check_peer, False if it cannot, None if it did not respond"""
        try:
            request = dht_pb2.PingRequest(peer=dht_pb2.NodeInfo(node_id=check_peer.to_bytes()))
            timeout = self.wait_timeout if check_peer == remote_peer else self.wait_timeout * 2
            response = await self.get_stub(self.probe, remote_peer).rpc_check(request, timeout=timeout)
            return response.available
        except Exception as e:
            logger.debug(f"Requested {remote_peer} to check {check_peer}, but got:", exc_info=True)
            return None

    async def rpc_check(self, request: dht_pb2.PingRequest, context: P2PContext) -> dht_pb2.PingResponse:
        """Help another peer to check its reachability"""
        response = dht_pb2.PingResponse(available=True)
        check_peer = PeerID(request.peer.node_id)
        if check_peer != context.local_id:  # remote peer wants us to check someone other than ourselves
            response.available = await self.call_check(check_peer, check_peer=check_peer) is True
        return response

    @asynccontextmanager
    async def serve(self):
        try:
            await self.add_p2p_handlers(self.p2p)
            yield self
        finally:
            await self.remove_p2p_handlers(self.p2p)

    @classmethod
    def attach_to_dht(cls, dht: DHT, **kwargs):
        async def _serve_with_probe():
            protocol = cls(p2p=await dht.replicate_p2p(), **kwargs)
            initial_peers = list(map(str, await protocol.p2p.get_visible_maddrs(latest=True)))
            for info in await protocol.p2p.list_peers():
                initial_peers.extend(f"{addr}/p2p/{info.peer_id}" for addr in info.addrs)
            protocol.probe = await P2P.create(initial_peers, **PROBE_P2P_ARGS)
            try:
                async with protocol.serve():
                    while dht.is_alive():
                        await asyncio.sleep(protocol.wait_timeout)  # not awaiting join because it freezes dht.shutdown
            finally:
                await protocol.probe.shutdown()

        threading.Thread(target=partial(asyncio.run, _serve_with_probe()), daemon=True).start()
