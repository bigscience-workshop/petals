import asyncio
from functools import partial
from typing import Optional, Sequence

import hivemind
from hivemind.dht import DHTNode
from hivemind.p2p import P2P, P2PContext, PeerID, ServicerBase
from hivemind.proto import dht_pb2
from hivemind.utils import get_logger

logger = get_logger(__name__)


async def check_reachability(max_peers: int = 5, threshold: float = 0.5, **kwargs) -> Optional[bool]:
    """test if your peer is accessible by others in the swarm with the specified network options in **kwargs"""
    dht_tester = await DHTNode.create(client_mode=True, **kwargs)
    protocol = ReachabilityProtocol(dht_tester.protocol.p2p)
    cancel_event = asyncio.Event()
    serve_task = asyncio.create_task(protocol.serve(cancel_event))
    try:
        successes = requests = 0
        for remote_peer in list(dht_tester.protocol.routing_table.peer_id_to_uid.keys()):
            probe_available = await protocol.call_check(remote_peer=remote_peer, check_peer=dht_tester.peer_id)
            if probe_available is None:
                continue  # remote peer failed to check probe
            successes += probe_available
            requests += 1
            if requests >= max_peers:
                break
        logger.debug(f"Reachability: observed {successes} successes out of {requests} requests")
        return (successes / requests) >= threshold if requests > 0 else None
    finally:
        cancel_event.set()
        await serve_task
        await dht_tester.shutdown()


class ReachabilityProtocol(ServicerBase):
    """Mini protocol to test if a locally running peer is accessible by other devices in the swarm"""

    def __init__(self, p2p: P2P, *, probe: Optional[P2P] = None, wait_timeout: float = 5.0):
        probe = probe if probe is not None else p2p
        self.p2p, self.probe, self.wait_timeout = p2p, probe, wait_timeout
        super().__init__()

    async def call_check(self, remote_peer: PeerID, *, check_peer: PeerID) -> Optional[bool]:
        """return True if remote_peer can reach check_peer, False if cannot, None means remote_peer did not respond"""
        try:
            request = dht_pb2.PingRequest(peer=dht_pb2.NodeInfo(node_id=check_peer.to_bytes()))
            timeout = self.wait_timeout if check_peer == remote_peer else self.wait_timeout * 2
            response = await self.get_stub(self.probe, remote_peer).rpc_check(request, timeout=timeout)
            return response.available
        except Exception as e:
            logger.debug(f"requested {remote_peer} to check {check_peer}, but got {repr(e)}", exc_info=True)

    async def rpc_check(self, request: dht_pb2.PingRequest, context: P2PContext) -> dht_pb2.PingResponse:
        """Another peer wants us to help it check reachability"""
        response = dht_pb2.PingResponse(available=True)
        check_peer = PeerID(request.peer.node_id)
        if check_peer != context.local_id:  # remote peer wants us to check someone other than ourselves
            response.available = await self.call_check(check_peer, check_peer=check_peer) is True
        return response

    async def serve(self, cancel_event: Optional[asyncio.Event] = None):
        try:
            await self.add_p2p_handlers(self.p2p)
            await (asyncio.Event() if cancel_event is None else cancel_event).wait()
        finally:
            await self.remove_p2p_handlers(self.p2p)

    @classmethod
    def attach_to_dht(cls, dht: hivemind.DHT, **kwargs):
        return dht.run_coroutine(partial(_attach_to_dht, cls=cls, **kwargs))


async def _attach_to_dht(_: hivemind.DHT, node: DHTNode, cls: callable, **kwargs):
    p2p = node.protocol.p2p
    initial_peers = [f"{addr}/p2p/{info.peer_id.to_base58()}" for info in await p2p.list_peers() for addr in info.addrs]
    initial_peers.extend(map(str, await p2p.get_visible_maddrs()))
    probe = await P2P.create(initial_peers=initial_peers, dht_mode="client", no_listen=True)
    asyncio.create_task(cls(p2p, probe=probe, **kwargs).serve())
