import asyncio
from functools import partial
from typing import Optional, Sequence

import hivemind
from hivemind.dht import DHTNode
from hivemind.p2p import P2P, P2PContext, PeerID, ServicerBase
from hivemind.proto import dht_pb2
from hivemind.utils import get_logger

logger = get_logger(__name__)


async def check_if_reachable(
        initial_peers: Sequence[str], max_peers: int = 5, threshold: float = 0.5, **kwargs) -> Optional[bool]:
    """test if your peer is accessible by others in the swarm with the specified network options in **kwargs"""
    probe = await P2P.create(initial_peers=initial_peers, **kwargs)
    cancel_event = asyncio.Event()
    probe_task = asyncio.create_task(ReachabilityProtocol(probe).serve(cancel_event))
    dht_tester = await DHTNode.create(initial_peers=initial_peers, client_mode=True, no_listen=True)

    try:
        # close existing connections so that remote peers will attempt to open new ones
        for peer_info in (await probe._client.list_peers()):
            await probe._client.disconnect(peer_info.peer_id)
        protocol = ReachabilityProtocol(dht_tester.protocol.p2p)
        successes = requests = 0
        for remote_peer in dht_tester.protocol.routing_table.peer_id_to_uid.keys():
            probe_available = await protocol.call_check(remote_peer=remote_peer, check_peer=probe.peer_id)
            if probe_available is None:
                continue  # remote peer failed to check probe
            successes += probe_available
            requests += 1
            if requests >= max_peers:
                break

        logger.debug(f"Reachability: found {successes} successes out of {requests} requests")
        if requests:
            return (successes / requests) >= threshold
        else:
            return None  # could not determine result
    finally:
        cancel_event.set()
        await probe_task
        await probe.shutdown()
        await dht_tester.shutdown()


class ReachabilityProtocol(ServicerBase):
    """Mini protocol to test if a locally running peer is accessible by other devices in the swarm"""
    def __init__(self, p2p: P2P, wait_timeout: float = 5.0):
        self.p2p, self.wait_timeout = p2p, wait_timeout
        super().__init__()

    async def call_check(self, remote_peer: PeerID, *, check_peer: PeerID) -> Optional[bool]:
        """return True if remote_peer can reach check_peer, False if cannot, None means remote_peer did not respond"""
        try:
            request = dht_pb2.PingRequest(peer=dht_pb2.NodeInfo(node_id=check_peer.to_bytes()))
            timeout = self.wait_timeout if check_peer == remote_peer else self.wait_timeout * 2
            response = await self.get_stub(self.p2p, remote_peer).rpc_check(request, timeout=timeout)
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
    asyncio.create_task(cls(node.protocol.p2p, **kwargs).serve())
