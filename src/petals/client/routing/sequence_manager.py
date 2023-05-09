from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
import random
import threading
import time
from typing import Any, Collection, Dict, List, Optional, Sequence, Union
from weakref import WeakMethod

import numpy as np
from hivemind import DHT, P2P, MSGPackSerializer, PeerID, get_dht_time
from hivemind.dht.node import Blacklist
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger

import petals.dht_utils
from petals.client.routing.sequence_info import RemoteSequenceInfo
from petals.client.routing.spending_policy import NoSpendingPolicy
from petals.data_structures import ModuleUID, RemoteSpanInfo, ServerState
from petals.server.handler import TransformerConnectionHandler

logger = get_logger(__name__)


@dataclasses.dataclass
class SequenceManagerConfig:
    allowed_servers: Optional[Collection[Union[PeerID, str]]] = None  # if defined, send requests only to these servers

    request_timeout: float = 3 * 60  # timeout for forward/backward/inference requests
    update_period: float = 60  # refresh DHT information once in this many seconds

    max_retries: Optional[int] = None  # max number retries before the client raises an exception (default: inf)
    min_backoff: float = 1  # after a repeated failure, sleep for this many seconds times 2 ** (num_failures - 1)
    max_backoff: float = 60  # limit maximal sleep time between retries to this value
    ban_timeout: float = 15  # when a remote peer fails to respond, prevent routing to that peer for this many seconds


@dataclasses.dataclass
class SequenceManagerState:
    p2p: P2P = None
    sequence_info: Optional[RemoteSequenceInfo] = None
    rpc_info: Optional[dict] = None
    banned_peers: Optional[Blacklist] = None

    def __getitem__(self, ix: Union[int, slice]) -> SequenceManagerState:
        return dataclasses.replace(self, sequence_info=self.sequence_info[ix])

    def __len__(self) -> int:
        return len(self.sequence_info)


class RemoteSequenceManager:
    """
    Sequence manager is a thread that keeps track of remote servers that hold the specified sequence of blocks.
    TL;DR it tells you, which peers you should ask to get a specific layer. It is used in RemoteSequential.
    When created, RemoteSequenceManager looks up which servers serve necessary layers by reading from DHT.
    Using this information, sequence manager can form sequences of servers that collectively have the full sequence.
    To form such a sequence, call .make_sequence with the appropriate optimization policy (see make_sequence docstr).

    :note: RemoteSequenceManager takes up some CPU and network I/O to operate in background. It is recommended to avoid
      running redundant sequence managers for the same set of layers.
    """

    def __init__(
        self,
        config: SequenceManagerConfig,
        block_uids: Sequence[ModuleUID],
        *,
        dht: Optional[DHT] = None,
        state: Optional[SequenceManagerState] = None,
    ):
        assert len(block_uids) > 0, "Sequences must contain at least one block"

        self.config = config
        if state is None:
            state = SequenceManagerState()
        self.state = state

        if dht is None:
            dht = DHT(
                initial_peers=config.initial_peers,
                client_mode=True,
                num_workers=config.n_layer,
                startup_timeout=config.daemon_startup_timeout,
                start=True,
            )
        assert isinstance(dht, DHT) and dht.is_alive(), "`dht` must be a running hivemind.DHT instance"
        self.dht = dht

        if state.p2p is None:
            state.p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p())

        self.lock_changes = threading.Lock()
        self._thread = _SequenceManagerUpdateThread(config.update_period, WeakMethod(self._update))
        self._thread_start_lock = threading.Lock()
        self.policy = NoSpendingPolicy()

        if state.banned_peers is None:
            state.banned_peers = Blacklist(base_time=config.ban_timeout, backoff_rate=2.0)
        if state.sequence_info is None:
            state.sequence_info = RemoteSequenceInfo.make_empty(block_uids)

        if state.sequence_info.last_updated_time is None:
            # Pre-fetch module infos in DHT in parallel with .from_pretrained(), then use cached records
            # in the first _update() instead of the latest ones. This makes the first .update() faster.
            petals.dht_utils.get_remote_module_infos(self.dht, self.block_uids, latest=True, return_future=True)
            self._need_latest_infos = False
        else:
            assert block_uids == state.sequence_info.block_uids
            self._thread.ready.set()  # no need to await the first dht fetch
            self._need_latest_infos = True

    def make_sequence(
        self, start_index: int = 0, end_index: Optional[int] = None, *, mode: str
    ) -> List[RemoteSpanInfo]:
        """
        Form a sequence of remote servers that collectively serve all consecutive layers

        :param start_index: optional index of the first module in a sequence, default = the first of block_uids
        :param end_index: optional index of the last module (non-inclusive), default = after last of block uids
        :param mode: one of ["max_throughput", "min_latency"]
        """
        with self._thread_start_lock:
            if not self.is_alive():
                self._thread.start()
        if not self.ready.is_set():
            self.update(wait=True)  # this will await an existing update or trigger a new one (if not updating)

        end_index = end_index if end_index is not None else len(self)
        span_sequence = []
        current_index = start_index
        while current_index < end_index:
            candidate_spans = self.state.sequence_info.spans_containing_block[current_index]
            if not candidate_spans:
                raise MissingBlocksError(current_index)

            if mode == "max_throughput":
                span_weights = np.array([span.throughput for span in candidate_spans], dtype=np.float64)
            elif mode == "min_latency":
                span_weights = np.array([span.end - current_index for span in candidate_spans], dtype=np.float64)
            else:
                raise RuntimeError(f"Unexpected mode {mode}")
            chosen_span = np.random.choice(candidate_spans, p=span_weights / span_weights.sum())

            assert chosen_span.start <= current_index < chosen_span.end
            span_sequence.append(dataclasses.replace(chosen_span, start=current_index))
            current_index = chosen_span.end

        route_repr = " => ".join([f"{span.start}:{span.end} via …{str(span.peer_id)[-6:]}" for span in span_sequence])
        logger.debug(f"Route found: {route_repr}")
        return span_sequence

    def __getitem__(self, ix: Union[int, slice]) -> RemoteSequenceManager:
        """Get a RemoteSequenceManager for a sub-sequence of blocks"""
        assert isinstance(ix, (int, slice))
        if not isinstance(ix, slice):
            ix = slice(int(ix), int(ix) + 1, 1)
        return type(self)(self.config, self.block_uids[ix], dht=self.dht, state=self.state[ix])

    def update(self, *, wait: bool):
        """Run an asynchronous update in background as soon as possible"""
        self.ready.clear()
        self._thread.trigger.set()
        if wait:
            self.ready.wait()

    def _update(self):
        """Perform an immediate and synchronous refresh, may take time"""
        new_block_infos = petals.dht_utils.get_remote_module_infos(
            self.dht, self.block_uids, latest=self._need_latest_infos
        )
        self._need_latest_infos = True  # All future _update() should use latest infos

        for block_info in new_block_infos:
            if not block_info:
                continue

            # Apply whitelist, if defined
            if self.config.allowed_servers is not None:
                block_info.servers = {
                    peer_id: server_info
                    for peer_id, server_info in block_info.servers.items()
                    if peer_id in self.config.allowed_servers or str(peer_id) in self.config.allowed_servers
                }

            # Remove temporarily banned peers, unless there are no peers left
            valid_servers = {
                peer_id: server_info
                for peer_id, server_info in block_info.servers.items()
                if peer_id not in self.state.banned_peers
            }
            if len(valid_servers) < len(block_info.servers):
                if valid_servers:
                    logger.debug(
                        f"Kept {len(valid_servers)} out of {len(block_info.servers)} servers holding {block_info.uid}"
                    )
                    block_info.servers = valid_servers
                else:
                    # If we blacklisted all servers, the error may actually be client-caused
                    logger.debug(f"All servers holding {block_info.uid} are blacklisted, ignoring blacklist")

        with self.lock_changes:
            self.state.sequence_info.update_(new_block_infos)
        self.ready.set()

    def on_request_failure(self, peer_id: Optional[PeerID]):
        """remove a given peer from the routing table. If the routing is no longer possible, trigger an update"""
        if peer_id is not None:
            logger.debug(f"Peer {peer_id} did not respond, banning it temporarily")
            self.state.banned_peers.register_failure(peer_id)
        with self.lock_changes:
            should_update = False
            for info in self.state.sequence_info.block_infos:
                info.servers.pop(peer_id, None)
                if not info.servers:
                    should_update = True
            if should_update:
                self.ready.clear()
                self.update(wait=False)

    def on_request_success(self, peer_id: PeerID):
        """if peer has a failure streak, clear that streak"""
        self.state.banned_peers.register_success(peer_id)

    def __len__(self):
        return len(self.block_uids)

    @property
    def is_alive(self):
        return self._thread.is_alive

    @property
    def ready(self) -> threading.Event:
        return self._thread.ready

    @property
    def block_uids(self):
        return self.state.sequence_info.block_uids

    @property
    def rpc_info(self):
        """Return the rpc_info queried from one of the servers that hold the first block"""
        if self.state.rpc_info is not None:
            return self.state.rpc_info

        with self._thread_start_lock:
            if not self.is_alive():
                self._thread.start()

        for attempt_no in itertools.count():
            peer_id = None
            try:
                if not self.ready.is_set():
                    self.update(wait=True)

                active_servers = [
                    peer_id
                    for peer_id, server in self.state.sequence_info.block_infos[0].servers.items()
                    if server.state == ServerState.ONLINE
                ]
                if not active_servers:
                    raise MissingBlocksError(0)
                peer_id = random.choice(active_servers)

                stub = TransformerConnectionHandler.get_stub(self.state.p2p, peer_id)
                outputs = RemoteExpertWorker.run_coroutine(
                    stub.rpc_info(runtime_pb2.ExpertUID(uid=self.block_uids[0]), timeout=self.config.request_timeout)
                )
                self.state.rpc_info = MSGPackSerializer.loads(outputs.serialized_info)
                self.on_request_success(peer_id)
                break
            except Exception as e:
                self.on_request_failure(peer_id)
                if attempt_no + 1 == self.config.max_retries:
                    raise
                delay = self.get_retry_delay(attempt_no)
                logger.warning(
                    f"Caught exception when gathering information from peer {peer_id} "
                    f"(retry in {delay:.0f} sec): {repr(e)}"
                )
                maybe_log_traceback(e)
                time.sleep(delay)

        return self.state.rpc_info

    def get_retry_delay(self, attempt_no: int) -> float:
        if attempt_no == 0:
            return 0
        return min(self.config.min_backoff * 2 ** (attempt_no - 1), self.config.max_backoff)

    def get_request_metadata(self, protocol: str, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        :param protocol: one of "rpc_forward", "rpc_backward" or "rpc_inference"
        :param args: request-specific inputs, typically block uids and input tensors
        :param kwargs: additional request context, such as remote peer ID
        :returns: msgpack-serialized metadata dict that will be passed alongside a given request
        """
        return dict(points=self.policy.get_points(protocol, *args, **kwargs))

    def shutdown(self):
        self._thread.shutdown()


class _SequenceManagerUpdateThread(threading.Thread):
    def __init__(self, update_period: float, ref_update_manager: WeakMethod):
        super().__init__(daemon=True)
        self.ref_update_manager = ref_update_manager
        self.ready = threading.Event()
        self.trigger = threading.Event()
        self.update_period = update_period
        self.should_shutdown = False

    def run(self) -> None:
        while not self.should_shutdown:
            update_manager = self.ref_update_manager()
            if update_manager is None:
                logger.debug(f"{self.__class__.__name__} exited because the sequence manager no longer exists")
                break

            try:
                self.trigger.clear()
                update_manager()
            except Exception as e:
                logger.exception(e)
            finally:
                del update_manager

            self.trigger.wait(self.update_period)

        logger.debug(f"{self.__class__.__name__} thread exited")

    def shutdown(self, timeout: Optional[float] = None):
        self.should_shutdown = True
        self.trigger.set()
        if self.is_alive():
            self.join(timeout)

    def __del__(self):
        self.shutdown()


def maybe_log_traceback(exc: Exception):
    traceback_level = logging.DEBUG if str(exc) or isinstance(exc, asyncio.TimeoutError) else logging.WARNING
    logger.log(traceback_level, "See detailed traceback below:", exc_info=True)


class MissingBlocksError(RuntimeError):
    def __init__(self, block_indices: Union[int, Sequence[int]]):
        super().__init__(
            f"No servers holding blocks {block_indices} are online. "
            f"You can check the public swarm's state at http://health.petals.ml "
            f"If there are not enough servers, please connect your GPU: "
            f"https://github.com/bigscience-workshop/petals#connect-your-gpu-and-increase-petals-capacity "
        )
