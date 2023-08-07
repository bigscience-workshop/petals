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

import dijkstar
import numpy as np
from hivemind import DHT, P2P, MSGPackSerializer, PeerID
from hivemind.dht.node import Blacklist
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger

import petals.dht_utils
from petals.client.routing.sequence_info import RemoteSequenceInfo
from petals.client.routing.spending_policy import NoSpendingPolicy
from petals.constants import PUBLIC_INITIAL_PEERS
from petals.data_structures import ModuleUID, RemoteSpanInfo, ServerState
from petals.server.handler import TransformerConnectionHandler
from petals.utils.ping import PingAggregator
from petals.utils.random import sample_up_to

logger = get_logger(__name__)


@dataclasses.dataclass
class SequenceManagerConfig:
    initial_peers: Sequence[str] = tuple(PUBLIC_INITIAL_PEERS)  # a list of initial peers for hivemind DHT
    dht_prefix: Optional[str] = None  # a prefix for all dht keys that correspond to this model (default: model name)
    daemon_startup_timeout: int = 60  # timeout for the libp2p daemon connecting to initial peers

    show_route: Union[str, bool] = "inference"  # show chosen route through servers. one of [False, "inference", True]
    allowed_servers: Optional[Collection[Union[PeerID, str]]] = None  # if defined, send requests only to these servers
    use_server_to_server: bool = True  # Use direct server-to-server communication

    connect_timeout: float = 5  # timeout for opening a connection
    request_timeout: float = 3 * 60  # timeout for forward/backward/inference requests
    update_period: float = 60  # refresh DHT information once in this many seconds

    max_retries: Optional[int] = None  # max number retries before the client raises an exception (default: inf)
    min_backoff: float = 1  # after a repeated failure, sleep for this many seconds times 2 ** (num_failures - 1)
    max_backoff: float = 60  # limit maximal sleep time between retries to this value
    ban_timeout: float = 15  # when a remote peer fails to respond, prevent routing to that peer for this many seconds
    active_adapter: Optional[str] = None  # name of active LoRA adapter (usually, Hugging Face repo)

    max_pinged: int = 5  # max servers to ping from each sequence side, per update
    ping_timeout: float = 2  # max time to wait for pings, per update


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
        assert config.initial_peers or dht is not None, "Please specify `config.initial_peers` or `dht`"
        assert config.dht_prefix, "Could not find dht_prefix in config, please create model with dht_prefix=..."
        assert len(block_uids) > 0, "Sequences must contain at least one block"

        self.config = config
        if state is None:
            state = SequenceManagerState()
        self.state = state

        if dht is None:
            dht = DHT(
                initial_peers=config.initial_peers,
                client_mode=True,
                num_workers=32,
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

        self.ping_aggregator = PingAggregator(dht)

        if state.banned_peers is None:
            state.banned_peers = Blacklist(base_time=config.ban_timeout, backoff_rate=2.0)
        if state.sequence_info is None:
            state.sequence_info = RemoteSequenceInfo.make_empty(block_uids)

        if state.sequence_info.last_updated_time is not None:
            assert block_uids == state.sequence_info.block_uids
            self._thread.ready.set()  # no need to await the first dht fetch
            self._need_latest_infos = True

    def make_sequence(
        self,
        start_index: int = 0,
        end_index: Optional[int] = None,
        *,
        mode: str,
        cache_tokens_needed: Optional[int] = None,
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

        if mode == "min_latency":
            span_sequence = self._make_sequence_with_min_latency(
                start_index, end_index, cache_tokens_needed=cache_tokens_needed
            )
        elif mode == "max_throughput":
            span_sequence = self._make_sequence_with_max_throughput(start_index, end_index)
        else:
            raise RuntimeError(f"Unexpected mode {mode}")

        if self.config.show_route is True or (mode == "min_latency" and self.config.show_route == "inference"):
            route_repr = " => ".join(
                [f"{span.start}:{span.end} via …{str(span.peer_id)[-6:]}" for span in span_sequence]
            )
            logger.info(f"Route found: {route_repr}")
        return span_sequence

    def _make_sequence_with_min_latency(
        self, start_index: int, end_index: int, *, cache_tokens_needed: Optional[int]
    ) -> List[RemoteSpanInfo]:
        if start_index == end_index:
            return []

        with self.lock_changes:
            missing_blocks = [
                block_idx
                for block_idx in range(start_index, end_index)
                if not self.state.sequence_info.spans_containing_block[block_idx]
            ]
            if missing_blocks:
                raise MissingBlocksError(missing_blocks)
            server_infos = {
                span.peer_id: span.server_info
                for block_idx in range(start_index, end_index)
                for span in self.state.sequence_info.spans_containing_block[block_idx]
            }

            graph = self._build_inference_graph(start_index, end_index, cache_tokens_needed=cache_tokens_needed)

        path = dijkstar.find_path(graph, "start", "end")
        logger.debug(f"Path info: {path}")
        if start_index == 0 and end_index == len(self):
            logger.debug(f"Expected speed: {1 / path.total_cost:.1f} steps/sec")

        span_sequence = []
        for peer_id, block_idx in path.nodes[1:-1]:
            if not span_sequence or span_sequence[-1].peer_id != peer_id:
                span_sequence.append(RemoteSpanInfo(peer_id, block_idx, block_idx, server_infos[peer_id]))
            else:
                span_sequence[-1].end = block_idx

        # Remove empty spans that can appear if we don't force to go to the end of each server and network delay
        # don't follow triangle inequality (delay(A, B) + delay(B, C) < delay(A, C)) due to measurement errors
        span_sequence = [span for span in span_sequence if span.length > 0]

        return span_sequence

    def _build_inference_graph(
        self,
        start_index: int,
        end_index: int,
        *,
        cache_tokens_needed: Optional[int],
        overhead_delay: float = 0.018,  # Serialization overhead (empirically measured)
        default_inference_rps: float = 300,  # If inference RPS unknown
        alloc_delay: float = 10,  # If not enough cache left, we penalize the edge
    ) -> dijkstar.Graph:
        missing_blocks = [
            block_idx
            for block_idx in range(start_index, end_index)
            if not self.state.sequence_info.spans_containing_block[block_idx]
        ]
        if missing_blocks:
            raise MissingBlocksError(missing_blocks)

        client_server_rtts = self.ping_aggregator.to_dict()

        graph = dijkstar.Graph()

        # Clent -> server network delays
        for span in self.state.sequence_info.spans_containing_block[start_index]:
            delay = self._rtt_to_delay(client_server_rtts.get(span.peer_id))
            delay += overhead_delay
            if not self._has_cache_for(span, cache_tokens_needed):
                delay += alloc_delay
            graph.add_edge("start", (span.peer_id, start_index), delay)

        # Server -> client network delays
        for span in self.state.sequence_info.spans_containing_block[end_index - 1]:
            delay = self._rtt_to_delay(client_server_rtts.get(span.peer_id))
            graph.add_edge((span.peer_id, end_index), "end", delay)

        # Server -> server network delays
        for block_idx in range(start_index + 1, end_index):
            for cur_span in self.state.sequence_info.spans_containing_block[block_idx - 1]:
                if cur_span.end != block_idx:
                    # If we choose a server, we force to go to the end of it before switching to a new one
                    # to avoid O(N^2) graphs for N servers
                    continue

                for next_span in self.state.sequence_info.spans_containing_block[block_idx]:
                    rtt = None
                    if cur_span.server_info.next_pings is not None:
                        rtt = cur_span.server_info.next_pings.get(next_span.peer_id.to_base58())
                    delay = self._rtt_to_delay(rtt)
                    delay += overhead_delay
                    if not self._has_cache_for(next_span, cache_tokens_needed):
                        delay += alloc_delay
                    graph.add_edge((cur_span.peer_id, block_idx), (next_span.peer_id, block_idx), delay)

        # Compute delays
        for span in self.state.sequence_info.spans_by_priority:
            for block_idx in range(max(span.start, start_index), min(span.end, end_index)):
                inference_rps = span.server_info.inference_rps
                if inference_rps is None:
                    inference_rps = default_inference_rps
                graph.add_edge((span.peer_id, block_idx), (span.peer_id, block_idx + 1), 1.0 / inference_rps)

        return graph

    @staticmethod
    def _rtt_to_delay(
        rtt: float,
        *,
        default_delay: float = 0.15,  # If network delay unknown
        max_delay: float = 5,  # If unreachable, we don't want to discard the edge completely
    ) -> float:
        if rtt is None:
            return default_delay
        return min(rtt / 2, max_delay)

    @staticmethod
    def _has_cache_for(span: RemoteSpanInfo, cache_tokens_needed: Optional[int] = None) -> bool:
        if cache_tokens_needed is None or span.server_info.cache_tokens_left is None:
            return True

        # Here, `span` contains all blocks hosted by a server - but we won't necessarily run all of them through
        # this particular server in our path. It is difficult to estimate how many blocks we'll use at this stage,
        # so we assume that we'll use all of them (the worst case for the cache size) and get a pessimistic estimate.
        # This is okay since false positives are more costly than false negatives here.
        return cache_tokens_needed * 2 * span.length <= span.server_info.cache_tokens_left

    def _make_sequence_with_max_throughput(self, start_index: int, end_index: int) -> List[RemoteSpanInfo]:
        span_sequence = []
        current_index = start_index
        while current_index < end_index:
            candidate_spans = self.state.sequence_info.spans_containing_block[current_index]
            if not candidate_spans:
                raise MissingBlocksError(current_index)

            span_weights = np.array([span.server_info.throughput for span in candidate_spans], dtype=np.float64)
            chosen_span = np.random.choice(candidate_spans, p=span_weights / span_weights.sum())

            assert chosen_span.start <= current_index < chosen_span.end
            span_sequence.append(dataclasses.replace(chosen_span, start=current_index))
            current_index = chosen_span.end
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
            self.dht, self.block_uids, active_adapter=self.config.active_adapter, latest=True
        )

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

            first_servers = [span.peer_id for span in self.state.sequence_info.spans_containing_block[0]]
            last_servers = [span.peer_id for span in self.state.sequence_info.spans_containing_block[-1]]

        pinged_servers = set(sample_up_to(first_servers, self.config.max_pinged))
        pinged_servers |= set(sample_up_to(last_servers, self.config.max_pinged))
        self.ping_aggregator.ping(list(pinged_servers), wait_timeout=self.config.ping_timeout)

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
        return dict(points=self.policy.get_points(protocol, *args, **kwargs), active_adapter=self.config.active_adapter)

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
            f"You can check the public swarm's state at https://health.petals.dev "
            f"If there are not enough servers, please connect your GPU: "
            f"https://github.com/bigscience-workshop/petals#connect-your-gpu-and-increase-petals-capacity "
        )
