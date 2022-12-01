from __future__ import annotations

import random
import threading
import time
from typing import List, Optional, Sequence, Tuple, Union

from hivemind import DHT, P2P, DHTExpiration, MSGPackSerializer
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

import petals.dht_utils
from petals.client.routing.sequence_info import RemoteSequenceInfo
from petals.client.spending_policy import NoSpendingPolicy
from petals.data_structures import ModuleUID, RemoteModuleInfo, RemoteSpanInfo, ServerState
from petals.server.handler import TransformerConnectionHandler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteSequenceManager(threading.Thread):
    """
    Sequence manager is a thread that keeps track of remote servers that hold the specified sequence of blocks.
    TL;DR it tells you, which peers you should ask to get a specific layer. It is used in RemoteSequential.
    When created, RemoteSequenceManager looks up which servers serve necessary layers by reading from DHT.
    Using this information, sequence manager can form sequences of servers that collectively have the full sequence.
    To form such a sequence, call .make_sequence with the appropriate optimization policy (see make_sequence docstr).

    :param dht: a running hivemind.DHT instance, connected to peers that serve the corresponding blocks
    :param block_uids: a sequence of DHT keys (strings) corresponding to remote layers
    :param p2p: an optional P2P replica (if not specified, create one via dht.replicate_p2p())
    :param update_period: by default, refresh DHT information once in this many seconds
    :param timeout: float, in seconds, default timeout for RPC forwad/backward/inference requests
    :param min_backoff: after a repeated failure, sleep for this many seconds times 2 ^ (num_failures - 1)
    :param max_retries: DEPRECATED. If you are reading this, yozh needs to fix the PR
    :param start: start the background thread (see the note below). If false, you will need to start it manually.
    :note: RemoteSequenceManager takes up some CPU and network I/O to operate in background. It is recommended to avoid
      running redundant sequence managers for the same set of layers.

    """

    def __init__(
        self,
        dht: DHT,
        block_uids: Sequence[ModuleUID],
        p2p: P2P,
        max_retries: int = 3,
        update_period: float = 30,
        timeout: float = 20,
        min_backoff: float = 1,
        *,  # dear dev, if you add more parameters to this class, please make sure to handle them in __getitem__ (below)
        start: bool,
    ):
        super().__init__(daemon=True)
        assert len(block_uids) > 0, "Sequences must contain at least one block"
        self.dht, self.p2p = dht, p2p
        self.sequence_info = RemoteSequenceInfo.make_empty(block_uids)
        self.spans_by_priority: List[RemoteSpanInfo] = []  # sorted from best to worst
        self.spans_containing_block: Tuple[List[RemoteSpanInfo], ...] = tuple([] for _ in range(len(self)))
        self.last_update_time: DHTExpiration = -float("inf")
        self.max_retries = max_retries
        self.update_period = update_period
        self.timeout, self.min_backoff = timeout, min_backoff
        self._rpc_info = None
        self._should_shutdown = False
        self.policy = NoSpendingPolicy()
        self.ready = threading.Event()  # TODO-USED? # whether or not you are ready to make_sequence
        self.lock_changes = threading.Lock()  # TODO-USED? # internal lock on sequence_info and strategies
        self.update_trigger = threading.Event()  # TODO-USED?

        self.update_() #TODO switch away to async update and await?
        assert self.spans_by_priority and self.spans_containing_block

        if start:
            self.run_in_background()

    def run_in_background(self, await_ready: bool = True, timeout: Optional[float] = None) -> None:
        """
        Starts averager in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready:
            self.ready.wait(timeout)

    def run(self) -> None:
        self.ready.set()

        while not self._should_shutdown:
            self.update_trigger.wait(max(0.0, min(self.update_period, time.perf_counter() - self.last_update_time)))

            if self._should_shutdown:
                logger.debug(f"{self.__class__.__name__} is shutting down")
                break

            if not self.update_trigger.is_set() and time.perf_counter() - self.last_update_time >= self.update_period:
                continue  # waited for update_period, but found that our info was already updated in the meantime

            try:
                self.update_()
                self.update_trigger.clear()
            except Exception as e:
                logger.exception(e)

        logger.info(f"{self.__class__.__name__} thread exited")

    def make_sequence(self, start_index: int = 0, end_index: Optional[int] = None) -> List[RemoteSpanInfo]:
        """
        Form a sequence of remote servers that collectively serve all consecutive layers

        :param start_index: optional index of the first module in a sequence, default = the first of block_uids
        :param end_index: optional index of the last module (non-inclusive), default = after last of block uids
        """
        if not self.is_alive():
            logger.error("Using a sequence manager that is not running: it has either crashed or never started")
        end_index = end_index if end_index is not None else len(self)
        span_sequence = []
        current_index = start_index
        while current_index < end_index:
            candidate_spans = self.spans_containing_block[current_index]
            chosen_span = random.choice(candidate_spans)  # TODO this should be replaced with proper load balancing

            assert chosen_span.start <= current_index < chosen_span.end
            span_sequence.append(RemoteSpanInfo(start=current_index, end=chosen_span.end, peer_id=chosen_span.peer_id))
            current_index = chosen_span.end

        return span_sequence

    def __getitem__(self, ix: Union[int, slice]) -> RemoteSequenceManager:
        """Get a RemoteSequenceManager for a sub-sequence of blocks"""
        assert isinstance(ix, (int, slice))
        if not isinstance(ix, slice):
            ix = slice(int(ix), int(ix) + 1, 1)
        with self.lock_changes:
            subseq = RemoteSequenceManager(
                self.dht,
                self.block_uids[ix],#TODO pass sequence info nicely
                self.p2p,
                max_retries=self.max_retries,
                update_period=self.update_period,
                timeout=self.timeout,
                min_backoff=self.min_backoff,
                start=False,
            )
            subseq.block_infos = self.sequence_info.block_infos[ix]#TODO make sure this is actually used, not overriden by init update
            subseq.spans_by_priority, subseq.spans_containing_block = subseq.compute_spans(subseq.block_infos)
            subseq.last_update_time = self.last_update_time
        return subseq

    def trigger_update(self):
        """Run an asynchronous update in background as soon as possible"""
        self.update_trigger.set()

    def update_(self):
        """Perform an immediate and synchronous refresh, may take time"""
        self.sequence_info.update_(self.dht)

        with self.lock_changes:
            self.spans_by_priority, self.spans_containing_block = self.compute_spans(self.sequence_info.block_infos)
            self.last_update_time = time.perf_counter()

    @staticmethod
    def compute_spans(block_infos: Sequence[RemoteModuleInfo]):
        closed_spans = []
        active_spans = {}
        for block_index, info in enumerate(block_infos):
            if info is not None:
                for peer_id, (server, _) in info.servers.items():
                    if server.state != ServerState.ONLINE:
                        continue
                    if peer_id not in active_spans:
                        active_spans[peer_id] = RemoteSpanInfo(start=block_index, end=block_index + 1, peer_id=peer_id)
                    else:  # peer_id in active_spans
                        active_spans[peer_id].end = block_index + 1

            for peer_id in list(active_spans.keys()):
                server_state, _ = info.servers.get(peer_id) or (None, None)
                if (
                    info is None
                    or peer_id not in info.servers
                    or server_state != ServerState.ONLINE
                    or block_index == len(block_infos) - 1
                ):
                    closed_spans.append(active_spans.pop(peer_id))
        assert not active_spans, f"spans: {active_spans}"

        closed_spans.sort(key=lambda span: span.end - span.start, reverse=True)

        spans_containing_block = tuple(list() for _ in range(len(block_infos)))
        for span in closed_spans:
            for block_index in range(span.start, span.end):
                spans_containing_block[block_index].append(span)

        return closed_spans, spans_containing_block

    def __len__(self):
        return len(self.sequence_info)

    @property
    def block_uids(self):
        return self.sequence_info.block_uids

    @property
    def rpc_info(self):
        """Return the rpc_info queried from one of the servers that hold the first block"""
        if self._rpc_info is None:
            retries = 0
            for i in range(self.max_retries):
                # TODO remove max_retries and introduce backoff
                try:
                    self.update_()
                    peer_id, _ = random.choice(list(self.sequence_info.block_infos[0].servers.items()))
                    stub = TransformerConnectionHandler.get_stub(self.p2p, peer_id)
                    outputs = RemoteExpertWorker.run_coroutine(
                        stub.rpc_info(runtime_pb2.ExpertUID(uid=self.block_uids[0]))
                    )
                    self._rpc_info = MSGPackSerializer.loads(outputs.serialized_info)
                    break
                except Exception as e:
                    retries += 1
                    if retries >= self.max_retries:
                        raise e
                    else:
                        logger.warning(f"Tried to call rpc_info, but caught {repr(e)}", exc_info=True)
        return self._rpc_info

    def get_retry_delay(self, attempt_no: int) -> float:
        if attempt_no == 0:
            return 0
        return self.min_backoff * 2 ** (attempt_no - 1)

    def get_request_metadata(self, protocol: str, *args, **kwargs) -> Optional[bytes]:
        """
        :param protocol: one of "rpc_forward", "rpc_backward" or "rpc_inference"
        :param args: request-specific inputs, typicall block uids and input tensors
        :param kwargs: additional request context, such as remote peer ID
        :returns: msgpack-serialized metadata dict that will be passed alongside a given request
        """
        return MSGPackSerializer.dumps(dict(points=self.policy.get_points(protocol, *args, **kwargs)))

    def shutdown(self):
        self._should_shutdown = True
        self.update_trigger.set()

    def __del__(self):
        if self.is_alive():
            self.shutdown()
