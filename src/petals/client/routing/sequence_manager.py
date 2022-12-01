from __future__ import annotations

import itertools
import logging
import random
import threading
import time
from typing import List, Optional, Sequence, Union
from weakref import WeakMethod

from hivemind import DHT, P2P, MSGPackSerializer
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

import petals.dht_utils
from petals.client.routing.sequence_info import RemoteSequenceInfo
from petals.client.routing.spending_policy import NoSpendingPolicy
from petals.data_structures import ModuleUID, RemoteSpanInfo
from petals.server.handler import TransformerConnectionHandler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteSequenceManager:
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
    :param sequence_info: optionally, specify pre-generated sequence info. by default, create a new one using dht
    :param rpc_info: optionally, specify rpc info (communicated tensor shapes and compression) to save time
    :param start: start the background thread (see the note below). If false, you will need to start it manually.
    :note: RemoteSequenceManager takes up some CPU and network I/O to operate in background. It is recommended to avoid
      running redundant sequence managers for the same set of layers.

    """

    def __init__(
        self,
        dht: DHT,
        block_uids: Sequence[ModuleUID],
        p2p: P2P,
        update_period: float = 30,
        timeout: float = 20,
        min_backoff: float = 1,
        sequence_info: Optional[RemoteSequenceInfo] = None,
        rpc_info: Optional[dict] = None,
        *,  # dear dev, if you add more parameters to this class, please make sure to handle them in __getitem__ (below)
        start: bool,
    ):
        assert len(block_uids) > 0, "Sequences must contain at least one block"
        self.dht, self.p2p = dht, p2p
        self.timeout, self.min_backoff = timeout, min_backoff
        self.lock_changes = threading.Lock()
        self._thread = _SequenceManagerUpdateThread(update_period, WeakMethod(self.update_))
        self.policy = NoSpendingPolicy()
        self._rpc_info = rpc_info

        if sequence_info is None:
            self.sequence_info = RemoteSequenceInfo.make_empty(block_uids)
            self.trigger_update()
        else:
            self.sequence_info = sequence_info
            assert block_uids == sequence_info.block_uids
            self._thread.ready.set()  # no need to await the first dht fetch

        if start:
            self.run_in_background()

    def run_in_background(self, await_ready: bool = True, timeout: Optional[float] = None) -> None:
        """
        Starts the updater thread in a background. if await_ready, this method will wait until sequence manager
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self._thread.start()
        if await_ready:
            self._thread.ready.wait(timeout)

    def make_sequence(self, start_index: int = 0, end_index: Optional[int] = None) -> List[RemoteSpanInfo]:
        """
        Form a sequence of remote servers that collectively serve all consecutive layers

        :param start_index: optional index of the first module in a sequence, default = the first of block_uids
        :param end_index: optional index of the last module (non-inclusive), default = after last of block uids
        """
        if not self.is_alive():
            logger.error("Using a sequence manager that is not running: it has either crashed or never started")
        if not self.ready.is_set():
            logger.warning("Remote SequenceManager is still searching for routes, waiting for it to become ready")
            self.ready.wait()

        end_index = end_index if end_index is not None else len(self)
        span_sequence = []
        current_index = start_index
        while current_index < end_index:
            candidate_spans = self.sequence_info.spans_containing_block[current_index]
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
        return type(self)(
            self.dht,
            self.block_uids[ix],
            self.p2p,
            update_period=self._thread.update_period,
            timeout=self.timeout,
            min_backoff=self.min_backoff,
            sequence_info=self.sequence_info[ix],
            rpc_info=self._rpc_info,
            start=True,
        )

    def trigger_update(self):
        """Run an asynchronous update in background as soon as possible"""
        self._thread.trigger.set()

    def update_(self):
        """Perform an immediate and synchronous refresh, may take time"""
        for attempt_no in itertools.count():
            new_block_infos = petals.dht_utils.get_remote_module_infos(
                self.dht, self.block_uids, expiration_time=float("inf")
            )
            with self.lock_changes:
                self.sequence_info.update_(new_block_infos)
            missing_blocks = [i for i in range(len(self)) if not self.sequence_info.spans_containing_block[i]]
            if not missing_blocks:
                self.ready.set()  # if there is an active server for every block, we may begin running
                break
            else:
                delay = self.get_retry_delay(attempt_no)
                logger.warning(f"Could not find blocks {missing_blocks} (retry in {delay:.0f} sec)")
                time.sleep(delay)

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
        return self.sequence_info.block_uids

    @property
    def rpc_info(self):
        """Return the rpc_info queried from one of the servers that hold the first block"""
        if self._rpc_info is None:
            for attempt_no in itertools.count():
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
                    delay = self.get_retry_delay(attempt_no)
                    logger.warning(
                        f"Caught exception when gathering information from peer {peer_id} "
                        f"(retry in {delay:.0f} sec): {repr(e)}"
                    )
                    traceback_level = logging.DEBUG if str(e) else logging.WARNING
                    logger.log(traceback_level, "See detailed traceback below:", exc_info=True)
                    time.sleep(delay)

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
        self._thread.shutdown()


class _SequenceManagerUpdateThread(threading.Thread):
    def __init__(self, update_period: float, ref_update_manager: WeakMethod):
        super().__init__(daemon=True)
        self.ref_update_manager = ref_update_manager
        self.ready = threading.Event()
        self.trigger = threading.Event()
        self.last_update_time = -float("inf")
        self.update_period = update_period
        self.should_shutdown = False

    def run(self) -> None:
        while not self.should_shutdown:

            self.trigger.wait(max(0.0, min(self.update_period, time.perf_counter() - self.last_update_time)))

            if self.should_shutdown:
                logger.debug(f"{self.__class__.__name__} is shutting down")
                break

            if not self.trigger.is_set() and time.perf_counter() - self.last_update_time >= self.update_period:
                continue  # waited for update_period, but found that our info was already updated in the meantime

            update_manager = self.ref_update_manager()
            if update_manager is None:
                logger.debug(f"{self.__class__.__name__} exited because the sequence manager no longer exists")
                break

            try:
                update_manager()
                self.trigger.clear()
            except Exception as e:
                logger.exception(e)
            finally:
                del update_manager

        logger.info(f"{self.__class__.__name__} thread exited")

    def shutdown(self, timeout: Optional[float] = None):
        self.should_shutdown = True
        self.trigger.set()
        self.join(timeout)

    def __del__(self):
        if self.is_alive():
            self.shutdown()
