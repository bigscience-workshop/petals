from __future__ import annotations

import random
import threading
from typing import List, Optional, Sequence, Tuple, Union

from hivemind import DHT, DHTExpiration
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from src.data_structures import ModuleUID, RemoteModuleInfo, RemoteSpanInfo, ServerState
from src.dht_utils import get_remote_module_infos

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteSequenceManager:
    """Keeps and updates the meta-information about which peers host which blocks"""

    dht: DHT
    block_uids: List[ModuleUID]
    block_infos: List[Optional[RemoteModuleInfo]]
    spans_by_priority: List[RemoteSpanInfo]  # sorted from best to worst
    spans_containing_block: Tuple[List[RemoteSpanInfo], ...]
    last_update_time: DHTExpiration
    lock_changes: threading.Lock

    def __init__(self, dht: DHT, block_uids: Sequence[ModuleUID]):
        self.dht = dht
        self.block_uids = list(block_uids)
        self.block_infos = [None] * len(self.block_uids)
        self.spans_by_priority = []
        self.spans_containing_block = tuple(list() for _ in range(len(self.block_uids)))
        self.last_update_time = -float("inf")
        self.lock_changes = threading.Lock()
        self.update_()

        for uid, info in zip(self.block_uids, self.block_infos):
            assert info is not None, f"Found no remote peers for block {uid}"
        assert self.spans_by_priority and self.spans_containing_block

    def make_sequence(self, start_index: int = 0, end_index: Optional[int] = None) -> Sequence[RemoteSpanInfo]:
        """
        Form a sequence of remote servers that collectively serve all consecutive layers

        :param start_index: optional index of the first module in a sequence, default = the first of block_uids
        :param end_index: optional index of the last module (non-inclusive), default = after last of block uids
        """
        end_index = end_index if end_index is not None else len(self.block_uids)
        span_sequence = []
        current_index = start_index
        while current_index != end_index - 1:
            candidate_spans = self.spans_containing_block[current_index]

            chosen_span = random.choice(candidate_spans)  # TODO this should be replaced with proper load balancing

            assert chosen_span.start <= current_index < chosen_span.end
            span_sequence.append(chosen_span)

        return span_sequence

    def __getitem__(self, ix: Union[int, slice]) -> RemoteSequenceManager:
        """Get a RemoteSequenceManager for a sub-sequence of blocks"""
        assert isinstance(ix, (int, slice))
        if not isinstance(ix, slice):
            ix = slice(int(ix), int(ix) + 1, 1)
        with self.lock_changes:
            subseq = RemoteSequenceManager(self.dht, self.block_uids[ix])
            subseq.block_infos = self.block_infos[ix]
            subseq.spans_by_priority, subseq.spans_containing_block = subseq.compute_spans(subseq.block_infos)
            subseq.last_update_time = self.last_update_time
        return subseq

    def update_(self):
        with self.lock_changes:
            self.update_block_infos_()
            self.spans_by_priority, self.spans_containing_block = self.compute_spans(self.block_infos)

    def update_block_infos_(self):
        new_block_infos = get_remote_module_infos(self.dht, self.block_uids, expiration_time=float("inf"))
        assert len(new_block_infos) == len(self.block_uids)
        for block_index, (uid, info) in enumerate(zip(self.block_uids, new_block_infos)):
            if info is None:
                logger.warning(f"Found no block info for block {uid}")
            if not isinstance(info, RemoteModuleInfo):
                logger.warning(f"Unexpected dht entry type for {uid}: {info}")
            if not info.servers:
                logger.warning(f"Found no active peers for block {uid}")
            if info.uid != uid:
                logger.warning(f"The DHT entry for {uid} actually points to {info.uid}")
            self.block_infos[block_index] = info

    @staticmethod
    def compute_spans(block_infos: Sequence[RemoteModuleInfo]):
        closed_spans = []
        active_spans = {}
        for block_index, info in enumerate(block_infos):
            for peer_id, server in info.servers.items():
                if server.state != ServerState.ONLINE:
                    continue
                if peer_id not in active_spans:
                    active_spans[peer_id] = RemoteSpanInfo(start=block_index, end=block_index + 1, peer_id=peer_id)
                else:  # peer_id in active_spans
                    active_spans[peer_id].end = block_index + 1

            for peer_id in list(active_spans.keys()):
                if (
                    peer_id not in info.servers
                    or info.servers[peer_id].state != ServerState.ONLINE
                    or block_index == len(block_infos) - 1
                ):
                    closed_spans.append(active_spans.pop(peer_id))
        assert not active_spans

        closed_spans.sort(key=lambda span: span.end - span.start, reverse=True)

        spans_containing_block = tuple(list() for _ in range(len(block_infos)))
        for span in closed_spans:
            for block_index in range(span.start, span.end):
                spans_containing_block[block_index].append(span)

        return closed_spans, spans_containing_block

    def __len__(self):
        return len(self.block_uids)
