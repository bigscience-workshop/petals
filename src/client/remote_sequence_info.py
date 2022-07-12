from __future__ import annotations

import threading
from typing import List, NamedTuple, Optional, Sequence, Tuple

from hivemind import DHT, PeerID
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from src.data_structures import ModuleUID, RemoteModuleInfo, ServerState
from src.dht_utils import get_remote_module_infos

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


Span = NamedTuple("Span", [("start", int), ("end", Optional[int]), ("peer_id", PeerID)])


class RemoteSequenceInfo:
    """Keeps and updates the meta-information about which peers host which blocks"""

    dht: DHT
    block_uids: List[ModuleUID]
    block_infos: List[Optional[RemoteModuleInfo]]
    spans_by_priority: List[Span]  # sorted from best to worst
    spans_containing_block: Tuple[List[Span]]
    lock_changes: threading.Lock

    def __init__(self, dht: DHT, block_uids: Sequence[ModuleUID]):
        self.dht = dht
        self.block_uids = list(block_uids)
        self.block_infos = [None] * len(self.block_uids)
        self.spans_by_priority = []
        self.spans_containing_block = tuple(list() for _ in range(len(self.block_uids)))
        self.lock_changes = threading.Lock()
        self.update_()

        for uid, info in zip(self.block_uids, self.block_infos):
            assert info is not None, f"Found no remote peers for block {uid}"
        assert self.spans_by_priority and self.spans_containing_block

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
                    active_spans[peer_id] = Span(start=block_index, end=block_index + 1, peer_id=peer_id)
                else:  # peer_id in active_spans
                    active_spans[peer_id] = active_spans[peer_id]._replace(end=block_index + 1)

            for peer_id in list(active_spans.keys()):
                if (
                    peer_id not in info.servers or
                    info.servers[peer_id].state != ServerState.ONLINE or
                    block_index == len(block_infos) - 1
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
