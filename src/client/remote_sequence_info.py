from __future__ import annotations

import dataclasses
import threading
from functools import partial
from typing import Tuple, List, Optional, Sequence, NamedTuple

from hivemind import DHT, PeerID
from hivemind.utils.logging import use_hivemind_log_handler, get_logger

from src.data_structures import ModuleUID, RemoteModuleInfo
from src.dht_utils import _get_remote_module_infos

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


Span = NamedTuple("Span", [("start", int), ("end", Optional[int]), ("peer_id", PeerID)])


@dataclasses.dataclass(frozen=False, init=False)  # TODO[borzunov@] eto ne dataclass
class RemoteSequenceInfo:
    """Keeps and updates the meta-information about which peers host which blocks"""

    dht: DHT
    block_uids: List[ModuleUID, ...]
    block_infos: List[Optional[RemoteModuleInfo], ...]
    spans_by_priority: List[Span]  # sorted from best to worst
    spans_containing_block: Tuple[List[Span], ...]
    lock_changes: threading.Lock

    def __init__(self, dht: DHT, block_uids: Sequence[ModuleUID]):
        self.dht = dht
        self.block_uids = list(block_uids)
        self.block_infos: List[Optional[RemoteModuleInfo], ...] = [None] * len(self.block_uids)
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
        new_block_infos: Sequence[RemoteModuleInfo] = self.dht.run_coroutine(
            partial(_get_remote_module_infos, uids=self.block_uids, expiration_time=float("inf")), return_future=False
        )
        assert len(new_block_infos) == len(self.block_uids)
        for block_index, (uid, info) in enumerate(zip(self.block_uids, new_block_infos)):
            if info is None:
                logger.warning(f"Found no block info for block {uid}")
            if not isinstance(info, RemoteModuleInfo):
                logger.warning(f"Unexpected dht entry type for {uid}: {info}")
            if not info.peer_ids:
                logger.warning(f"Found no active peers for block {uid}")
            if info.uid != uid:
                logger.warning(f"The DHT entry for {uid} actually points to {info.uid}")
            if not isinstance(info.peer_ids, set):
                logger.warning(f"Expected peer_ids for {uid} to be a set, got {type(info.peer_ids)}")
            self.block_infos[block_index] = info

    @staticmethod
    def compute_spans(block_infos: Sequence[RemoteModuleInfo]):
        closed_spans = []
        active_spans = {}
        for block_index, info in enumerate(block_infos):
            for peer_id in info.peer_ids:
                if peer_id not in active_spans:
                    active_spans[peer_id] = Span(start=block_index, end=block_index + 1, peer_id=peer_id)
                else:  # peer_id in active_spans
                    active_spans[peer_id] = active_spans[peer_id]._replace(end=block_index + 1)

            for peer_id in list(active_spans.keys()):
                if peer_id not in info.peer_ids or block_index == len(block_infos) - 1:
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
