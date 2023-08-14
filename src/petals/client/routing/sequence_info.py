import dataclasses
import time
from typing import Iterable, List, Optional, Sequence, Tuple, Type, TypeVar

from hivemind import get_logger

from petals.data_structures import ModuleUID, RemoteModuleInfo, RemoteSpanInfo, ServerState

logger = get_logger(__name__)


T = TypeVar("T")


@dataclasses.dataclass
class RemoteSequenceInfo:
    """
    A dataclass that stores general information about which servers hold any given layer;
    - updated by RemoteSequenceManager in a background thread
    - accessed by routing strategies in .on_update
    :note: this class should *not* be modified by RoutingStrategy.on_update to avoid interference between strategies;
     Any metadata specific to one routing strategy, it should be stored inside that strategy. Any information that
     is used by most routing strategies should be moved from said strategies to this class.
    """

    block_uids: Tuple[ModuleUID, ...]
    block_infos: Tuple[RemoteModuleInfo, ...]  # note: the contents of RemoteModuleInfo can and will be updated
    spans_by_priority: List[RemoteSpanInfo]
    spans_containing_block: Tuple[List[RemoteSpanInfo], ...]
    last_updated_time: Optional[float]

    @classmethod
    def make_empty(cls: Type[T], block_uids: Iterable[ModuleUID]) -> T:
        block_uids = tuple(block_uids)
        empty_block_infos = tuple(RemoteModuleInfo(uid, {}) for uid in block_uids)
        empty_spans = tuple([] for _ in range(len(block_uids)))
        return cls(block_uids, empty_block_infos, [], empty_spans, last_updated_time=None)

    def __getitem__(self, ix: slice):
        assert isinstance(ix, slice)
        block_uids, block_infos = self.block_uids[ix], self.block_infos[ix]
        spans_by_priority, spans_containing_block = self.compute_spans(block_infos)
        return RemoteSequenceInfo(
            block_uids, block_infos, spans_by_priority, spans_containing_block, self.last_updated_time
        )

    def __len__(self):
        return len(self.block_uids)

    def update_(self, new_block_infos: List[Optional[RemoteModuleInfo]]):
        assert len(new_block_infos) == len(self.block_uids)
        for block_index, (uid, info) in enumerate(zip(self.block_uids, new_block_infos)):
            if info is None:
                logger.debug(f"Found no block info for block {uid}")
                continue
            if not isinstance(info, RemoteModuleInfo):
                logger.warning(f"Unexpected dht entry type for {uid}: {info}")
                continue
            if not info.servers:
                logger.debug(f"Found no active peers for block {uid}")
                continue
            if info.uid != uid:
                logger.warning(f"The DHT entry for {uid} actually points to {info.uid}")
                continue
            self.block_infos[block_index].servers = info.servers

        self.spans_by_priority, self.spans_containing_block = self.compute_spans(self.block_infos)
        self.last_updated_time = time.perf_counter()

    @staticmethod
    def compute_spans(block_infos: Sequence[RemoteModuleInfo]):
        closed_spans = []
        active_spans = {}
        for block_index, info in enumerate(block_infos):
            if info is not None:
                for peer_id, server_info in info.servers.items():
                    if server_info.state != ServerState.ONLINE:
                        continue
                    if peer_id not in active_spans:
                        active_spans[peer_id] = RemoteSpanInfo(
                            peer_id=peer_id,
                            start=block_index,
                            end=block_index + 1,
                            server_info=server_info,
                        )
                    else:  # peer_id in active_spans
                        active_spans[peer_id].end = block_index + 1

            for peer_id in list(active_spans.keys()):
                if (
                    info is None
                    or peer_id not in info.servers
                    or info.servers[peer_id].state != ServerState.ONLINE
                    or block_index == len(block_infos) - 1
                ):
                    closed_spans.append(active_spans.pop(peer_id))
        assert not active_spans, f"spans: {active_spans}"

        closed_spans.sort(key=lambda span: span.length, reverse=True)

        spans_containing_block = tuple(list() for _ in range(len(block_infos)))
        for span in closed_spans:
            for block_index in range(span.start, span.end):
                spans_containing_block[block_index].append(span)

        return closed_spans, spans_containing_block
