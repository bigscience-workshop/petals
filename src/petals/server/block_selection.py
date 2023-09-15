from typing import Dict, List

import numpy as np
from hivemind import PeerID, get_logger

from petals.data_structures import RemoteModuleInfo, RemoteSpanInfo, ServerState
from petals.utils.dht import compute_spans

logger = get_logger(__name__)


def compute_throughputs(spans: Dict[PeerID, RemoteSpanInfo], *, total_blocks: int) -> np.ndarray:
    # We sort servers here to ensure that we get exactly the same throughputs for a given set of servers.
    # If the order were not defined, we would get slightly different values due to floating point errors,
    # which may cause excess block replacements.

    throughputs = np.zeros(total_blocks)
    for span in sorted(spans.values(), key=lambda span: span.peer_id):
        throughputs[span.start : span.end] += span.throughput
    return throughputs


def _choose_best_start(throughputs: np.ndarray, num_blocks: int) -> int:
    options = ((sorted(throughputs[i : i + num_blocks]), i) for i in range(0, len(throughputs) - num_blocks + 1))
    return min(options)[-1]


def choose_best_blocks(num_blocks: int, module_infos: List[RemoteModuleInfo]) -> List[int]:
    spans = compute_spans(module_infos, min_state=ServerState.JOINING)
    throughputs = compute_throughputs(spans, total_blocks=len(module_infos))

    start = _choose_best_start(throughputs, num_blocks)
    return list(range(start, start + num_blocks))


def _move_span(span: RemoteSpanInfo, new_start: int):
    span.start, span.end = new_start, new_start + span.length


def should_choose_other_blocks(
    local_peer_id: PeerID, module_infos: List[RemoteModuleInfo], balance_quality: float
) -> bool:
    if balance_quality > 1.0:
        return True  # Forces rebalancing on each check (may be used for debugging purposes)

    spans = compute_spans(module_infos, min_state=ServerState.JOINING)
    throughputs = compute_throughputs(spans, total_blocks=len(module_infos))
    initial_throughput = throughputs.min()
    eps = 1e-3

    assert local_peer_id in spans, "Span served by this server is not present in the DHT"
    local_span = spans[local_peer_id]
    throughputs[local_span.start : local_span.end] -= local_span.throughput * (1 + eps)
    # Without (1 + eps) here, we would sometimes subtract a value slightly less than local_span.throughput
    # due to the floating point error, which would cause excess block replacements.
    # Also, subtracting local_span.throughput * (1 + eps) makes _choose_best_start() prefer
    # the previous server position in case of other things being almost equal.

    if initial_throughput > eps and throughputs.min() <= 0:
        return False  # Switching blocks would make the swarm disjoint

    new_start = _choose_best_start(throughputs, local_span.length)
    if local_span.start == new_start:
        return False  # This server is on its best place already

    throughputs[local_span.start : local_span.end] += local_span.throughput * eps
    _move_span(local_span, new_start)
    throughputs[local_span.start : local_span.end] += local_span.throughput

    moved = True
    while moved:
        servers = list(spans.keys())
        np.random.shuffle(servers)

        moved = False
        for peer_id in servers:
            span = spans[peer_id]
            throughputs[span.start : span.end] -= span.throughput * (1 + eps)

            new_start = _choose_best_start(throughputs, span.length)

            throughputs[span.start : span.end] += span.throughput * eps
            if span.start != new_start:
                _move_span(span, new_start)
                moved = True
            throughputs[span.start : span.end] += span.throughput

    new_throughput = throughputs.min()
    if new_throughput < initial_throughput or new_throughput < eps:
        return False

    actual_quality = initial_throughput / new_throughput
    logger.info(f"Swarm balance quality: {actual_quality * 100:.1f}%")

    return actual_quality < balance_quality - eps
