from typing import List, Optional

from src.data_structures import RemoteModuleInfo, ServerState


def choose_best_blocks(num_blocks: int, remote_module_infos: List[Optional[RemoteModuleInfo]]) -> List[int]:
    throughputs = []
    for module in remote_module_infos:
        if module is None:
            throughputs.append(0)
            continue
        throughputs.append(
            sum(server.throughput for server in module.servers.values() if server.state != ServerState.OFFLINE)
        )

    options = [(sorted(throughputs[i : i + num_blocks]), i) for i in range(0, len(throughputs) - num_blocks + 1)]
    best_start = min(options)[1]
    return list(range(best_start, best_start + num_blocks))
