import unittest

from petals.server import block_selection
from petals.data_structures import ServerInfo, RemoteModuleInfo, RPS, ServerState
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from petals.dht_utils import get_remote_module_infos
from hivemind import DHT

from rich.pretty import pprint

class TestBlockSelection(unittest.TestCase):

    def test_should_choose_other_blocks(self):
        num_total_blocks = 24;
        num_blocks = 16;
        module_infos: List[Optional[RemoteModuleInfo]] = [None] * num_total_blocks

        server1_id = "12D3"

        for i in range(num_blocks):
            uid = f"bigscience/bloom-560m-petals.{i}"

            servers = {
                    "<libp2p.peer.id.ID (12D3)>": ServerInfo(
                    # libp2p.peer.id.ID=server1_id,
                    # offline = 0, joining = 1, online = 2
                    state=ServerState.ONLINE,
                    throughput=RPS(0),

                )
            }

            module_info = RemoteModuleInfo(
                    uid=uid,
                    servers=servers,
                    )

            module_infos[i] = module_info

        # pprint(module_infos)

        # figure out how to test this later;
# def get_remote_module_infos(
#     dht: DHT,
#     uids: Sequence[ModuleUID],
#     expiration_time: Optional[DHTExpiration] = None,
#     active_adapter: Optional[str] = None,
#     *,
#     latest: bool = False,
#     return_future: bool = False,
# ) -> Union[List[Optional[RemoteModuleInfo]], MPFuture]:
        # mod = get_remote_module_infos(server1_id, module_info)
        # pprint(module_infos)
        
        r = block_selection.should_choose_other_blocks(
                server1_id,
                module_infos,
                75
                )

        pprint(r)


        

    # module_infos are made of RemoteModuleInfo's equal to the number of blocks on the model

   # "RemoteModuleInfo(uid=""bigscience/bloom-560m-petals.12",
            # the 12 at the end of that line is the block number (zero-indexed)
   # "servers="{
            # this lists the servers that serve this block (block 12, in this case)
   #    "<libp2p.peer.id.ID (12D3KooW9r2TP8BBRPDZcsd2szZuNV3tZgydib6y6KMeen9jBz63)>":"ServerInfo(state=<ServerState.ONLINE":2>,
   #    throughput=1.0,
   #    "public_name=None",
   #    "version=""2.0.1",
   #    "network_rps=None",
   #    "forward_rps=None",
   #    "inference_rps=None",
   #    "adapters=(""artek0chumak/bloom-560m-safe-peft",
   #    ")",
   #    "torch_dtype=""float32",
   #    "quant_type=""none",
   #    "using_relay=False",
   #    cache_tokens_left=32768,
   #    "next_pings="{
   #       "12D3KooWCxzU7o1FNJHg9bgYK5ucYkVFcAQ7TDGYfxg27aUaNz9d":"inf"
   #    }")",
   #    "<libp2p.peer.id.ID (12D3KooWCxzU7o1FNJHg9bgYK5ucYkVFcAQ7TDGYfxg27aUaNz9d)>":"ServerInfo(state=<ServerState.OFFLINE":0>,
   #    throughput=1.0,
   #    "public_name=None",
   #    "version=""2.0.1",
   #    "network_rps=None",
   #    "forward_rps=None",
   #    "inference_rps=None",
   #    "adapters=(""artek0chumak/bloom-560m-safe-peft",
   #    ")",
   #    "torch_dtype=""float32",
   #    "quant_type=""none",
   #    "using_relay=False",
   #    cache_tokens_left=32768,
   #    "next_pings=None)"
   # }")",

    def test_choose_best_start(self):
        
        throughputs = np.ones(24)
        throughputs[15:] = 0
        # throughputs looks like this:
        # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        print(throughputs)
        # this is the number of blocks the local server runs
        num_blocks_local_server = 2
        start = block_selection._choose_best_start(throughputs, num_blocks_local_server)
        self.assertEqual(start, 15)
        

         
