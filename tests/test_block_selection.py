import unittest
import codecs

from petals.server import block_selection
from petals.server.block_selection import Span
from hivemind.p2p import PeerID
from petals.data_structures import ServerInfo, RemoteModuleInfo, RPS, ServerState
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from petals.dht_utils import get_remote_module_infos
from hivemind import DHT

from rich.pretty import pprint

class TestBlockSelection(unittest.TestCase):

    def test_llama65b_infinite_restart_01(self):
        num_total_blocks = 80
        module_infos: List[Optional[RemoteModuleInfo]] = [None] * num_total_blocks

        server1_id = "12D3"
        server1_id_bytes = codecs.encode(server1_id, 'utf-8')
        server1_PeerID = PeerID(server1_id_bytes)

        server2_id = "hj23"
        server2_id_bytes = codecs.encode(server2_id, 'utf-8')
        server2_PeerID = PeerID(server2_id_bytes)

        server3_id = "12Yh3"
        server3_id_bytes = codecs.encode(server3_id, 'utf-8')
        server3_PeerID = PeerID(server3_id_bytes)

        server4_id = "542D3"
        server4_id_bytes = codecs.encode(server4_id, 'utf-8')
        server4_PeerID = PeerID(server4_id_bytes)

        server5_id = "1hk2D3"
        server5_id_bytes = codecs.encode(server5_id, 'utf-8')
        server5_PeerID = PeerID(server5_id_bytes)

        server6_id = "5412D3"
        server6_id_bytes = codecs.encode(server6_id, 'utf-8')
        server6_PeerID = PeerID(server6_id_bytes)

        server7_id = "dft12D3"
        server7_id_bytes = codecs.encode(server7_id, 'utf-8')
        server7_PeerID = PeerID(server7_id_bytes)

        servers = {}

        servers[0] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[1] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[2] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }
        
        servers[3] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[4] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[5] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[6] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[7] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
        }

        servers[8] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[9] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[10] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[11] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[12] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[13] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[14] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
        }

        servers[15] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[16] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[17] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[18] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
        }

        servers[19] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server3_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1723.5781832870298),
            ),
        }

        servers[20] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
        }

        servers[21] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server4_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(3154.063460812297),
            ),
        }

        servers[22] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[23] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server4_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(3154.063460812297),
            ),
        }

        servers[24] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server4_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(3154.063460812297),
            ),
        }

        servers[25] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[26] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[27] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server4_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(3154.063460812297),
            ),
        }

        servers[28] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server4_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(3154.063460812297),
            ),
        }

        servers[29] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server4_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(3154.063460812297),
            ),
        }


        servers[30] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
        }

        servers[31] = {
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[32] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server1_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(762.939453125),
            ),
        }

        servers[33] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server4_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(3154.063460812297),
            ),
        }

        servers[34] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server4_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(3154.063460812297),
            ),
        }

        servers[35] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[36] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[37] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server4_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(3154.063460812297),
            ),
        }

        servers[38] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[39] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[40] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[41] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[42] = {
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[43] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[44] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[45] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }


        servers[46] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[47] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server6_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1843.87298726553),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[48] = {
            server6_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1843.87298726553),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[49] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server6_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1843.87298726553),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[50] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server6_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1843.87298726553),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[51] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server6_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1843.87298726553),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[52] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server6_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1843.87298726553),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }
        servers[53] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server6_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1843.87298726553),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[54] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server6_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1843.87298726553),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[55] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
            server6_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1843.87298726553),
            ),
        }

        servers[56] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server6_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(1843.87298726553),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[57] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[58] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[59] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server7_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2203.28331063272),
            ),
        }

        servers[60] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }


        servers[61] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[62] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[63] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[64] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[65] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[66] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }


        servers[67] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[68] = {
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[69] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[70] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[71] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[72] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }


        servers[73] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[74] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[75] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[76] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[77] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        servers[78] = {
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
        }

        servers[79] = {
            server2_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(696.5283411730552),
            ),
            server5_PeerID: ServerInfo(
                state=ServerState.ONLINE,
                throughput=RPS(2235.7226389647903),
            ),
        }

        for i in range(len(servers)):
            uid = f"llama-65b-hf.{i}"
            module_info = RemoteModuleInfo(
                    uid=uid,
                    servers=servers[i],
                    )
            module_infos[i] = module_info

        r = block_selection.should_choose_other_blocks(
                server1_PeerID, #local_peer_id 
                module_infos,
                .75
                )
        print(r)

        spans, throughputs = block_selection.compute_spans(module_infos)
        print(throughputs)


        eps = 1e-3
        local_span = spans[server1_PeerID]
        throughputs[local_span.start : local_span.end] -= local_span.throughput * (1 + eps)
        new_start = block_selection._choose_best_start(throughputs, local_span.length)
        print(throughputs)


    def test_choose_best_start(self):
        
        throughputs = np.ones(24)
        throughputs[15:] = 0
        # throughputs looks like this:
        # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        # print(throughputs)
        # num_blocks_local_server is the number of blocks the local server runs:
        num_blocks_local_server = 2
        start = block_selection._choose_best_start(throughputs, num_blocks_local_server)
        self.assertEqual(start, 15)


    def test_should_choose_other_blocks(self):
        num_total_blocks = 24;
        num_blocks = 16;
        module_infos: List[Optional[RemoteModuleInfo]] = [None] * num_total_blocks

        server1_id = "12D3"
        server1_id_bytes = codecs.encode(server1_id, 'utf-8')
        server1_PeerID = PeerID(server1_id_bytes)
        
        server2_id = "43WB"
        server2_id_bytes = codecs.encode(server2_id, 'utf-8')
        server2_PeerID = PeerID(server2_id_bytes)

        for i in range(num_blocks):
            uid = f"bigscience/bloom-560m-petals.{i}"

            # Dict[PeerID, ServerInfo]
            servers = {
                server1_PeerID: ServerInfo(
                    state=ServerState.ONLINE,
                    throughput=RPS(1),
                ),

                server2_PeerID: ServerInfo(
                    state=ServerState.ONLINE,
                    throughput=RPS(1),
                ),
            }

            module_info = RemoteModuleInfo(
                    uid=uid,
                    servers=servers,
                    )

            module_infos[i] = module_info


        # pprint(module_infos)
        spans, throughputs = block_selection.compute_spans(module_infos)

        throughputs_t = np.ones(24)
        throughputs_t[:16] = 2
        throughputs_t[16:] = 0
        
        np.testing.assert_array_equal(throughputs, throughputs_t)

        eps = 1e-3
        local_span = spans[server1_PeerID]
        throughputs[local_span.start : local_span.end] -= local_span.throughput * (1 + eps)
        new_start = block_selection._choose_best_start(throughputs, local_span.length)
        # there are 24 blocks and the local server can serve 16, so should start at 8: 
        self.assertEqual(8, new_start) 
        
        r = block_selection.should_choose_other_blocks(
                server1_PeerID, #local_peer_id 
                module_infos,
                .75
                )

        # pprint(r)
        self.assertTrue(r)


        

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

        

         
