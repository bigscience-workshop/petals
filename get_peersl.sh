docker logs petals-backbone-1 2>&1  |grep initial_peers |cut "-d " -f18-  | sort -u > peers.txt
