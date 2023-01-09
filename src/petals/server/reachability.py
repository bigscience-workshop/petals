import math
import time

import requests
from hivemind.utils.logging import get_logger

logger = get_logger(__file__)


def check_reachability(peer_id, wait_time: float = 7 * 60, retry_delay: float = 15) -> None:
    for attempt_no in range(math.floor(wait_time / retry_delay) + 1):
        try:
            r = requests.get(f"http://health.petals.ml/api/v1/is_reachable/{peer_id}", timeout=10)
            r.raise_for_status()
            response = r.json()

            if response["success"]:
                logger.info("Server is reachable from the Internet. It will appear at http://health.petals.ml soon")
                return

            if attempt_no == 0:
                # Usually, libp2p manages to set up relays before we finish loading blocks.
                # In other cases, we may need to wait for up to `wait_time` seconds before it's done.
                logger.info("Detected a NAT or a firewall, connecting to libp2p relays. This takes a few minutes")
            time.sleep(retry_delay)
        except Exception as e:
            logger.warning(f"Skipping reachability check because health.petals.ml is down: {repr(e)}")
            return

    raise RuntimeError(
        f"Server has not become reachable from the Internet:\n\n"
        f"{response['message']}\n\n"
        f"You need to fix your port forwarding and/or firewall settings. How to do that:\n\n"
        f"    1. Choose a specific port for the Petals server, for example, 31337.\n"
        f"    2. Ensure that this port is accessible from the Internet and not blocked by your firewall.\n"
        f"    3. Add these arguments to explicitly announce your IP address and port to other peers:\n"
        f"        python -m petals.cli.run_server ... --public_ip {response['your_ip']} --port 31337\n"
        f"    4. If it does not help, ask for help in our Discord: https://discord.gg/Wuk8BnrEPH\n"
    )
