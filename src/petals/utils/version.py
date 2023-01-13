import requests
from hivemind.utils.logging import TextStyle, get_logger
from packaging.version import parse

import petals

logger = get_logger(__file__)


def validate_version():
    logger.info(f"Running {TextStyle.BOLD}Petals {petals.__version__}{TextStyle.RESET}")
    try:
        r = requests.get("https://pypi.python.org/pypi/petals/json")
        r.raise_for_status()
        response = r.json()

        versions = [parse(ver) for ver in response.get("releases")]
        latest = max(ver for ver in versions if not ver.is_prerelease)

        if parse(petals.__version__) < latest:
            logger.info(
                f"A newer version {latest} is available. Please upgrade with: "
                f"{TextStyle.BOLD}pip install --upgrade petals{TextStyle.RESET}"
            )
    except Exception as e:
        logger.warning("Failed to fetch the latest Petals version from PyPI:", exc_info=True)
