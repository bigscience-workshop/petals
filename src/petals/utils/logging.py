import importlib
import os

from hivemind.utils import logging as hm_logging


def in_jupyter() -> bool:
    """Check if the code is run in Jupyter or Colab"""

    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def initialize_logs():
    """Initialize Petals logging tweaks. This function is called when you import the `petals` module."""

    # Env var PETALS_LOGGING=False prohibits Petals do anything with logs
    if os.getenv("PETALS_LOGGING", "True").lower() in ("false", "0"):
        return

    if in_jupyter():
        os.environ["HIVEMIND_COLORS"] = "True"
    importlib.reload(hm_logging)

    # Remove log handlers from previous import of hivemind.utils.logging and extra handlers on Colab
    hm_logging.get_logger().handlers.clear()
    hm_logging.get_logger("hivemind").handlers.clear()

    hm_logging.use_hivemind_log_handler("in_root_logger")

    # We suppress asyncio error logs by default since they are mostly not relevant for the end user,
    # unless there is env var PETALS_ASYNCIO_LOGLEVEL
    asyncio_loglevel = os.getenv("PETALS_ASYNCIO_LOGLEVEL", "FATAL" if hm_logging.loglevel != "DEBUG" else "DEBUG")
    hm_logging.get_logger("asyncio").setLevel(asyncio_loglevel)
