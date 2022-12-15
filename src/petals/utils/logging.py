import importlib
import os

from hivemind.utils import logging as _logging


def in_jupyter() -> bool:
    """Check if the code is run in Jupyter or Colab"""

    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def initialize():
    """Initialize Petals logging tweaks, called when you import the `petals` module."""

    # Env var PETALS_LOGGING=False prohibits Petals do anything with logs
    if os.getenv("PETALS_LOGGING", "True").lower() in ("false", "0"):
        return

    if in_jupyter():
        os.environ["HIVEMIND_COLORS"] = "True"
    importlib.reload(_logging)

    _logging.get_logger().handlers.clear()  # Remove extra default handlers on Colab
    _logging.use_hivemind_log_handler("in_root_logger")

    # We suppress asyncio error logs by default since they are mostly not relevant for the end user,
    # unless there is env var PETALS_ASYNCIO_LOGLEVEL
    asyncio_loglevel = os.getenv("PETALS_ASYNCIO_LOGLEVEL", "FATAL" if _logging.loglevel != "DEBUG" else "DEBUG")
    _logging.get_logger("asyncio").setLevel(asyncio_loglevel)
