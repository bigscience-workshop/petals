import os

from hivemind.utils import logging as hm_logging


def initialize_logs():
    """Initialize Petals logging tweaks. This function is called when you import the `petals` module."""

    # Env var PETALS_LOGGING=False prohibits Petals do anything with logs
    if os.getenv("PETALS_LOGGING", "True").lower() in ("false", "0"):
        return

    hm_logging.use_hivemind_log_handler("in_root_logger")

    # We suppress asyncio error logs by default since they are mostly not relevant for the end user,
    # unless there is env var PETALS_ASYNCIO_LOGLEVEL
    asyncio_loglevel = os.getenv("PETALS_ASYNCIO_LOGLEVEL", "FATAL" if hm_logging.loglevel != "DEBUG" else "DEBUG")
    hm_logging.get_logger("asyncio").setLevel(asyncio_loglevel)
