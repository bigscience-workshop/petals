import os

import hivemind

from petals.client import *
from petals.utils.logging import initialize_logs as _initialize_logs

__version__ = "1.1.5"


def _override_bfloat16_mode_default():
    if os.getenv("USE_LEGACY_BFLOAT16") is None:
        hivemind.compression.base.USE_LEGACY_BFLOAT16 = False


_initialize_logs()
_override_bfloat16_mode_default()
