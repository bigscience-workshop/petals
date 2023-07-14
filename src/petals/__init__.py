import os

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

import hivemind
import transformers
from packaging import version

from petals.client import *
from petals.models import *
from petals.utils import *
from petals.utils.logging import initialize_logs as _initialize_logs

__version__ = "1.2.0.dev2"


if not os.getenv("PETALS_IGNORE_DEPENDENCY_VERSION"):
    assert (
        version.parse("4.30.1") <= version.parse(transformers.__version__) < version.parse("5.0.0")
    ), "Please install a proper transformers version: pip install transformers>=4.30.1,<5.0.0"


def _override_bfloat16_mode_default():
    if os.getenv("USE_LEGACY_BFLOAT16") is None:
        hivemind.compression.base.USE_LEGACY_BFLOAT16 = False


_initialize_logs()
_override_bfloat16_mode_default()
