from petals.client import *
from petals.utils.logging import initialize_logs as _initialize_logs
from petals.utils.mpfuture import patch_mpfuture as _patch_mpfuture

__version__ = "1.1.1"

_initialize_logs()
_patch_mpfuture()
