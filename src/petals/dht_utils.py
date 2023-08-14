import warnings

warnings.warn(
    "petals.dht_utils has been moved to petals.utils.dht. The old name will be removed in Petals 2.1.0+",
    DeprecationWarning,
    stacklevel=2,
)

from petals.utils.dht import *
