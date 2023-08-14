import warnings

warnings.warn(
    "petals.dht_utils has been moved to petals.utils.dht. This alias will be removed in Petals 2.2.0+",
    DeprecationWarning,
    stacklevel=2,
)

from petals.utils.dht import *
