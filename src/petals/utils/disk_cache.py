import os
from pathlib import Path

DEFAULT_CACHE_DIR = os.getenv("PETALS_CACHE", Path(Path.home(), ".cache", "petals"))
