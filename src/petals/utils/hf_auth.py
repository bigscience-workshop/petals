import os
from typing import Union


def always_needs_auth(model_name: Union[str, os.PathLike, None]) -> bool:
    loading_from_repo = model_name is not None and not os.path.isdir(model_name)
    return loading_from_repo and model_name.startswith("meta-llama/Llama-2-")
