import time

from typing import Optional

from hivemind.utils.logging import get_logger
from huggingface_hub import HfFileSystem, hf_hub_url, get_hf_file_metadata
from peft.utils import CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME, PeftConfig
from safetensors.torch import load_file
from transformers.utils import get_file_from_repo

from petals.utils.disk_cache import allow_cache_reads, allow_cache_writes, free_disk_space_for


logger = get_logger(__name__)


def check_peft_repository(repo_id: str) -> bool:
    fs = HfFileSystem()
    list_of_files = fs.glob(f"{repo_id}/{SAFETENSORS_WEIGHTS_NAME}", detail=False)
    return len(list_of_files) > 0


def load_peft(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    use_auth_token: Optional[str] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
    delay: float = 30
):
    # TODO: Check is it possible to add safetensors loading inside petals/server/from_pretrained.py and reuse it here

    if not check_peft_repository(repo_id):
        raise ValueError(f"Repo: {repo_id} doesn't have safetensors inside for a safe loading.")
    
    try:
        with allow_cache_reads(cache_dir):
            config_path = get_file_from_repo(
                repo_id,
                CONFIG_NAME,
                revision=revision,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            assert config_path is not None
            config = PeftConfig.from_json_file(config_path)

            weight_path = get_file_from_repo(
                repo_id,
                SAFETENSORS_WEIGHTS_NAME,
                revision=revision,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            assert weight_path is not None
            return config, load_file(weight_path)
    except Exception:
        logger.warning(f"Cache for peft weights {repo_id} is corrupted, it will be downloaded again", exc_info=True)

    while True:
        try:
            with allow_cache_writes(cache_dir):
                config_url = hf_hub_url(repo_id, CONFIG_NAME, revision=revision)
                config_file_size = get_hf_file_metadata(config_url, token=use_auth_token).size
                wieght_url = hf_hub_url(repo_id, SAFETENSORS_WEIGHTS_NAME, revision=revision)
                wieght_file_size = get_hf_file_metadata(wieght_url, token=use_auth_token).size

                file_size = config_file_size + wieght_file_size
                if file_size is not None:
                    free_disk_space_for(repo_id, file_size, cache_dir=cache_dir, max_disk_space=max_disk_space)
                else:
                    logger.warning(f"Failed to fetch size from peft repo {repo_id}")

                config_path = get_file_from_repo(
                    repo_id,
                    CONFIG_NAME,
                    revision=revision,
                    use_auth_token=use_auth_token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
                if config_path is None:
                    raise RuntimeError(f"File {CONFIG_NAME} does not exist in repo {repo_id}")
                config = PeftConfig.from_json_file(config_path)

                weight_path = get_file_from_repo(
                    repo_id,
                    SAFETENSORS_WEIGHTS_NAME,
                    revision=revision,
                    use_auth_token=use_auth_token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
                if weight_path is None:
                    raise RuntimeError(f"File {SAFETENSORS_WEIGHTS_NAME} does not exist in repo {repo_id}")
                return config, load_file(weight_path)
        except Exception as e:
            logger.warning(f"Failed to load file {CONFIG_NAME} from HF Hub (retry in {delay:.0f} sec)", exc_info=True)
            time.sleep(delay)
