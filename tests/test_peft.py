import os
import pytest
import shutil

from huggingface_hub import hf_hub_download

from petals.utils import check_peft_repository, load_peft


NOSAFE_PEFT_REPO = "timdettmers/guanaco-7b"
SAFE_PEFT_REPO = "artek0chumak/guanaco-7b"
TMP_CACHE_DIR = "tmp_cache/"


def clear_dir(path_to_dir):
    shutil.rmtree(path_to_dir)
    os.mkdir(path_to_dir)


def dir_empty(path_to_dir):
    files = os.listdir(path_to_dir)
    return files.empty()


@pytest.mark.forked
def test_check_peft():
    assert not check_peft_repository(NOSAFE_PEFT_REPO), "NOSAFE_PEFT_REPO is safe to load."
    assert check_peft_repository(SAFE_PEFT_REPO), "SAFE_PEFT_REPO is not safe to load."


@pytest.mark.forked
def test_load_noncached():
    clear_dir(TMP_CACHE_DIR)
    with pytest.raises(Exception):
        load_peft(NOSAFE_PEFT_REPO, cache_dir=TMP_CACHE_DIR)
        
    assert dir_empty(TMP_CACHE_DIR), "NOSAFE_PEFT_REPO is loaded"

    status = load_peft(SAFE_PEFT_REPO, cache_dir=TMP_CACHE_DIR)
    
    assert status, "PEFT is not loaded"
    assert not dir_empty(TMP_CACHE_DIR), "SAFE_PEFT_REPO is not loaded"


@pytest.mark.forked
def test_load_cached():
    clear_dir(TMP_CACHE_DIR)
    hf_hub_download(SAFE_PEFT_REPO, cache_dir=TMP_CACHE_DIR)
    
    status = load_peft(SAFE_PEFT_REPO, cache_dir=TMP_CACHE_DIR)
    assert status, "PEFT is not loaded"
