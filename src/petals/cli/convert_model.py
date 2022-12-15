import argparse
import os

import psutil
import torch.backends.quantized
import torch.nn as nn
import transformers
from hivemind.utils.logging import get_logger
from huggingface_hub import Repository
from tqdm.auto import tqdm
from transformers.models.bloom.modeling_bloom import BloomModel

from petals.bloom.from_pretrained import BLOCK_BRANCH_PREFIX, CLIENT_BRANCH
from petals.client import DistributedBloomConfig

logger = get_logger(__file__)

DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load bloom layers and convert to 8-bit using torch quantization.")

    parser.add_argument("--model", type=str, default="bigscience/bloom-6b3", help="Model name for from_pretrained")
    parser.add_argument("--revision", type=str, default=None, help="Optional commit id from HF hub")
    parser.add_argument("--torch_dtype", type=str, default="auto", help="Load initial model in this dtype")
    parser.add_argument("--output_path", type=str, default="./converted_model", help="Track output repo to this folder")
    parser.add_argument("--output_repo", type=str, default="bigscience/test-bloomd", help="Push to this HF hub repo")
    parser.add_argument("--client_branch", type=str, default=CLIENT_BRANCH, help="Save client version to this branch")
    parser.add_argument(
        "--block_branch_prefix", type=str, default=BLOCK_BRANCH_PREFIX, help="Save blocks to branches with this prefix"
    )
    parser.add_argument(
        "--commit_message", type=str, default="push-o-matic", help="Use this commit message for all parts"
    )
    parser.add_argument("--use_auth_token", type=str, default=None, help="auth token for from_pretrained")
    parser.add_argument("--resize_token_embeddings", type=int, default=None, help="change the vocabulary size")
    args = parser.parse_args()

    free_ram_gb = psutil.virtual_memory().available / 2**30
    if args.model == "bigscience/bloom" and free_ram_gb < 400:
        logger.warning(f"ACHTUNG! converting bloom-176b will use up 350-400GB RAM, you have {free_ram_gb:.3f} free")

    assert args.torch_dtype in DTYPE_MAP, f"torch_dtype must be one of {list(DTYPE_MAP.keys())}"
    if os.path.exists(args.output_path) and (
        len(os.listdir(args.output_path)) != 0 or not os.path.isdir(args.output_path)
    ):
        raise FileExistsError(f"Output path {args.output_path} already exists and is not an empty directory")

    logger.info(f"Loading source model {args.model} (this may take a few minutes)")
    config = DistributedBloomConfig.from_pretrained(
        args.model, use_auth_token=args.use_auth_token, revision=args.revision
    )
    config.dht_prefix = args.output_repo

    model = BloomModel.from_pretrained(
        args.model, use_auth_token=args.use_auth_token, revision=args.revision, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )
    if args.resize_token_embeddings:
        logger.info(f"Resizing token embeddings, new size = {args.resize_token_embeddings}")
        model.resize_token_embeddings(args.resize_token_embeddings)
        config.vocab_size = args.resize_token_embeddings

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, use_auth_token=args.use_auth_token, revision=args.revision
    )
    os.makedirs(args.output_path, exist_ok=True)

    repo = Repository(args.output_path, clone_from=args.output_repo, use_auth_token=args.use_auth_token)
    repo.git_pull()

    transformer_blocks = model.h
    logger.info(
        f"Saving transformer blocks to {args.output_repo}@{args.block_branch_prefix}0"
        f" - {args.output_repo}@{args.block_branch_prefix}{len(transformer_blocks)}"
    )
    for i, block in enumerate(tqdm(transformer_blocks)):
        repo.git_checkout(args.client_branch, create_branch_ok=True)
        with repo.commit(
            commit_message=args.commit_message, branch=args.block_branch_prefix + str(i), track_large_files=True
        ):
            torch.save(block.state_dict(), "./pytorch_model.bin")

    logger.info(f"Saving client-side modules to {args.output_repo}@{args.client_branch}")
    repo.git_checkout(args.client_branch, create_branch_ok=True)
    with repo.commit(commit_message=args.commit_message, branch=args.client_branch, track_large_files=True):
        model.h = nn.ModuleList()
        model.save_pretrained(".")
        tokenizer.save_pretrained(".")
        config.save_pretrained(".")

    logger.info(f"Converted {args.model} and pushed to {args.output_repo}")
