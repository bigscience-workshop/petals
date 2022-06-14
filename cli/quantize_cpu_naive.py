import argparse
import copy
import os

import psutil
import torch.backends.quantized
import transformers
from hivemind.utils.logging import get_logger
from tqdm.auto import trange

logger = get_logger(__file__)

DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load bloom layers and convert to 8-bit using torch quantization.")
    parser.add_argument("--output_path", required=True, type=str, help="Save quantized layers to this folder")
    parser.add_argument("--model", type=str, default="bigscience/bloom", help="Model name for from_pretrained")
    parser.add_argument("--revision", type=str, default=None, help="Optional commit id from HF hub")
    parser.add_argument("--torch_dtype", type=str, default="auto", help="Load initial model in this dtype")
    parser.add_argument("--use_auth_token", type=str, default=None, help="auth token for from_pretrained")
    args = parser.parse_args()

    free_ram_gb = psutil.virtual_memory().available / 2**30
    if free_ram_gb < 400:
        logger.warning(f"ACHTUNG! converting bloom-176b will use up 370-400GB RAM, you have {free_ram_gb:.3f} free")

    assert args.torch_dtype in DTYPE_MAP, f"torch_dtype must be one of {list(DTYPE_MAP.keys())}"
    if os.path.exists(args.output_path) and (
        len(os.listdir(args.output_path)) != 0 or not os.path.isdir(args.output_path)
    ):
        raise FileExistsError(f"Output path {args.output_path} already exists and is not an empty directory")

    model = transformers.BloomForCausalLM.from_pretrained(
        args.model, use_auth_token=args.use_auth_token, revision=args.revision, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )

    qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.backends.quantized.engine = "fbgemm"

    os.makedirs(args.output_path, exist_ok=True)

    for i in trange(len(model.transformer.h)):
        layer_fp32 = copy.deepcopy(model.transformer.h[i]).float()
        layer_quantized = torch.quantization.quantize_dynamic(
            layer_fp32, {torch.nn.Linear: qconfig}, dtype=torch.qint8, inplace=True
        )
        torch.save(layer_quantized.state_dict(), os.path.join(args.output_path, f"block_{i}_qint8.pth"))

    model.transformer.h = torch.nn.ModuleList()
    torch.save(model.state_dict(), os.path.join(args.output_path, f"client.pth"))

