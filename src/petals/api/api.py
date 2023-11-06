import argparse
import dataclasses
import uvicorn

from typing import Any

from fastapi import FastAPI, Request
from transformers import GenerationConfig, AutoTokenizer
from hivemind import get_logger

from petals.utils.auto_config import AutoDistributedModelForCausalLM

from petals.data_structures import FromPretrainedInfo

TIMEOUT_KEEP_ALIVE = 10  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
model = None
tokenizer = None


logger = get_logger(__name__)


@app.post("/generate")
async def generate(request: Request) -> dict[str, Any]:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the generation parameters (See `GenerationConfig` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]

    sampling_params = GenerationConfig(**request_dict)

    results_generator = model.generate(tokens, **sampling_params.to_diff_dict())

    if stream:
        # handle streaming case
        raise NotImplementedError("Streaming is not supported yet")

    # Non-streaming case
    assert results_generator is not None

    logger.debug(f"Generated {len(results_generator)} results")

    return {"response": tokenizer.decode(results_generator[0])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)

    # arguments for from_pretrained method

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--low_cpu_mem_usage", type=bool, default=True)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--args", type=tuple, default=tuple())

    args = parser.parse_args()

    pretrained_info = FromPretrainedInfo(
        model_name_or_path=args.model_name_or_path,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        torch_dtype=args.torch_dtype,
    )

    model = AutoDistributedModelForCausalLM.from_pretrained(*args.args, **dataclasses.asdict(pretrained_info))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, add_bos_token=False)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
