#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import torch
from hivemind.utils.logging import get_logger
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM
from petals.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

logger = get_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bigscience/bloom")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--n_processes", type=str, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=1)
    args = parser.parse_args()

    if args.n_processes == "n_gpus":
        args.n_processes = torch.cuda.device_count()
    else:
        args.n_processes = int(args.n_processes)

    processes = [mp.Process(target=benchmark_inference, args=(i, args)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()


@torch.inference_mode()
def benchmark_inference(process_idx, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )
    logger.info(f"Created model: {process_idx=} {model.device=} {model.config.torch_dtype=}")

    result = ""
    with model.transformer.h.inference_session(max_length=args.seq_len) as sess:
        for step in range(args.seq_len):
            if step == args.warmup_steps:
                start_time = perf_counter()

            outputs = model.generate(max_new_tokens=1, session=sess)
            result += tokenizer.decode(outputs[0])

            if step >= args.warmup_steps:
                speed = step / (perf_counter() - start_time)
                logger.info(f"{process_idx=} {step=} {speed=:.3f}")

    logger.info(f"Final result: {process_idx=} {speed=:.3f}")


if __name__ == "__main__":
    main()
