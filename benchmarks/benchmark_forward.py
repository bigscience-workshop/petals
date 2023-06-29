#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import torch
from hivemind.utils.logging import get_logger

from petals import AutoDistributedModel
from petals.constants import PUBLIC_INITIAL_PEERS

logger = get_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bigscience/bloom")
    parser.add_argument("-i", "--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS)
    parser.add_argument("-p", "--n_processes", type=str, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("--warmup_steps", type=int, default=1)
    args = parser.parse_args()

    if args.n_processes == "n_gpus":
        args.n_processes = torch.cuda.device_count()
    else:
        args.n_processes = int(args.n_processes)

    processes = [mp.Process(target=benchmark_forward, args=(i, args)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()


@torch.inference_mode()
def benchmark_forward(process_idx, args):
    model = AutoDistributedModel.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=torch.bfloat16
    )
    logger.info(f"Created model: {process_idx=} {model.device=}")

    torch.manual_seed(42)
    for step in range(args.n_steps):
        if step == args.warmup_steps:
            start_time = perf_counter()

        input_ids = torch.randint(0, model.config.vocab_size, size=(args.batch_size, args.seq_len))

        logger.info(f"{process_idx=} Fwd begin {input_ids.shape=}")
        h = model(input_ids)
        # We don't use model.lm_head
        logger.info(f"{process_idx=} Fwd end")

        if step >= args.warmup_steps:
            speed = step / (perf_counter() - start_time) * input_ids.numel()
            logger.info(f"{process_idx=} {step=} {speed=:.3f}")

    logger.info(f"Final result: {process_idx=} {speed=:.3f}")


if __name__ == "__main__":
    main()
