#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import numpy as np
import torch
from hivemind.utils.logging import get_logger

from petals import AutoDistributedModel
from petals.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS, help="Initial peers")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Torch dtype")
    parser.add_argument("--n_processes", type=str, default=1, help="Number of concurrent processes")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--n_steps", type=int, default=100, help="Number of benchmark steps")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps")
    args = parser.parse_args()

    if args.n_processes == "n_gpus":
        args.n_processes = torch.cuda.device_count()
    else:
        args.n_processes = int(args.n_processes)

    pipe_recv, pipe_send = mp.Pipe(duplex=False)
    processes = [mp.Process(target=benchmark_forward, args=(i, args, pipe_send)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    speed = np.mean([pipe_recv.recv() for _ in range(args.n_processes)])
    logger.info(f"Final result: {speed=:.2f}")


@torch.inference_mode()
def benchmark_forward(process_idx, args, result_pipe):
    model = AutoDistributedModel.from_pretrained(
        args.model,
        initial_peers=args.initial_peers,
        torch_dtype=DTYPE_MAP[args.torch_dtype],
    )
    logger.info(f"Created model: {process_idx=} {model.device=}")

    torch.manual_seed(42)
    step_times = []
    for step in range(args.warmup_steps + args.n_steps):
        start_time = perf_counter()

        input_ids = torch.randint(0, model.config.vocab_size, size=(args.batch_size, args.seq_len))

        logger.info(f"{process_idx=} Fwd begin {input_ids.shape=}")
        h = model(input_ids)
        # We don't use model.lm_head
        logger.info(f"{process_idx=} Fwd end")

        if step >= args.warmup_steps:
            step_times.append(perf_counter() - start_time)
            speed = input_ids.numel() / np.mean(step_times)
            logger.info(f"{process_idx=} {step=} {speed=:.2f}")

    result_pipe.send(speed)


if __name__ == "__main__":
    main()
