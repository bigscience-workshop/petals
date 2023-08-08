#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import numpy as np
import torch
from hivemind.utils.logging import get_logger

from petals import AutoDistributedModelForCausalLM, AutoDistributedModelForSequenceClassification
from petals.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="bigscience/bloom")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task", type=str, default="cls")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--n_processes", type=str, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--pre_seq_len", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--warmup_steps", type=int, default=1)
    args = parser.parse_args()

    assert args.task in ["cls", "causal_lm"]

    if args.n_processes == "n_gpus":
        args.n_processes = torch.cuda.device_count()
    else:
        args.n_processes = int(args.n_processes)

    processes = [mp.Process(target=benchmark_training, args=(i, args)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()


def benchmark_training(process_idx, args):
    if args.task == "cls":
        model = AutoDistributedModelForSequenceClassification.from_pretrained(
            args.model,
            initial_peers=args.initial_peers,
            torch_dtype=DTYPE_MAP[args.torch_dtype],
            tuning_mode="deep_ptune",
            pre_seq_len=args.pre_seq_len,
            num_labels=2,
        )
    elif args.task == "causal_lm":
        model = AutoDistributedModelForCausalLM.from_pretrained(
            args.model,
            initial_peers=args.initial_peers,
            torch_dtype=DTYPE_MAP[args.torch_dtype],
            tuning_mode="deep_ptune",
            pre_seq_len=args.pre_seq_len,
        )
    model = model.to(args.device)
    opt = torch.optim.Adam(model.parameters())
    logger.info(f"Created model: {process_idx=} {model.device=}")

    torch.manual_seed(42)
    fwd_times = []
    bwd_times = []
    for step in range(args.warmup_steps + args.n_steps):
        input_ids = torch.randint(0, model.config.vocab_size, size=(args.batch_size, args.seq_len), device=args.device)
        if args.task == "cls":
            labels = torch.randint(0, 2, size=[args.batch_size], device=args.device)
        else:
            labels = input_ids

        logger.info(f"{process_idx=} {step=} Forward")
        start_time = perf_counter()
        outputs = model(input_ids, labels=labels)
        if step >= args.warmup_steps:
            fwd_times.append(perf_counter() - start_time)

        logger.info(f"{process_idx=} {step=} Backward")
        start_time = perf_counter()
        outputs.loss.backward()
        if step >= args.warmup_steps:
            bwd_times.append(perf_counter() - start_time)

        logger.info(f"{process_idx=} {step=} Optimizer step")
        opt.step()
        opt.zero_grad()

        if step >= args.warmup_steps:
            fwd_speed = input_ids.numel() / np.mean(fwd_times)
            bwd_speed = input_ids.numel() / np.mean(bwd_times)
            logger.info(f"{process_idx=} Fwd speed: {fwd_speed:.2f} | Bwd speed: {bwd_speed:.2f}")

    logger.info(f"Final result: {process_idx=} {fwd_speed=:.2f} | {bwd_speed=:.2f}")


if __name__ == "__main__":
    main()
