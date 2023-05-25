#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import numpy as np
import torch
import petals.client.sequential_autograd
from hivemind.utils.logging import get_logger
from petals import DistributedBloomForSequenceClassification, DistributedBloomForCausalLM
from transformers import BloomTokenizerFast

logger = get_logger()

petals.client.sequential_autograd.MAX_TOKENS_IN_BATCH = 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bigscience/bloom-petals")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task", type=str, default="cls")
    parser.add_argument("-i", "--initial_peers", type=str, nargs='+', required=True)
    parser.add_argument("--n_processes", type=str, default="1")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--pre_seq_len", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    args = parser.parse_args()

    assert args.task in ["cls", "causal_lm"]

    if args.n_processes == "n_gpus":
        args.n_processes = torch.cuda.device_count()
    else:
        args.n_processes = int(args.n_processes)

    processes = [mp.Process(target=benchmark_training, args=(i, args,)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()


def benchmark_training(process_idx, args):
    tokenizer = BloomTokenizerFast.from_pretrained(args.model)
    if args.task == "cls":
        model = DistributedBloomForSequenceClassification.from_pretrained(
            args.model, initial_peers=args.initial_peers, tuning_mode="deep_ptune",
            pre_seq_len=args.pre_seq_len, num_labels=2)
    elif args.task == "causal_lm":
        model = DistributedBloomForCausalLM.from_pretrained(
            args.model, initial_peers=args.initial_peers, tuning_mode="deep_ptune",
            pre_seq_len=args.pre_seq_len)
    model = model.to(args.device)
    opt = torch.optim.Adam(model.parameters())
    logger.info(f"Created model: {process_idx=} {model.device=}")

    torch.manual_seed(42)
    fwd_times = []
    bwd_times = []
    for step in range(args.n_steps):
        input_ids = torch.randint(100, 10000, size=(args.batch_size, args.seq_len), device=args.device)
        if args.task == "cls":
            labels = torch.randint(0, 2, size=[args.batch_size], device=args.device)
        else:
            labels = input_ids

        logger.info(f"{process_idx=} {step=} Forward")
        start_time = perf_counter()
        outputs = model(input_ids, labels=labels)
        fwd_times.append(perf_counter() - start_time)

        logger.info(f"{process_idx=} {step=} Backward")
        start_time = perf_counter()
        outputs.loss.backward()
        bwd_times.append(perf_counter() - start_time)

        logger.info(f"{process_idx=} {step=} Optimizer step")
        opt.step()
        opt.zero_grad()

        if step >= 1:
            fwd_speed = input_ids.numel() / np.mean(fwd_times[1:])
            bwd_speed = input_ids.numel() / np.mean(bwd_times[1:])
            logger.info(f"{process_idx=} Fwd speed: {fwd_speed:.2f} | Bwd speed: {bwd_speed:.2f}")

    logger.info(f"Final result: {process_idx=} {fwd_speed=:.2f} | {bwd_speed=:.2f}")


if __name__ == "__main__":
    main()
