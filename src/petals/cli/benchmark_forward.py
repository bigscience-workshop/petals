#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import torch
from hivemind.utils.logging import get_logger
from petals import DistributedBloomForCausalLM
from transformers import BloomTokenizerFast

logger = get_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bigscience/bloom-petals")
    parser.add_argument("--initial_peers", type=str, nargs='+', default=["/ip4/185.244.175.92/tcp/31337/p2p/QmehSoMKScoMF3HczLwaLVnw2Lgsap4bhAMrULEzGc1fSV"])
    parser.add_argument("-p", "--n_processes", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    args = parser.parse_args()

    processes = [mp.Process(target=benchmark_forward, args=(i, args,)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()


@torch.inference_mode()
def benchmark_forward(process_idx, args):
    tokenizer = BloomTokenizerFast.from_pretrained(args.model)
    model = DistributedBloomForCausalLM.from_pretrained(args.model, initial_peers=args.initial_peers, torch_dtype=torch.float32)
    logger.info(f"Created model: {process_idx=} {model.device=}")

    torch.manual_seed(42)
    for step in range(args.n_steps):
        input_ids = torch.randint(100, 10000, size=(args.batch_size, args.seq_len))
        logger.info(f"Fwd begin {input_ids.shape=}")
        outputs = model.forward(input_ids)
        logger.info("Fwd end")

        if step == 0:
            start_time = perf_counter()
        else:
            speed = step / (perf_counter() - start_time) * input_ids.numel()
            logger.info(f"{process_idx=} {step=} {speed=:.3f}")

    logger.info(f"Final result: {process_idx=} {speed=:.3f}")


if __name__ == "__main__":
    main()
