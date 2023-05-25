#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import torch
from hivemind.utils.logging import get_logger
from transformers import BloomTokenizerFast

from petals import DistributedBloomForCausalLM

logger = get_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bigscience/bloom-petals")
    parser.add_argument("-i", "--initial_peers", type=str, nargs="+", required=True)
    parser.add_argument("-p", "--n_processes", type=str, required=True)
    parser.add_argument("-l", "--seq_len", type=int, required=True)
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
    tokenizer = BloomTokenizerFast.from_pretrained(args.model)
    model = DistributedBloomForCausalLM.from_pretrained(args.model, initial_peers=args.initial_peers)
    logger.info(f"Created model: {process_idx=} {model.device=}")

    result = ""
    with model.transformer.h.inference_session(max_length=args.seq_len) as sess:
        for step in range(args.seq_len):
            outputs = model.generate(max_new_tokens=1, session=sess)
            result += tokenizer.decode(outputs[0])

            if step == 0:
                start_time = perf_counter()
            else:
                speed = step / (perf_counter() - start_time)
                logger.info(f"{process_idx=} {step=} {speed=:.3f}")

    logger.info(f"Final result: {process_idx=} {speed=:.3f}")


if __name__ == "__main__":
    main()
