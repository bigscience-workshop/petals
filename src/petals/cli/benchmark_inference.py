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
    parser.add_argument("-p", "--n_processes", type=int)
    parser.add_argument("-l", "--seq_len", type=int)
    args = parser.parse_args()

    processes = [mp.Process(target=benchmark_inference, args=(i, args,)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()


@torch.inference_mode()
def benchmark_inference(process_idx, args):
    tokenizer = BloomTokenizerFast.from_pretrained(args.model)
    model = DistributedBloomForCausalLM.from_pretrained(args.model)
    logger.info(f"Created model: {process_idx=} {model.device=}")

    result = ""
    with model.transformer.h.inference_session(max_length=args.seq_len) as sess:
        for step in range(args.seq_len):
            outputs = model.generate(max_new_tokens=1, session=sess)
            result += tokenizer.decode(outputs[0])

            if step == 0:
                start_time = perf_counter()
            else:
                average_time = (perf_counter() - start_time) / step
                logger.info(f"{process_idx=} {step=} {average_time=:.3f}")

    logger.info(f"Final result: {process_idx=} {average_time=:.3f}")


if __name__ == "__main__":
    main()
