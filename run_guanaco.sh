#! /bin/bash

export MODEL_NAME="Enoch/llama-65b-hf"
export ADAPTER_NAME="artek0chumak/guanaco-65b"

# export MODEL_NAME="bigscience/bloom-560m"
# export ADAPTER_NAME="artek0chumak/bloom-560m-safe-peft"

python -m petals.cli.run_server --converted_model_name_or_path $MODEL_NAME --block_indices 0:80 \
            --new_swarm --identity tests/test.id --host_maddrs /ip4/127.0.0.1/tcp/31337 --throughput 1 \
            --torch_dtype bfloat16 --compression NONE --attn_cache_tokens 2048 \
            --adapters $ADAPTER_NAME \
            # --quant_type none
