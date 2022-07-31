#!/usr/bin/env bash

source ~/miniconda3/etc/profile.d/conda.sh
# If you use anaconda, uncomment:
# source ~/anaconda3/etc/profile.d/conda.sh

if conda env list | grep ".*bloom-demo-benchmark.*"  >/dev/null 2>/dev/null; then
    conda activate bloom-demo-benchmark
else
    conda create -y --name bloom-demo-benchmark python=3.8.12 pip
    conda activate bloom-demo-benchmark

    conda install -y -c conda-forge cudatoolkit-dev==11.3.1 cudatoolkit==11.3.1 cudnn==8.2.1.32
    pip install -i https://pypi.org/simple torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

    pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda113
    pip install -i https://pypi.org/simple -r demo-requirements.txt
fi

# Please set up
INITIAL_PEER="/ip4/172.27.77.65/tcp/38457/p2p/QmWCiRzNYhtSUdPT3toMjFpG9BWPMrrce4WYGWCaWqrESV"
MODEL_NAME="bigscience/test-bloomd"
HOST_MADDR="/ip4/0.0.0.0/tcp/30000"
SERVER_ID_PATH="./server.id"
GPU_ID="0" # GPU must have Tensor Cores: RTX, Titan, A100, V100, ...
NUM_BLOCKS="3" # one converted block consumes ~3.5Gb 

export OMP_NUM_THREADS="16" # just in case
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m cli.run_server --converted_model_name_or_path ${MODEL_NAME} --torch_dtype float16 --initial_peer ${INITIAL_PEER} \
                                                        --compression BLOCKWISE_8BIT --identity_path ${SERVER_ID_PATH} --host_maddrs ${HOST_MADDR} \
                                                        --num_blocks ${NUM_BLOCKS} --load_in_8bit