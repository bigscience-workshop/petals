# bloom-demo
Early dev prototype for decentralized bloom. Not for public eyes **yet**.

```python
if you.read(this) and you.name not in '@timdettmers @borzunov @mryab @greenfatguy'.split():
  you.go("away")
```



# install


```bash
conda create -y --name bloom-demo python=3.8.12 pip
conda activate bloom-demo

conda install -y -c conda-forge cudatoolkit-dev==11.3.1 cudatoolkit==11.3.1 cudnn==8.2.1.32
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install accelerate==0.10.0 huggingface-hub==0.7.0
pip install bitsandbytes-cuda113==0.26.0
pip install https://github.com/learning-at-home/hivemind/archive/master.zip
pip install https://github.com/huggingface/transformers/archive/6589e510fa4e6c442059de2fab84752535de9b23.zip
```


# tests

```bash
# run one bloom block for a few steps
python -m cli.inference_one_block --config cli/config.json  # see other args


# convert model from HF hub to a distributed format (can take hours depending on your connection!)
MY_WRITE_TOKEN=TODO_WRITE_TOKEN_FROM_https://huggingface.co/settings/token
python -m cli.convert_model --model bigscience/bloom-6b3  \
  --output_path ./converted_model --output_repo bigscience/test-bloomd-6b3 \
  --use_auth_token $MY_WRITE_TOKEN  # ^-- todo replace output repo with something you have access to


# minimalistic server with non-trained bloom blocks
python -m cli.run_server --prefix smol --block_config bigscience/bloom-6b3 --num_blocks 2 \
  --identity_path ./server1.id --host_maddrs /ip4/127.0.0.1/tcp/31337
# when running multiple servers:
# - give each server a unique --identity_path (or remote --identity_path arg when debugging)
# - if running multiple servers on the same machine, give each a unique port (last integer in --host_maddrs, 0 means random port)
# - when running over the internet, change --host_maddrs according to https://learning-at-home.readthedocs.io/en/latest/user/dht.html#running-across-the-internet
# - each server except first should have --initial_peers pointing to one of pre-existing servers 
```