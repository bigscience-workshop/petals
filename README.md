# PETALS: Collaborative Inference of Large Models

Run BLOOM-176B, the largest open language model, by collaborating over the Internet.

__[EARLY PROTOTYPE]__ - this project is a work in progress. Stuff breaks and gets fixed every day. Docs are nonexistent.
If you want us to wake you up when it's ready, click Watch -> Custom and tick "Releases".

Roadmap: [__Issue #12__](https://github.com/learning-at-home/bloom-demo/issues/12)

### Installation

```bash
conda install -y -c conda-forge cudatoolkit-dev==11.3.1 cudatoolkit==11.3.1 cudnn==8.2.1.32
pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda113
```


### Basic functionality

All tests is run on localhost

First, run one or more servers like this:
```bash
# minimalistic server with non-trained bloom blocks
python -m cli.run_server --converted_model_name_or_path bigscience/test-bloomd-6b3 \
  --block_indices 3:5 --torch_dtype float32 --identity_path ./server1.id --host_maddrs /ip4/127.0.0.1/tcp/31337
# when running multiple servers:
# - give each server a unique --identity_path (or remote --identity_path arg when debugging)
# - if running multiple servers on the same machine, give each a unique port (last integer in --host_maddrs, 0 means random port)
# - when running over the internet, change --host_maddrs according to https://learning-at-home.readthedocs.io/en/latest/user/dht.html#running-across-the-internet
# - each server except first should have --initial_peers pointing to one of pre-existing servers
```

Then open a python notebook or console and run:
```python
import torch
import hivemind
from src import DistributedBloomConfig, get_remote_module


dht = hivemind.DHT(
    initial_peers=[TODO_COPY_FULL_ADDRESS_FROM_ANY_OF_THE_SERVERS],  # e.g. /ip4/127.0.0.1/...
    client_mode=True, start=True,
)
config = DistributedBloomConfig.from_pretrained("bigscience/test-bloom-6b3")
layer3, layer4 = get_remote_module(dht, ['bigscience/test-bloomd-6b3.3', 'bigscience/test-bloomd-6b3.4'], config)
assert layer3 is not None and layer4 is not None, "one or both layers were not found in DHT"
# test forward/backward, two blocks
outputs = layer4(layer3(torch.randn(1, 64, 4096)))
loss = (outputs * torch.randn_like(outputs)).norm()
loss.backward()

# test inference, one block
with layer3.inference_session(max_length=10) as sess:
    for i in range(10):
        res = sess.step(torch.ones(1, 1, 4096))
```


### Convert regular BLOOM into distributed
```bash

# convert model from HF hub to a distributed format (can take hours depending on your connection!)
MY_WRITE_TOKEN=TODO_WRITE_TOKEN_FROM_https://huggingface.co/settings/token
python -m cli.convert_model --model bigscience/bloom-6b3  \
  --output_path ./converted_model --output_repo bigscience/test-bloomd-6b3 \
  --use_auth_token $MY_WRITE_TOKEN  # ^-- todo replace output repo with something you have access to
```


### Test local vs remote block (allclose)

To test distributed inference, run one or more servers, then open a new shell and run pytest with environment variables:
```bash
# shell A: serve model
python -m cli.run_server --converted_model_name_or_path bigscience/test-bloomd-6b3 \
  --torch_dtype float32 --identity_path ./server1.id --host_maddrs /ip4/127.0.0.1/tcp/31337

# shell B:
export PYTHONPATH=.
export INITIAL_PEERS="/ip4/TODO_COPY_INITIAL_PEERS_FROM_SERVER_OUTPUT"
export MODEL_NAME="bigscience/test-bloomd-6b3"

# test individual random blocks for exact match
pytest tests/test_block_exact_match.py

# test the full model
pytest tests/test_full_model.py
```
