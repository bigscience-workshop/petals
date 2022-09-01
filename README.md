<p align="center">
    <img src="https://i.imgur.com/7eR7Pan.png" width="500"><br>
    Decentralized platform for running 100B+ language models<br><br>
    <a href="https://github.com/bigscience-workshop/petals/actions">
        <img src="https://github.com/bigscience-workshop/petals/actions/workflows/run-tests.yaml/badge.svg?branch=main">
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>

## Key features

- Run inference or fine-tune [BLOOM-176B](https://huggingface.co/bigscience/bloom) by joining compute resources with people all over the Internet. No need to have high-end GPUs.
- One inference step takes ≈ 1 sec — much faster than possible with offloading. Enough for chatbots and other interactive apps.
- Employ any fine-tuning and sampling methods by accessing model's hidden states and changing its control flow — something you can't do in proprietary APIs.

<p align="center">
    <b><a href="https://petals.ml/petals.pdf">[Read paper]</a></b> | <b><a href="https://petals.ml/">[View website]</a></b>
</p>

## How it works?

<p align="center">
    <img src="https://i.imgur.com/75LFA0Y.png" width="800">
</p>

### 🚧 This project is in active development

Be careful: some features may not work, interfaces may change, and we have no detailed docs yet (see [roadmap](https://github.com/bigscience-workshop/petals/issues/12)).

A stable version of the code and a public swarm open to everyone will be released in November 2022. You can [subscribe](https://petals.ml/) to be emailed when it happens or fill in [this form](https://forms.gle/TV3wtRPeHewjZ1vH9) to help the public launch by donating GPU time. In the meantime, you can launch and use your own private swarm.

## Code examples

Solving a sequence classification task via soft prompt tuning of BLOOM-176B:

```python
# Initialize distributed BLOOM with soft prompts
model = AutoModelForPromptTuning.from_pretrained(
       "bigscience/distributed-bloom")
# Define optimizer for prompts and linear head
optimizer = torch.optim.AdamW(model.parameters())

for input_ids, labels in data_loader:
    # Forward pass with local and remote layers
    outputs = model.forward(input_ids)
    loss = cross_entropy(outputs.logits, labels)

    # Distributed backward w.r.t. local params
    loss.backward() # Compute model.prompts.grad
    optimizer.step() # Update local params only
    optimizer.zero_grad()
```

## Installation

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

--------------------------------------------------------------------------------

<p align="center">
    This project is a part of the <a href="https://bigscience.huggingface.co/">BigScience</a> research workshop.
</p>
<p align="center">
    <img src="https://petals.ml/bigscience.png" width="150">
</p>
