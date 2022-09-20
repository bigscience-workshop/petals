<p align="center">
    <img src="https://i.imgur.com/7eR7Pan.png" width="400"><br>
    Decentralized platform for running 100B+ language models<br><br>
    <a href="https://github.com/bigscience-workshop/petals/actions">
        <img src="https://github.com/bigscience-workshop/petals/actions/workflows/run-tests.yaml/badge.svg?branch=main">
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>

## Key features

- Run inference or fine-tune large language models like [BLOOM-176B](https://huggingface.co/bigscience/bloom) by joining compute resources with people all over the Internet. No need to have high-end GPUs.
- It's difficult to fit the whole BLOOM-176B into GPU memory [unless](https://twitter.com/Tim_Dettmers/status/1559892918395031552) you have multiple high-end GPUs. Instead, **Petals** allows to load and serve a small part of the model, then team up with people serving all the other parts to run inference or fine-tuning.
- This way, one inference step takes ≈ 1 sec — much faster than possible with offloading. Enough for chatbots and other interactive apps.
- Beyond traditional language model APIs — you can employ any fine-tuning and sampling methods by executing custom paths through the model or accessing its hidden states. This allows for the comforts of an API with the flexibility of PyTorch.

<p align="center">
    <b><a href="https://arxiv.org/pdf/2209.01188.pdf">[Read paper]</a></b> | <b><a href="https://petals.ml/">[View website]</a></b>
</p>

## How it works?

<p align="center">
    <img src="https://i.imgur.com/RTYF3yW.png" width="800">
</p>

### 🛠️ Examples

Petals integrates seamlessly with PyTorch and the Hugging Face [Transformers](https://github.com/huggingface/transformers) library.

This snippet shows how to **(a)** generate text with BLOOM and **(b)** solve a sequence classification task via soft prompt tuning:

```python
# Initialize distributed BLOOM and connect to the swarm
model = DistributedBloomForCausalLM.from_pretrained(
    "bigscience/bloom-petals", tuning_mode="ptune", initial_peers=SEE_BELOW
)  # Embeddings & prompts are on your device, BLOOM blocks are distributed

print("Generated:", model.generate(tokenized_prefix, max_new_tokens=5))

# Training (updates only local prompts / adapters)
optimizer = torch.optim.AdamW(model.parameters())
for input_ids, labels in data_loader:
    outputs = model.forward(input_ids)
    loss = cross_entropy(outputs.logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 🚧 This project is in active development

Be careful: some features may not work, interfaces may change, and we have no detailed docs yet (see [roadmap](https://github.com/bigscience-workshop/petals/issues/12)).

A stable version of the code and a public swarm open to everyone will be released in November 2022. You can [subscribe](https://petals.ml/) to be emailed when it happens or fill in [this form](https://forms.gle/TV3wtRPeHewjZ1vH9) to help the public launch by donating GPU time. In the meantime, you can launch and use your own private swarm.

### 🔒 Privacy and security

If you work with sensitive data, you should only use a private swarm (or a subset of servers in the public swarm) hosted by people and institutions you trust, who are authorized to process this data.

This is important because it's technically possible for peers serving model layers to recover input data or model outputs. Also, if there are malicious peers, they may alter their outputs to influence the model outputs. See a more detailed discussion in Section 4 of our [paper](https://arxiv.org/pdf/2209.01188.pdf).

## FAQ

1. **What's the motivation for people to host model layers in the public swarm?**

    People who run inference and fine-tuning themselves get a certain speedup if they host a part of the model locally. Some may be also motivated to "give back" to the community helping them to run the model (similarly to how [BitTorrent](https://en.wikipedia.org/wiki/BitTorrent) users help others by sharing data they have already downloaded).

    Since it may be not enough for everyone, we are also working on introducing explicit __incentives__ ("bloom points") for people donating their GPU time to the public swarm. Once this system is ready, people who earned these points will be able to spend them on inference/fine-tuning with higher priority or increased security guarantees, or (maybe) exchange them for other rewards.

2. **Why is the platform named "Petals"?**

    "Petals" is a metaphor for people serving different parts of the model. Together, they host the entire language model &mdash; [BLOOM](https://huggingface.co/bigscience/bloom).

    While our platform focuses on BLOOM now, we aim to support more [foundation models](https://arxiv.org/abs/2108.07258) in future.

## Installation

Here's how to install the dependencies with conda:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install bitsandbytes==0.33.2  # for 8-bit quantization
pip install -r requirements.txt
```

This script uses Anaconda to install cuda-enabled PyTorch.
If you don't have anaconda, you can get it from [here](https://www.anaconda.com/products/distribution).
If you don't want anaconda, you can install PyTorch [any other way](https://pytorch.org/get-started/locally/).
If you want to run models with 8-bit weights, please install **PyTorch with CUDA 11** or newer for compatility with [bitsandbytes](https://github.com/timDettmers/bitsandbytes).

__OS support:__ currently, PETALS only supports Linux operating systems. On Windows 11, you can run PETALS with GPU enabled inside WSL2 ([read more](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).
For macOS, you can *probably* run everything normally if you manage to install dependencies, but we do not guarantee this.


### Getting Started

This is a toy example running on a local machine without GPU and with a tiny model. 
For a more detailed instruction with larger models, see ["Launch your own swarm"](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm).

First, run a couple of servers, each in a separate shell. First server runs like this
```bash
python -m cli.run_server bloom-testing/test-bloomd-560m-main --num_blocks 8 --torch_dtype float32 \
  --host_maddrs /ip4/127.0.0.1/tcp/31337   # use port 31337, local connections only
```

This server will host 8 (out of 24) layers for [this tiny bloom model](https://huggingface.co/bloom-testing/test-bloomd-560m-main) that was converted for PETALS.
To run a different model, please see [this wiki page](https://github.com/bigscience-workshop/petals/wiki/Run-a-custom-model-with-PETALS).


Once the server has started, it will print out a ton of information, including an (important) line like this:
```bash
Mon Day 01:23:45.678 [INFO] Running DHT node on ['/ip4/127.0.0.1/tcp/31337/p2p/ALongStringOfCharacters'], initial peers = []
```

You can use this address (/ip4/whatever/else) to connect additional servers. Open another terminal and run:
```bash
python -m cli.run_server bloom-testing/test-bloomd-560m-main --num_blocks 8 --torch_dtype float32 \
  --host_maddrs /ip4/127.0.0.1/tcp/0 --initial_peers /ip4/127.0...<TODO! copy the address of another server>
# e.g. --initial_peers /ip4/127.0.0.1/tcp/31337/p2p/QmS1GecIfYouAreReadingThisYouNeedToCopyYourServerAddressCBBq
```

You can assign `--initial_peers` to one or multiple addresses of other servers, not necessarily the first one.
The only requirement is that at least one of them is alive, i.e. running at the time.

Before you proceed, __please run 3 servers__ for a total of 24 blocks (3x8). If you are running a different model,
make sure your servers have enough total `--num_blocks` to cover that model. 


Once your have enough servers, you can use them to train and/or inference the model:
```python
import torch
import torch.nn.functional as F
import transformers
from src import DistributedBloomForCausalLM

initial_peers = [TODO_put_one_or_more_server_addresses_here]  # e.g. ["/ip4/127.0.0.1/tcp/more/stuff/here"]
tokenizer = transformers.BloomTokenizerFast.from_pretrained("bloom-testing/test-bloomd-560m-main")
model = DistributedBloomForCausalLM.from_pretrained(
  "bloom-testing/test-bloomd-560m-main", initial_peers=initial_peers, low_cpu_mem_usage=True, torch_dtype=torch.float32
)  # this model has only embeddings / logits, all transformer blocks rely on remote servers


inputs = tokenizer("a cat sat", return_tensors="pt")["input_ids"]
remote_outputs = model.generate(inputs, max_length=10)
print(tokenizer.decode(remote_outputs[0]))  # "a cat sat in the back of the car,"

model = DistributedBloomForCausalLM.from_pretrained(
  "bloom-testing/test-bloomd-560m-main", initial_peers=initial_peers, low_cpu_mem_usage=True, torch_dtype=torch.float32
)  # this model has only embeddings / logits, all transformer blocks rely on remote servers 

# "train" input embeddings by backprop through distributed transformer blocks
model.transformer.word_embeddings.weight.requires_grad = True
outputs = model.forward(input_ids=inputs)
loss = F.cross_entropy(outputs.logits.flatten(0, 1), inputs.flatten())
loss.backward()
print("Gradients (norm):", model.transformer.word_embeddings.weight.grad.norm())
```

Of course, this is a simplified code snippet. For actual training, see our example on "deep" prompt-tuning here: [examples/prompt-tuning-personachat.ipynb](./examples/prompt-tuning-personachat.ipynb).

Here's a [more advanced tutorial](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm) that covers 8-bit quantization and best practices for running PETALS.

### Development

PETALS uses pytest with a few plugins. To install them, run `pip install -r requirements-dev.txt`

To run minimalistic tests, spin up some servers:
```bash
export MODEL_NAME=bloom-testing/test-bloomd-560m-main
export INITIAL_PEERS=/ip4/127.0.0.1/tcp/31337/p2p/QmS9KwZptnVdB9FFV7uGgaTq4sEKBwcYeKZDfSpyKDUd1g
python -m cli.run_server $MODEL_NAME --block_indices 0:12 --throughput 1 --torch_dtype float32 \
  --identity tests/test.id --host_maddrs /ip4/127.0.0.1/tcp/31337  &> server1.log &
sleep 5  # wait for the first server to initialize DHT
python -m cli.run_server $MODEL_NAME --block_indices 12:24 --throughput 1 --torch_dtype float32 \
  --initial_peers /ip4/127.0.0.1/tcp/31337/p2p/QmS9KwZptnVdB9FFV7uGgaTq4sEKBwcYeKZDfSpyKDUd1g &> server2.log &

tail -f server1.log server2.log  # view logs for both servers
# after you're done, kill servers with 'pkill -f cli.run_server'
```

Then launch pytest:
```
export MODEL_NAME=bloom-testing/test-bloomd-560m-main REF_NAME=bigscience/bloom-560m
export INITIAL_PEERS=/ip4/127.0.0.1/tcp/31337/p2p/QmS9KwZptnVdB9FFV7uGgaTq4sEKBwcYeKZDfSpyKDUd1g
PYTHONPATH=. pytest tests --durations=0 --durations-min=1.0 -v
```

The automated tests use a more complex server configuration that can be found [here](https://github.com/bigscience-workshop/petals/blob/main/.github/workflows/run-tests.yaml)  

We use [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) and [isort](https://pycqa.github.io/isort/) for all pull requests.
Before commiting your code, simply run `black . && isort .` and you will be fine.



--------------------------------------------------------------------------------

<p align="center">
    This project is a part of the <a href="https://bigscience.huggingface.co/">BigScience</a> research workshop.
</p>
<p align="center">
    <img src="https://petals.ml/bigscience.png" width="150">
</p>
