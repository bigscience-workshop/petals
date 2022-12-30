<p align="center">
    <img src="https://i.imgur.com/7eR7Pan.png" width="400"><br>
    Run 100B+ language models at home, BitTorrent-style.<br>
    Fine-tuning and inference up to 10x faster than offloading<br><br>
</p>

Generate text using distributed BLOOM and fine-tune it for your own tasks:

```python
from petals import DistributedBloomForCausalLM

model = DistributedBloomForCausalLM.from_pretrained("bigscience/bloom-petals", tuning_mode="ptune", pre_seq_len=16)
# Embeddings & prompts are on your device, BLOOM blocks are distributed across the Internet

inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))  # A cat sat on a mat...

# Fine-tuning (updates only prompts or adapters hosted locally)
optimizer = torch.optim.AdamW(model.parameters())
for input_ids, labels in data_loader:
    outputs = model.forward(input_ids)
    loss = cross_entropy(outputs.logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

<p align="center">
    🚀 &nbsp;<b><a href="https://colab.research.google.com/drive/1Ervk6HPNS6AYVr3xVdQnY5a-TjjmLCdQ?usp=sharing">Try now in Colab</a></b>
</p>

Connect your own GPU and increase Petals capacity:

```bash
# In an Anaconda env
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/bigscience-workshop/petals
python -m petals.cli.run_server bigscience/bloom-petals

# Or using our GPU-enabled Docker image
sudo docker run --net host --ipc host --gpus all --volume petals-cache:/cache --rm \
    learningathome/petals:main python -m petals.cli.run_server bigscience/bloom-petals
```

💬 If you have any issues or feedback, please join [our Discord server](https://discord.gg/D9MwApKgWa)!

Check out more examples and tutorials:

- Chatbot web app: [link](http://chat.petals.ml), [source code](https://github.com/borzunov/petals-chat)
- Training a personified chatbot: [notebook](./examples/prompt-tuning-personachat.ipynb)
- Fine-tuning BLOOM for text semantic classification: [notebook](./examples/prompt-tuning-sst2.ipynb)
- Launching your own swarm: [tutorial](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm)
- Running a custom foundation model: [tutorial](https://github.com/bigscience-workshop/petals/wiki/Run-a-custom-model-with-Petals)

## How does it work?

- Petals runs large language models like BLOOM-176B **collaboratively** — you load a small part of the model, then team up with people serving the other parts to run inference or fine-tuning.
- Inference runs at ≈ 1 sec per step (token) — 10x faster than possible with offloading, enough for chatbots and other interactive apps. Parallel inference reaches hundreds of tokens/sec.
- Beyond classic language model APIs — you can employ any fine-tuning and sampling methods by executing custom paths through the model or accessing its hidden states. You get the comforts of an API with the flexibility of PyTorch.

<p align="center">
    <img src="https://i.imgur.com/RTYF3yW.png" width="800">
</p>

<p align="center">
    📜 &nbsp;<b><a href="https://arxiv.org/pdf/2209.01188.pdf">Read paper</a></b>
</p>

### 🔒 Privacy and security

The Petals public swarm is designed for research and academic use. **Please do not use the public swarm to process sensitive data.** We ask for that because it is an open network, and it is technically possible for peers serving model layers to recover input data and model outputs or modify them in a malicious way. Instead, you can [set up a private Petals swarm](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm) hosted by people and organization you trust, who are authorized to process your data. We discuss privacy and security in more detail [here](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety).

### 📋 Model's terms of use

Before building your own application that runs a language model with Petals, please check out the model's **terms of use, risks, and limitations**. In case of BLOOM, they are described in its [model card](https://huggingface.co/bigscience/bloom) and [license](https://huggingface.co/spaces/bigscience/license).

## FAQ

1. **What's the motivation for people to host model layers in the public swarm?**

    People who run inference and fine-tuning themselves get a certain speedup if they host a part of the model locally. Some may be also motivated to "give back" to the community helping them to run the model (similarly to how [BitTorrent](https://en.wikipedia.org/wiki/BitTorrent) users help others by sharing data they have already downloaded).

    Since it may be not enough for everyone, we are also working on introducing explicit __incentives__ ("bloom points") for people donating their GPU time to the public swarm. Once this system is ready, people who earned these points will be able to spend them on inference/fine-tuning with higher priority or increased security guarantees, or (maybe) exchange them for other rewards.

2. **Why is the platform named "Petals"?**

    "Petals" is a metaphor for people serving different parts of the model. Together, they host the entire language model &mdash; [BLOOM](https://huggingface.co/bigscience/bloom).

    While our platform focuses on BLOOM now, we aim to support more [foundation models](https://arxiv.org/abs/2108.07258) in future.

## Installation

Here's how to install Petals with conda:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/bigscience-workshop/petals
```

This script uses Anaconda to install cuda-enabled PyTorch.
If you don't have anaconda, you can get it from [here](https://www.anaconda.com/products/distribution).
If you don't want anaconda, you can install PyTorch [any other way](https://pytorch.org/get-started/locally/).
If you want to run models with 8-bit weights, please install **PyTorch with CUDA 11** or newer for compatility with [bitsandbytes](https://github.com/timDettmers/bitsandbytes).

__System requirements:__ Petals only supports Linux for now. If you don't have a Linux machine, consider running Petals in Docker (see our [image](https://hub.docker.com/r/learningathome/petals)) or, in case of Windows, in WSL2 ([read more](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)). CPU is enough to run a client, but you probably need a GPU to run a server efficiently.

## 🛠️ Development

Petals uses pytest with a few plugins. To install them, run:

```python
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
git clone https://github.com/bigscience-workshop/petals.git && cd petals
pip install -e .[dev]
```

To run minimalistic tests, you need to make a local swarm with a small model and some servers. You may find more information about how local swarms work and how to run them in [this tutorial](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm).

```bash
export MODEL_NAME=bloom-testing/test-bloomd-560m-main

python -m petals.cli.run_server $MODEL_NAME --block_indices 0:12 \
  --identity tests/test.id --host_maddrs /ip4/127.0.0.1/tcp/31337 --new_swarm  &> server1.log &
sleep 5  # wait for the first server to initialize DHT

python -m petals.cli.run_server $MODEL_NAME --block_indices 12:24 \
  --initial_peers SEE_THE_OUTPUT_OF_THE_1ST_PEER &> server2.log &

tail -f server1.log server2.log  # view logs for both servers
```

Then launch pytest:

```
export MODEL_NAME=bloom-testing/test-bloomd-560m-main REF_NAME=bigscience/bloom-560m
export INITIAL_PEERS=/ip4/127.0.0.1/tcp/31337/p2p/QmS9KwZptnVdB9FFV7uGgaTq4sEKBwcYeKZDfSpyKDUd1g
PYTHONPATH=. pytest tests --durations=0 --durations-min=1.0 -v
```

After you're done, you can terminate the servers and ensure that no zombie processes are left with `pkill -f petals.cli.run_server && pkill -f p2p`.

The automated tests use a more complex server configuration that can be found [here](https://github.com/bigscience-workshop/petals/blob/main/.github/workflows/run-tests.yaml).

### Code style

We use [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) and [isort](https://pycqa.github.io/isort/) for all pull requests.
Before committing your code, simply run `black . && isort .` and you will be fine.

--------------------------------------------------------------------------------

<p align="center">
    This project is a part of the <a href="https://bigscience.huggingface.co/">BigScience</a> research workshop.
</p>
<p align="center">
    <img src="https://petals.ml/bigscience.png" width="150">
</p>
