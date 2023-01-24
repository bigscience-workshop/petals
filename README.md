<p align="center">
    <img src="https://i.imgur.com/7eR7Pan.png" width="400"><br>
    Run 100B+ language models at home, BitTorrent-style.<br>
    Fine-tuning and inference up to 10x faster than offloading<br><br>
    <a href="https://pypi.org/project/petals/"><img src="https://img.shields.io/pypi/v/petals.svg?color=green"></a><br>
</p>

Generate text using distributed 176B-parameter [BLOOM](https://huggingface.co/bigscience/bloom) or [BLOOMZ](https://huggingface.co/bigscience/bloomz) and fine-tune them for your own tasks:

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

🔏 Your data will be processed by other people in the public swarm. Learn more about privacy [here](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety). For sensitive data, you can set up a [private swarm](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm) among people you trust.

### Connect your GPU and increase Petals capacity

Run this in an [Anaconda](https://www.anaconda.com) env:

```bash
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U petals
python -m petals.cli.run_server bigscience/bloom-petals
```

Or use our [Docker](https://www.docker.com) image:

```bash
sudo docker run --net host --ipc host --gpus all --volume petals-cache:/cache --rm \
    learningathome/petals:main python -m petals.cli.run_server bigscience/bloom-petals
```

You can also host [BLOOMZ](https://huggingface.co/bigscience/bloomz), a version of BLOOM fine-tuned to follow human instructions in the zero-shot regime — just replace `bloom-petals` with `bloomz-petals`.

🔒 This does not allow others to run custom code on your computer. Learn more about security [here](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety).

💬 If you have any issues or feedback, let us know on [our Discord server](https://discord.gg/D9MwApKgWa)!

### Check out tutorials, examples, and more

Basic tutorials:

- Getting started: [tutorial](https://colab.research.google.com/drive/1Ervk6HPNS6AYVr3xVdQnY5a-TjjmLCdQ?usp=sharing)
- Prompt-tune BLOOM to create a personified chatbot: [tutorial](https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-personachat.ipynb)
- Prompt-tune BLOOM for text semantic classification: [tutorial](https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb)

Example apps built with Petals:

- [Chatbot web app](http://chat.petals.ml) (connects to Petals via an HTTP endpoint): [source code](https://github.com/borzunov/chat.petals.ml)

Useful tools and advanced guides:

- [Monitor](http://health.petals.ml) for the public swarm: [source code](https://github.com/borzunov/health.petals.ml)
- Launch your own swarm: [guide](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm)
- Run a custom foundation model: [guide](https://github.com/bigscience-workshop/petals/wiki/Run-a-custom-model-with-Petals)

📋 If you build an app running BLOOM with Petals, make sure it follows the BLOOM's [terms of use](https://huggingface.co/bigscience/bloom).

## How does it work?

- Petals runs large language models like [BLOOM-176B](https://huggingface.co/bigscience/bloom) **collaboratively** — you load a small part of the model, then team up with people serving the other parts to run inference or fine-tuning.
- Inference runs at ≈ 1 sec per step (token) — 10x faster than possible with offloading, enough for chatbots and other interactive apps. Parallel inference reaches hundreds of tokens/sec.
- Beyond classic language model APIs — you can employ any fine-tuning and sampling methods by executing custom paths through the model or accessing its hidden states. You get the comforts of an API with the flexibility of PyTorch.

<p align="center">
    <img src="https://i.imgur.com/RTYF3yW.png" width="800">
</p>

<p align="center">
    📜 &nbsp;<b><a href="https://arxiv.org/pdf/2209.01188.pdf">Read paper</a></b>
</p>

## FAQ

1. **What's the motivation for people to host model layers in the public swarm?**

    People who run inference and fine-tuning themselves get a certain speedup if they host a part of the model locally. Some may be also motivated to "give back" to the community helping them to run the model (similarly to how [BitTorrent](https://en.wikipedia.org/wiki/BitTorrent) users help others by sharing data they have already downloaded).

    Since it may be not enough for everyone, we are also working on introducing explicit __incentives__ ("bloom points") for people donating their GPU time to the public swarm. Once this system is ready, people who earned these points will be able to spend them on inference/fine-tuning with higher priority or increased security guarantees, or (maybe) exchange them for other rewards.

2. **Why is the platform named "Petals"?**

    "Petals" is a metaphor for people serving different parts of the model. Together, they host the entire language model &mdash; [BLOOM](https://huggingface.co/bigscience/bloom).

    While our platform focuses on BLOOM now, we aim to support more [foundation models](https://arxiv.org/abs/2108.07258) in future.

## Installation

Here's how to install Petals with conda:

```bash
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U petals
```

This script uses Anaconda to install CUDA-enabled PyTorch.
If you don't have anaconda, you can get it from [here](https://www.anaconda.com/products/distribution).
If you don't want anaconda, you can install PyTorch [any other way](https://pytorch.org/get-started/locally/).
If you want to run models with 8-bit weights, please install **PyTorch with CUDA 11** or newer for compatility with [bitsandbytes](https://github.com/timDettmers/bitsandbytes).

__System requirements:__ Petals only supports Linux for now. If you don't have a Linux machine, consider running Petals in Docker (see our [image](https://hub.docker.com/r/learningathome/petals)) or, in case of Windows, in WSL2 ([read more](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)). CPU is enough to run a client, but you probably need a GPU to run a server efficiently.

## 🛠️ Development

Petals uses pytest with a few plugins. To install them, run:

```bash
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
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

```bash
export MODEL_NAME=bloom-testing/test-bloomd-560m-main REF_NAME=bigscience/bloom-560m
export INITIAL_PEERS=/ip4/127.0.0.1/tcp/31337/p2p/QmS9KwZptnVdB9FFV7uGgaTq4sEKBwcYeKZDfSpyKDUd1g
PYTHONPATH=. pytest tests --durations=0 --durations-min=1.0 -v
```

After you're done, you can terminate the servers and ensure that no zombie processes are left with `pkill -f petals.cli.run_server && pkill -f p2p`.

The automated tests use a more complex server configuration that can be found [here](https://github.com/bigscience-workshop/petals/blob/main/.github/workflows/run-tests.yaml).

### Code style

We use [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) and [isort](https://pycqa.github.io/isort/) for all pull requests.
Before committing your code, simply run `black . && isort .` and you will be fine.

## 📜 Citation

Alexander Borzunov, Dmitry Baranchuk, Tim Dettmers, Max Ryabinin, Younes Belkada, Artem Chumachenko, Pavel Samygin, and Colin Raffel.
[Petals: Collaborative Inference and Fine-tuning of Large Models.](https://arxiv.org/abs/2209.01188)
_arXiv preprint arXiv:2209.01188,_ 2022.

```bibtex
@article{borzunov2022petals,
  title = {Petals: Collaborative Inference and Fine-tuning of Large Models},
  author = {Borzunov, Alexander and Baranchuk, Dmitry and Dettmers, Tim and Ryabinin, Max and Belkada, Younes and Chumachenko, Artem and Samygin, Pavel and Raffel, Colin},
  journal = {arXiv preprint arXiv:2209.01188},
  year = {2022},
  url = {https://arxiv.org/abs/2209.01188}
}
```

--------------------------------------------------------------------------------

<p align="center">
    This project is a part of the <a href="https://bigscience.huggingface.co/">BigScience</a> research workshop.
</p>
<p align="center">
    <img src="https://petals.ml/bigscience.png" width="150">
</p>
