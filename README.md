<p align="center">
    <img src="https://i.imgur.com/7eR7Pan.png" width="400"><br>
    Run large language models at home, BitTorrent-style.<br>
    Fine-tuning and inference <a href="https://github.com/bigscience-workshop/petals#benchmarks">up to 10x faster</a> than offloading
    <br><br>
    <a href="https://pypi.org/project/petals/"><img src="https://img.shields.io/pypi/v/petals.svg?color=green"></a>
    <a href="https://discord.gg/tfHfe8B34k"><img src="https://img.shields.io/discord/865254854262652969?label=discord&logo=discord&logoColor=white"></a>
    <br>
</p>

Generate text with distributed **Llama 2** (70B), **Falcon** (40B+), **BLOOM** (176B) (or their derivatives), and fine‑tune them for your own tasks &mdash; right from your desktop computer or Google Colab:

```python
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# Choose any model available at https://health.petals.dev
model_name = "petals-team/StableBeluga2"  # This one is fine-tuned Llama 2 (70B)

# Connect to a distributed network hosting model layers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)

# Run the model as if it were on your computer
inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))  # A cat sat on a mat...
```

<p align="center">
    🚀 &nbsp;<b><a href="https://colab.research.google.com/drive/1uCphNY7gfAUkdDrTx21dZZwCOUDCMPw8?usp=sharing">Try now in Colab</a></b>
</p>

🔏 **Privacy.** Your data will be processed with the help of other people in the public swarm. Learn more about privacy [here](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety). For sensitive data, you can set up a [private swarm](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm) among people you trust.

🦙 **Want to run Llama 2?** Request access to its weights at the ♾️ [Meta AI website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and 🤗 [Model Hub](https://huggingface.co/meta-llama/Llama-2-70b-hf), then run `huggingface-cli login` in the terminal before loading the model. Or just try it in our [chatbot app](https://chat.petals.dev).

💬 **Any questions?** Ping us in [our Discord](https://discord.gg/KdThf2bWVU)!

## Connect your GPU and increase Petals capacity

Petals is a community-run system &mdash; we rely on people sharing their GPUs. You can check out [available models](https://health.petals.dev) and help serving one of them! As an example, here is how to host a part of [Stable Beluga 2](https://huggingface.co/stabilityai/StableBeluga2) on your GPU:

🐧 **Linux + Anaconda.** Run these commands for NVIDIA GPUs (or follow [this](https://github.com/bigscience-workshop/petals/wiki/Running-on-AMD-GPU) for AMD):

```bash
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install git+https://github.com/bigscience-workshop/petals
python -m petals.cli.run_server petals-team/StableBeluga2
```

🪟 **Windows + WSL.** Follow [this guide](https://github.com/bigscience-workshop/petals/wiki/Run-Petals-server-on-Windows) on our Wiki.

🐋 **Docker.** Run our [Docker](https://www.docker.com) image for NVIDIA GPUs (or follow [this](https://github.com/bigscience-workshop/petals/wiki/Running-on-AMD-GPU) for AMD):

```bash
sudo docker run -p 31330:31330 --ipc host --gpus all --volume petals-cache:/cache --rm \
    learningathome/petals:main \
    python -m petals.cli.run_server --port 31330 petals-team/StableBeluga2
```

🍏 **macOS + Apple M1/M2 GPU.** Install [Homebrew](https://brew.sh/), then run these commands:

```bash
brew install python
python3 -m pip install git+https://github.com/bigscience-workshop/petals
python3 -m petals.cli.run_server petals-team/StableBeluga2
```

<p align="center">
    📚 &nbsp;<b><a href="https://github.com/bigscience-workshop/petals/wiki/FAQ:-Frequently-asked-questions#running-a-server">Learn more</a></b> (how to use multiple GPUs, start the server on boot, etc.)
</p>

💬 **Any questions?** Ping us in [our Discord](https://discord.gg/X7DgtxgMhc)!

🦙 **Want to host Llama 2?** Request access to its weights at the ♾️ [Meta AI website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and 🤗 [Model Hub](https://huggingface.co/meta-llama/Llama-2-70b-hf), generate an 🔑 [access token](https://huggingface.co/settings/tokens), then add `--token YOUR_TOKEN_HERE` to the `python -m petals.cli.run_server` command.

🔒 **Security.** Hosting a server does not allow others to run custom code on your computer. Learn more [here](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety).

🏆 **Thank you!** Once you load and host 10+ blocks, we can show your name or link on the [swarm monitor](https://health.petals.dev) as a way to say thanks. You can specify them with `--public_name YOUR_NAME`.

## How does it work?

- You load a small part of the model, then join a [network](https://health.petals.dev) of people serving the other parts. Single‑batch inference runs at up to **6 tokens/sec** for **Llama 2** (70B) and up to **4 tokens/sec** for **Falcon** (180B) — enough for [chatbots](https://chat.petals.dev) and interactive apps.
- You can employ any fine-tuning and sampling methods, execute custom paths through the model, or see its hidden states. You get the comforts of an API with the flexibility of **PyTorch** and **🤗 Transformers**.

<p align="center">
    <img src="https://i.imgur.com/RTYF3yW.png" width="800">
</p>

<p align="center">
    📜 &nbsp;<b><a href="https://arxiv.org/pdf/2209.01188.pdf">Read paper</a></b>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    📚 &nbsp;<b><a href="https://github.com/bigscience-workshop/petals/wiki/FAQ:-Frequently-asked-questions">See FAQ</a></b>
</p>

## 📚 Tutorials, examples, and more

Basic tutorials:

- Getting started: [tutorial](https://colab.research.google.com/drive/1uCphNY7gfAUkdDrTx21dZZwCOUDCMPw8?usp=sharing)
- Prompt-tune Llama-65B for text semantic classification: [tutorial](https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb)
- Prompt-tune BLOOM to create a personified chatbot: [tutorial](https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-personachat.ipynb)

Useful tools:

- [Chatbot web app](https://chat.petals.dev) (connects to Petals via an HTTP/WebSocket endpoint): [source code](https://github.com/petals-infra/chat.petals.dev)
- [Monitor](https://health.petals.dev) for the public swarm: [source code](https://github.com/petals-infra/health.petals.dev)

Advanced guides:

- Launch a private swarm: [guide](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm)
- Run a custom model: [guide](https://github.com/bigscience-workshop/petals/wiki/Run-a-custom-model-with-Petals)

### Benchmarks

Please see **Section 3.3** of our [paper](https://arxiv.org/pdf/2209.01188.pdf).

### 🛠️ Contributing

Please see our [FAQ](https://github.com/bigscience-workshop/petals/wiki/FAQ:-Frequently-asked-questions#contributing) on contributing.

### 📜 Citation

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
    <img src="https://petals.dev/bigscience.png" width="150">
</p>
