# [under construction] Running 🌸BLOOM with PETALS

This tutorial will walk you through the steps of setting up your private swarm to inference and fine-tune BLOOM.


### Spin up a server

Before you can use a model, you (or someone) needs to host its transformer blocks. In PETALS, this is done by running a
server: each server hosts one or multiple transformer blocks and connect to each other to form the full model. 

__Run the first server:__ every swarm begins with one server. You can start a basic server with this script:

```bash

export CUDA_VISIBLE_DEVICES=  # choose a GPU index (e.g. "0") or leave blank to run on CPU 
export IPV4=$(dig -4 TXT +short o-o.myaddr.l.google.com @ns1.google.com |  tr -d '"')
echo "My IP:[ " $IPV4 " ] - must be non-empty"
# if IP is empty, you can set it manually. To test PETALS on your local machine, export IPV4=127.0.0.1

export PORT=12345 # pick a free and open port; if you're not sure what it means, please see the "Details" section below

python -m cli.run_server \
 --identity_path ./serverA.id  --host_maddrs /ip4/$IPV4/tcp/$PORT /ip4/$IPV4/udp/6789/$PORT \
 --converted_model_name_or_path bigscience/test-bloomd-6b3 `# model name on huggingface hub; must be converted first` \
 --num_blocks 8 `# serve this many transformer layers; layer indices are determined automatically` \
 --throughput 1 `# server's performance, used for load-balancing; leave blank to auto-detect with speedtest`
```

* __TODO__ example outputs as in hivemind/moe
* __TODO__ describe outputs and explain to --initial_peers!

This initial server has 8 out of 30 total blocks. To run the full model, we will need to add more servers.

__Additional servers__ can join the swarm using the ```--initial_peers``` option. The new server runs 
You can open a new console and run a similar console script, but with two changes:

1. ```--initial_peers /ip4/.../tcp/...``` - copy the address string from the running server.
  If there are multiple active servers, you can use any one or both of them: the swarm is fully decentralzied.
2. Replace ```--identity_path``` and ```--host_maddrs``` with a unique name and address for each server. When testing 
  on a local machine, you can also remove these options altogether.

For example, this is how your second server could look like:

```bash
python -m cli.run_server --converted_model_name_or_path bigscience/test-bloomd-6b3 \
  --num_blocks 8 --initial_peers /ip4/127.0.0.1/tcp/12345/p2p/QmcTODOReplaceThisWithTheActualAddressOfAnotherServer
```

__TODO capture outputs__

Note that the second server chose a different subset of layers (8-15), since layers (0-7) have already been served.
To cover the entire 30-layer model, __please run 2 servers like this__, or run one server with `--num_blocks 14`. 

Running the full model requires 12-24GB of RAM between the servers, depending on your numeric precision.
For large models, PETALS can reduce memory usage with 8-bit quantization - see "Running at scale" section on how to use that.


__Details:__
* TODO about ports and how to open them
* __The host_maddrs__ line contains the so-called multiaddresses: learn more about them in [this guide](https://docs.libp2p.io/concepts/addressing/).
* TODO about identity


### Use the model

* TODO disclaimer - 6b3 is for testing and is not very efficient; petals is optimized for 100B+
 
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


```python
# Initialize distributed BLOOM and connect to the swarm
model = DistributedBloomForCausalLM.from_pretrained(
    "bigscience/distributed-bloom", tuning_mode="ptune", initial_peers=SEE_BELOW
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

__TODO link to Artek's training notebook__

### Running at scale

This section contains a step-by-step guide to spin up the large bloom (176B parameters) on servers

- TODO about hivemind-dht as persistent initial peers  
- TODO about cheap servers (e.g. hetzner)
```
 --torch_dtype bfloat16  # we recomment model's original dtype for GPU and float32 for CPU
 --load_in_8bit   # requires Turing or newer gpu, see LLM.8bit paper https://arxiv.org/abs/2208.07339
 # ^-- remove load_in_8bit when running on CPU or an older GPU (e.g. 1080Ti or V100)
 
```
__TODO activation quantization__

__TODO blocks per GPU memory__



### Deploy your own model with PETALS 

To run PETALS servers with your own model, you need to convert the model weights into a PETALS-compatible format.
This conversion splits each individual block into a separate branch. This allows each peer to download only the
layers they need, instead of the entire 350GB model.

For BLOOM models, you can convert them using the following script:
```bash

# convert model from HF hub to a distributed format (can take hours depending on your connection!)
MY_WRITE_TOKEN=TODO_WRITE_TOKEN_FROM_https://huggingface.co/settings/token
python -m cli.convert_model --model bigscience/bloom-6b3  \
  --output_path ./converted_model --output_repo bigscience/test-bloomd-6b3 \
  --use_auth_token $MY_WRITE_TOKEN  # ^-- todo replace output repo with something you have access to
```

If you want to run a non-BLOOM model (e.g. [OPT](https://arxiv.org/abs/2205.01068) or [YALM](https://github.com/yandex/YaLM-100B)),
you will need to edit the code a bit.
Currently, PETALS uses a vanilla implementation of BLOOM in `src/bloom`, so it is possible to replace it with other models from Hugging Face transformers. 

Assuming your model is already is compatible with Hugging Face, you will need 3 extra steps:

1. Edit `cli/convert_model.py` to partition your model checkpoint into individual blocks and non-transformer layers.
   Once you are done, run this script to convert your model and upload it to Hugging Face. If your model is private,
   you can use your internal storage instead (see next step).
2. In `src/bloom/from_pretrained.py`, edit `load_pretrained_block` to load a single block of your custom model.
  Your block should be able to run `.forward(hidden_states=..., use_cache=true_or_false, layer_past=optional_tensors)`.
  After this step, you should be able to launch a server with the new model name.
3. Open `src/client/remote_model.py` and change `DistributedBloomModel` to load the model of your choice.
  Create non-transformer layers (e.g. embeddings and logits) as usual. Instead of loading transformer blocks,
  create a RemoteSequential instance. 

Once you are done, run `tests/test_full_model.py` to verify that your conversion went correctly.
In future, we hope to streamline this process, making it possible to serve any language model available on Hugging Face.
If you with this future to come sooner and willing to work on a pull-request, please contact us.



<p align="center">
    This project is a part of the <a href="https://bigscience.huggingface.co/">BigScience</a> research workshop.
</p>
<p align="center">
    <img src="https://petals.ml/bigscience.png" width="150">
</p>
