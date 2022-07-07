

# Install (core only)
```bash

conda create -y --name demo-for-laion python=3.8.12 pip
conda activate demo-for-laion
conda install -y -c conda-forge cudatoolkit-dev==11.3.1 cudatoolkit==11.3.1 cudnn==8.2.1.32
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install https://github.com/learning-at-home/hivemind/archive/refs/heads/master.zip
```


### Run server
```python
python -m run_server --custom_module_path ./your_code_here.py --expert_cls ExampleModule --hidden_dim 512 \
   --dht_prefix "enter_name_here" --identity server1.id  --host_maddrs "/ip4/0.0.0.0/tcp/31337"
# connect extra servers via --initial_peers ADDRESS_PRINTED_BY_ONE_OR_MORE_EXISTNG_PEERS # e.g. /ip4/123.123.123.123/rcp/31337
```

### Call remote inference

```python
import torch
import hivemind
from client import BalancedRemoteExpert
dht = hivemind.DHT(
    initial_peers=['TODO_COPY_ADDRESS_FROM_ONE_OR_MODE_SERVERS'], start=True, client_mode=True
)

self = BalancedRemoteExpert(dht=dht, uid_prefix="enter_name_here.")

self(torch.randn(1, 512))
```


[
Based on assorted code by shuf(mryab@ younesbelkada@ borzunov@ timdettmers@ dbaranchuk@ greenfatguy@ artek0chumak@ and hivemind contributors)
]
