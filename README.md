[
Based on assorted code by shuf(mryab@ younesbelkada@ borzunov@ timdettmers@ dbaranchuk@ greenfatguy@ artek0chumak@)
]


# Install
```bash

git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/CompVis/taming-transformers
pip install -e ./taming-transformers
pip install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops
mkdir -p models/ldm/cin256-v2/
wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt 

```


```python
hivemind-server --custom_module_path ./your_code_here.py --expert_cls ExampleModule --hidden_dim 512 --num_experts 1 \
    --expert_pattern "expert.0.[0:9999]" --identity server1.id
```