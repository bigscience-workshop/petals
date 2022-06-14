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
pip install bitsandbytes-cuda113==0.26.0
pip install https://github.com/learning-at-home/hivemind/archive/dac8940c324dd612d89c773b51a53e4a04c59064.zip
pip install https://github.com/huggingface/transformers/archive/224bde91caff4ccfd12277ab5e9bf97c61e22ee9.zip
```


# tests

```bash
# run one bloom block for a few steps
python -m cli.inference_one_block --config cli/config.json  # see other args

# minimalistic server
python -m cli.run_server --block_config bigscience/bloom-6b3 --num_blocks 2
```