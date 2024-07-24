import random

import pytest
import torch
import transformers

from petals import (
    AutoDistributedConfig,
    AutoDistributedSpeculativeModel,
    DistributedLlamaForSpeculativeGeneration,
    RemoteSequential,
)
from petals.server.block_functions import MAX_SHORT_INFERENCE_TOKENS
from petals.server.from_pretrained import load_pretrained_block
from test_utils import *


@pytest.mark.forked
def test_remote_block_with_cache_invalidation_exact_match(atol_forward=1e-4, atol_inference=1e-3):
    config = AutoDistributedConfig.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    remote_sequential = RemoteSequential(config)

    block_index = random.randint(0, config.num_hidden_layers - 1)
    remote_block = remote_sequential[block_index]

    inputs = torch.randn(1, MAX_SHORT_INFERENCE_TOKENS - 50, config.hidden_size)
    short_inputs = torch.randn(1, MAX_SHORT_INFERENCE_TOKENS - 50, config.hidden_size)
    short_inputs[:, :2, :] = inputs[:, :2, :]

    initial_outputs_inference = None
    secondary_outputs_inference = None
    with torch.inference_mode():
        with remote_block.inference_session(max_length=inputs.shape[1]) as sess:
            initial_outputs_inference = sess.step(inputs)
            sess.position = 2
            secondary_outputs_inference = sess.step(short_inputs[:, 2:, :])
            result = torch.cat([initial_outputs_inference[:, :2, :], secondary_outputs_inference], dim=1)

    ref_block = load_pretrained_block(MODEL_NAME, block_index, torch_dtype=torch.float32)
    (outputs_local,) = ref_block(short_inputs)

    assert torch.allclose(outputs_local, result, rtol=0, atol=atol_inference)


@pytest.fixture
def noisy_model():
    noisy_model = transformers.AutoModelForCausalLM.from_pretrained(
        REF_NAME, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    lm_head = noisy_model.get_output_embeddings()
    assert isinstance(lm_head, torch.nn.Linear)
    with torch.no_grad():
        lm_head.weight += torch.randn_like(lm_head.weight) * 0.02
    return noisy_model


@pytest.fixture
def model():
    return transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )


@pytest.fixture
def tokenizer():
    # We set use_fast=False since LlamaTokenizerFast is slow on load
    return transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


@pytest.mark.forked
@pytest.mark.skipif(
    "llama" not in MODEL_NAME.lower(),
    reason="Speculative generation now works only for llama models",
)
def test_remote_speculative_generation(tokenizer, model, noisy_model, atol_inference=1e-3):
    speculated_distributed_model = AutoDistributedSpeculativeModel.from_pretrained(
        MODEL_NAME, initial_peers=INITIAL_PEERS, torch_dtype=torch.float32, small_model=noisy_model
    )

    inputs_single = tokenizer("A cat sat on a mat", return_tensors="pt")["input_ids"]

    generated_spec = speculated_distributed_model.generate(inputs_single, max_new_tokens=100, do_sample=False)
    generated_local = model.generate(inputs_single, max_new_tokens=100, do_sample=False)

    assert torch.allclose(generated_spec, generated_local, rtol=0, atol=atol_inference)
