import peft
import pytest
import torch
import transformers
from hivemind import get_logger

from petals import AutoDistributedModelForCausalLM
from test_utils import *

logger = get_logger(__name__)


@pytest.fixture
def tokenizer():
    # We set use_fast=False since LlamaTokenizerFast is slow on load
    return transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


@pytest.fixture(scope="module", params=[None, ADAPTER_NAME] if ADAPTER_NAME else [None])
def models(request):
    active_adapter = request.param

    model = AutoDistributedModelForCausalLM.from_pretrained(
        MODEL_NAME, initial_peers=INITIAL_PEERS, torch_dtype=torch.float32, active_adapter=active_adapter
    )

    ref_model = transformers.AutoModelForCausalLM.from_pretrained(
        REF_NAME, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    if active_adapter is not None:
        ref_model = peft.PeftModel.from_pretrained(active_adapter, ADAPTER_NAME)
        ref_model.train(False)

    return model, ref_model


@pytest.mark.parametrize("pass_empty_tensors", (True, False))
def test_full_model_exact_match(tokenizer, models, pass_empty_tensors, atol_forward=1e-3, atol_inference=1e-3):
    model, ref_model = models
    assert len(model.transformer.h) == model.config.num_hidden_layers

    test_inputs = tokenizer("A quick brown fox was minding its own buisness", return_tensors="pt")["input_ids"]

    with torch.inference_mode():
        parallel_outputs = model.forward(test_inputs).logits
        assert torch.all(torch.isfinite(parallel_outputs))
        logger.info("Forward outputs are finite")

        embs = model.transformer.word_embeddings(test_inputs)
        embs = model.transformer.word_embeddings_layernorm(embs)
        recurrent_outputs = []
        with model.transformer.h.inference_session(max_length=embs.shape[1]) as sess:
            if pass_empty_tensors:
                recurrent_outputs.append(sess.step(torch.empty(1, 0, model.config.hidden_size)))

            for t in range(embs.shape[1]):
                if t == 4:
                    recurrent_outputs.append(sess.step(embs[:, 4:9, :]))
                elif 4 < t < 9:
                    continue
                else:
                    recurrent_outputs.append(sess.step(embs[:, t : t + 1, :]))

                if t == 2 and pass_empty_tensors:
                    recurrent_outputs.append(sess.step(torch.empty(1, 0, model.config.hidden_size)))
                    recurrent_outputs.append(sess.step(torch.empty(1, 0, model.config.hidden_size)))

        recurrent_outputs = torch.cat(recurrent_outputs, dim=1)
        recurrent_outputs = model.transformer.ln_f(recurrent_outputs)
        recurrent_outputs = model.lm_head(recurrent_outputs)
        assert torch.allclose(
            recurrent_outputs, parallel_outputs, rtol=0, atol=atol_inference
        ), "Inference differs from forward pass"

        ref_outputs = ref_model.forward(test_inputs).logits.float()
        assert torch.allclose(
            ref_outputs, parallel_outputs, rtol=0, atol=atol_forward
        ), "Outputs are not identical to HF"


def test_greedy_generation(tokenizer, models, max_new_tokens=4):
    model, ref_model = models

    inputs_single = tokenizer("A cat sat on a mat", return_tensors="pt")["input_ids"]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs_batch = tokenizer(["A cat sat on a mat", "A dog sat on a mat"], return_tensors="pt", padding=True)[
        "input_ids"
    ]

    for inputs in [inputs_single, inputs_batch]:
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
        ref_outputs = ref_model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
        assert torch.allclose(outputs, ref_outputs), f"Greedy generation is not identical to HF with {inputs.shape=}"


@pytest.mark.parametrize(
    "sampling_options",
    [
        dict(do_sample=True),
        dict(do_sample=True, temperature=100.0),
        dict(do_sample=True, top_k=5),
        dict(do_sample=True, top_p=0.9),
    ],
)
def test_sampling(tokenizer, models, sampling_options, max_new_tokens=4):
    model, ref_model = models
    torch.manual_seed(0)

    inputs_single = tokenizer("A cat sat on a mat", return_tensors="pt")["input_ids"]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs_batch = tokenizer(["A cat sat on a mat", "A dog sat on a mat"], return_tensors="pt", padding=True)[
        "input_ids"
    ]

    for inputs in [inputs_single, inputs_batch]:
        with torch.random.fork_rng([model.device]):
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
        with torch.random.fork_rng([ref_model.device]):
            ref_outputs = ref_model.generate(inputs, max_new_tokens=max_new_tokens)
        assert torch.allclose(
            outputs, ref_outputs
        ), f"Sampling is not identical to HF with {inputs.shape=}, {sampling_options=}"


def test_beam_search_generation(tokenizer, models, max_new_tokens=4, num_beams=6):
    model, ref_model = models

    inputs = tokenizer("A cat sat on a mat", return_tensors="pt")["input_ids"]

    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
    ref_outputs = ref_model.generate(inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
    assert torch.allclose(outputs, ref_outputs), "Beam search results are not identical to HF"
