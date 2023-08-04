import peft
import pytest
import torch
import transformers
from hivemind import get_logger
from transformers.generation import BeamSearchScorer
from transformers.models.bloom import BloomForCausalLM

from petals import DistributedBloomForCausalLM
from test_utils import *

logger = get_logger(__name__)


@pytest.mark.forked
@pytest.mark.parametrize("use_peft", (True, False) if ADAPTER_NAME else (False,))
@pytest.mark.parametrize("pass_empty_tensors", (True, False))
def test_full_model_exact_match(use_peft: bool, pass_empty_tensors: bool, atol_forward=1e-3, atol_inference=1e-3):
    tokenizer = transformers.BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME,
        initial_peers=INITIAL_PEERS,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        active_adapter=ADAPTER_NAME if use_peft else None,
    )
    config = model.config
    assert isinstance(model, DistributedBloomForCausalLM)
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
                recurrent_outputs.append(sess.step(torch.empty(1, 0, config.hidden_size)))

            for t in range(embs.shape[1]):
                if t == 4:
                    recurrent_outputs.append(sess.step(embs[:, 4:9, :]))
                elif 4 < t < 9:
                    continue
                else:
                    recurrent_outputs.append(sess.step(embs[:, t : t + 1, :]))

                if t == 2 and pass_empty_tensors:
                    recurrent_outputs.append(sess.step(torch.empty(1, 0, config.hidden_size)))
                    recurrent_outputs.append(sess.step(torch.empty(1, 0, config.hidden_size)))

        recurrent_outputs = torch.cat(recurrent_outputs, dim=1)
        recurrent_outputs = model.transformer.ln_f(recurrent_outputs)
        recurrent_outputs = model.lm_head(recurrent_outputs)
        assert torch.allclose(recurrent_outputs, parallel_outputs, rtol=0, atol=atol_inference)
        logger.info("Inference is consistent with forward")

        del model, embs, recurrent_outputs

        if REF_NAME:
            ref_model = transformers.BloomForCausalLM.from_pretrained(
                REF_NAME, low_cpu_mem_usage=True, torch_dtype=torch.float32
            )
            if use_peft:
                ref_model = peft.PeftModel.from_pretrained(ref_model, ADAPTER_NAME)
                ref_model.train(False)
            if config.vocab_size < ref_model.config.vocab_size:
                ref_model.resize_token_embeddings(config.vocab_size)
                logger.warning(f"Resized the reference model embeddings, new total = {ref_model.config.vocab_size}")

            dummy_mask = torch.ones_like(test_inputs, dtype=torch.bool)
            # note: this creates a dummy mask to make the test compatible with older transformer versions
            # prior to https://github.com/huggingface/transformers/pull/17837
            ref_outputs = ref_model.forward(test_inputs, attention_mask=dummy_mask).logits.float()
            assert torch.allclose(ref_outputs, parallel_outputs, rtol=0, atol=atol_forward)
            logger.warning(f"Distributed forward is consistent with {type(ref_model)}.forward")
            del ref_model, ref_outputs, dummy_mask
        else:
            logger.warning("Did not test exact match with local model: REF_NAME environment variable is not set")
            assert False


@pytest.mark.forked
def test_greedy_generation(max_new_tokens=4):
    tokenizer = transformers.BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME, initial_peers=INITIAL_PEERS, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    inputs = tokenizer("A cat sat on a mat", return_tensors="pt")["input_ids"]
    remote_outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
    )
    hf_outputs = BloomForCausalLM.greedy_search(model, input_ids=inputs, max_length=inputs.size(1) + max_new_tokens)
    assert torch.allclose(remote_outputs, hf_outputs), "Greedy search results are not identical to HF"

    inputs_batch = tokenizer(["A cat sat on a mat", "A dog sat on a mat"], return_tensors="pt", padding=True)[
        "input_ids"
    ]
    remote_outputs_batch = model.generate(
        inputs_batch,
        max_new_tokens=max_new_tokens,
    )
    hf_outputs_batch = BloomForCausalLM.greedy_search(
        model, input_ids=inputs_batch, max_length=inputs_batch.size(1) + max_new_tokens
    )
    assert torch.allclose(
        remote_outputs_batch, hf_outputs_batch
    ), "Greedy search results are not identical to HF in multibatch mode"


@pytest.mark.forked
@pytest.mark.parametrize("sampling_options", [dict(), dict(temperature=100.0), dict(top_k=5), dict(top_p=0.9)])
@pytest.mark.skip("Sampling is currently not consistent with outputs from Transformers")
def test_sampling(sampling_options, max_new_tokens=4):
    torch.manual_seed(0)
    tokenizer = transformers.BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME, initial_peers=INITIAL_PEERS, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    logits_warper = BloomForCausalLM._get_logits_warper(model, num_beams=1, **sampling_options)
    inputs = tokenizer("A cat sat on a mat", return_tensors="pt")["input_ids"]
    with torch.random.fork_rng():
        remote_outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            **sampling_options,
        )
    with torch.random.fork_rng():
        hf_outputs = BloomForCausalLM.sample(
            model, input_ids=inputs, max_length=inputs.size(1) + max_new_tokens, logits_warper=logits_warper
        )
    assert torch.allclose(remote_outputs, hf_outputs), "Sampling results are not identical to HF"

    inputs_batch = tokenizer(["A cat sat on a mat", "A dog sat on a mat"], return_tensors="pt", padding=True)[
        "input_ids"
    ]
    with torch.random.fork_rng():
        remote_outputs_batch = model.generate(
            inputs_batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            **sampling_options,
        )
    with torch.random.fork_rng():
        hf_outputs_batch = BloomForCausalLM.sample(
            model,
            input_ids=inputs_batch,
            max_length=inputs_batch.size(1) + max_new_tokens,
            logits_warper=logits_warper,
        )
    assert torch.allclose(
        remote_outputs_batch, hf_outputs_batch
    ), "Sampling results are not identical to HF in multibatch mode"


@pytest.mark.forked
def test_beam_search_generation(max_new_tokens=4, num_beams=2):
    tokenizer = transformers.BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME, initial_peers=INITIAL_PEERS, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    text = "A cat sat on a mat"
    inputs = tokenizer(text, return_tensors="pt")["input_ids"]
    remote_outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    beam_scorer = BeamSearchScorer(
        batch_size=inputs.size(0),
        num_beams=num_beams,
        device=inputs.device,
        length_penalty=0,
        do_early_stopping=False,
    )
    hf_inputs = tokenizer([text] * 2, return_tensors="pt")["input_ids"]
    hf_outputs = BloomForCausalLM.beam_search(
        model, input_ids=hf_inputs, max_length=inputs.size(1) + max_new_tokens, beam_scorer=beam_scorer
    )
    assert torch.allclose(remote_outputs, hf_outputs), "Beam search results are not identical to HF"
