import pytest
import torch
import transformers
from hivemind import get_logger, use_hivemind_log_handler
from test_utils import *

from src.client.remote_model import DistributedBloomForCausalLM

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


@pytest.mark.forked
def test_full_model_exact_match(atol_forward=1e-3, atol_inference=1e-3):
    tokenizer = transformers.BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    assert isinstance(model, DistributedBloomForCausalLM)
    assert len(model.transformer.h) == model.config.n_layer

    test_inputs = tokenizer("A cat sat on a mat", return_tensors="pt")["input_ids"]

    with torch.no_grad():
        parallel_outputs = model.forward(test_inputs).logits
        assert torch.all(torch.isfinite(parallel_outputs))
        logger.info("Forward outputs are finite")

        embs = model.transformer.word_embeddings(test_inputs)
        embs = model.transformer.word_embeddings_layernorm(embs)
        recurrent_outputs = []
        with model.transformer.h.inference_session() as sess:
            for t in range(embs.shape[1]):
                recurrent_outputs.append(sess.step(embs[:, t : t + 1, :]))
        recurrent_outputs = torch.cat(recurrent_outputs, dim=1)
        recurrent_outputs = model.transformer.ln_f(recurrent_outputs)
        recurrent_outputs = model.lm_head(recurrent_outputs)
        assert torch.allclose(recurrent_outputs, parallel_outputs, rtol=0, atol=atol_inference)
        logger.info("Inference is consistent with forward")

        del model, recurrent_outputs

        if REF_NAME:
            ref_model = transformers.AutoModelForCausalLM.from_pretrained(REF_NAME)
            dummy_mask = torch.ones_like(test_inputs, dtype=torch.bool)
            # note: this creates a dummy mask to make the test compatible with older transformer versions
            # prior to https://github.com/huggingface/transformers/pull/17837
            ref_outputs = ref_model.forward(test_inputs, attention_mask=dummy_mask).logits
            assert torch.allclose(ref_outputs, parallel_outputs, rtol=0, atol=atol_forward)
            logger.warning(f"Distributed forward is consistent with {type(ref_model)}.forward")
            del ref_model, ref_outputs, dummy_mask
        else:
            logger.warning("Did not test exact match with local model: REF_NAME environment variable is not set")
            assert False
