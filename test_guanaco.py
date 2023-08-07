import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from petals import DistributedLlamaForCausalLM, DistributedBloomForCausalLM

# MODEL_NAME = "bigscience/bloom-560m"
# ADAPTER_NAME = "artek0chumak/bloom-560m-safe-peft"
MODEL_NAME = "Enoch/llama-65b-hf"
ADAPTER_NAME = "artek0chumak/guanaco-65b"
MODEL_MAX_LENGTH = 256
DEVICE = "cuda:0"
# INITIAL_PEERS = ["/ip4/127.0.0.1/tcp/31337/p2p/QmS9KwZptnVdB9FFV7uGgaTq4sEKBwcYeKZDfSpyKDUd1g"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.padding_side = "right"
tokenizer.model_max_length = MODEL_MAX_LENGTH
distr_model = DistributedLlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, active_adapter=ADAPTER_NAME).to(DEVICE)
# distr_model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS, torch_dtype=torch.bfloat16, active_adapter=ADAPTER_NAME).to(DEVICE)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)
model.eval()

peft_model = peft.PeftModel.from_pretrained(model, ADAPTER_NAME)
peft_model.eval()

text = "Hello, who are you?"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(DEVICE)

# print("With adapters, petals")
# with torch.inference_mode():
#     parallel_outputs = distr_model.forward(input_ids).logits
#     assert torch.all(torch.isfinite(parallel_outputs))
#     print("Forward outputs are finite")

#     embs = distr_model.transformer.word_embeddings(input_ids)
#     embs = distr_model.transformer.word_embeddings_layernorm(embs)
#     recurrent_outputs = []
#     with distr_model.transformer.h.inference_session(max_length=embs.shape[1]) as sess:
#         for t in range(embs.shape[1]):
#             recurrent_outputs.append(sess.step(embs[:, t : t + 1, :]))

#     recurrent_outputs = torch.cat(recurrent_outputs, dim=1)
#     recurrent_outputs = distr_model.transformer.ln_f(recurrent_outputs)
#     recurrent_outputs = distr_model.lm_head(recurrent_outputs)
    
#     print(f"Parallel, petals: {parallel_outputs}")
#     print(f"Recurrent, petals: {recurrent_outputs}")
with torch.no_grad():
    distr_outputs = distr_model(input_ids)
print(distr_outputs.logits)

with torch.inference_mode():
    outputs = model(input_ids)
print("With adapters, local")
print(outputs.logits)
