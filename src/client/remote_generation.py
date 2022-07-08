import torch

from typing import List, Optional

from src.utils.generation_algorithms import DecodingAlgorithm, GreedyAlgorithm, TopKAlgorithm, NucleusAlgorithm
from src.utils.generation_constraints import ABConstraint, MaxNewTokensConstraint

from transformers.modeling_utils import PreTrainedModel


class RemoteGenerationMixin(PreTrainedModel):
    def generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        do_sample: Optional[bool] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        decoding_algorithm: Optional[DecodingAlgorithm] = None,
        provided_constraints: List[ABConstraint] = [],
        **model_kwargs,
    ) -> torch.Tensor:
        if decoding_algorithm is None:
            if do_sample:
                if (top_k is None) == (top_p is None):
                    raise ValueError("You have to provide only top_k or top_p for sampling")
                if top_k:
                    decoding_algorithm = TopKAlgorithm(top_k, temperature)
                elif top_p:
                    decoding_algorithm = NucleusAlgorithm(top_p, temperature)
            else:
                decoding_algorithm = GreedyAlgorithm()
        
        constraints = []
        constraints.extend(provided_constraints)
                
        if max_new_tokens and eos_token_id:
            constraints.append(MaxNewTokensConstraint(max_new_tokens, eos_token_id))
            
        for constraint in constraints:
            constraint.consume_prefix(inputs)
       
        word_embeddings = self.transformer.word_embeddings.weight.t()

        with self.transformer.h.inference_session() as sess:
            last_token_id = inputs[:, -1]
            outputs = []
            while torch.any(last_token_id != eos_token_id):
                embs = self.transformer.word_embeddings(inputs)
                embs = self.transformer.word_embeddings_layernorm(embs)
                for emb_ids in range(embs.size(1)):
                    recurrent_output = sess.step(embs[:, emb_ids:emb_ids+1])
                recurrent_output = self.transformer.ln_f(recurrent_output)
                lm_logits = (recurrent_output @ word_embeddings).float()
                for constraint in constraints:
                    lm_logits = constraint.calculate_transation(lm_logits)
                last_token_id, _ = decoding_algorithm(lm_logits)
                for constraint in constraints:
                    constraint.update(last_token_id, torch.ones_like(last_token_id))
                outputs.append(last_token_id)
                inputs = last_token_id
            
        return torch.cat(outputs, dim=-1)

