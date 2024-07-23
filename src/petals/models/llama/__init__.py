from petals.models.llama.block import WrappedLlamaBlock
from petals.models.llama.config import DistributedLlamaConfig
from petals.models.llama.model import (
    DistributedLlamaForCausalLM,
    DistributedLlamaForSpeculativeGeneration,
    DistributedLlamaForSequenceClassification,
    DistributedLlamaModel,
    DistributedLlamaForSpeculativeGeneration,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedLlamaConfig,
    model=DistributedLlamaModel,
    model_for_causal_lm=DistributedLlamaForCausalLM,
    model_for_speculative=DistributedLlamaForSpeculativeGeneration,
    model_for_sequence_classification=DistributedLlamaForSequenceClassification,
)
