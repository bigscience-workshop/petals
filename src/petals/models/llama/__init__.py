from petals.models.llama.block import WrappedLlamaBlock
from petals.models.llama.config import DistributedLlamaConfig
from petals.models.llama.model import (
    DistributedLlamaForCausalLM,
    DistributedLlamaForSequenceClassification,
    DistributedLlamaModel,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedLlamaConfig,
    model=DistributedLlamaModel,
    model_for_causal_lm=DistributedLlamaForCausalLM,
    model_for_sequence_classification=DistributedLlamaForSequenceClassification,
)
