from petals.models.bloom.block import WrappedBloomBlock
from petals.models.bloom.config import DistributedBloomConfig
from petals.models.bloom.model import (
    DistributedBloomForCausalLM,
    DistributedBloomForSequenceClassification,
    DistributedBloomModel,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedBloomConfig,
    model=DistributedBloomModel,
    model_for_causal_lm=DistributedBloomForCausalLM,
    model_for_sequence_classification=DistributedBloomForSequenceClassification,
)
