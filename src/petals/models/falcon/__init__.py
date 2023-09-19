from petals.models.falcon.block import WrappedFalconBlock
from petals.models.falcon.config import DistributedFalconConfig
from petals.models.falcon.model import (
    DistributedFalconForCausalLM,
    DistributedFalconForSequenceClassification,
    DistributedFalconModel,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedFalconConfig,
    model=DistributedFalconModel,
    model_for_causal_lm=DistributedFalconForCausalLM,
    model_for_sequence_classification=DistributedFalconForSequenceClassification,
)
