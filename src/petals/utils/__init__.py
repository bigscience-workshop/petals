from petals.utils.auto_config import (
    AutoDistributedConfig,
    AutoDistributedModel,
    AutoDistributedModelForCausalLM,
    AutoDistributedSpeculativeModel,
    AutoDistributedModelForSequenceClassification,
)
from petals.utils.dht import declare_active_modules, get_remote_module_infos
