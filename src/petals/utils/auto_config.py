from typing import Type

from transformers import AutoConfig, PretrainedConfig

CONFIG_MAPPING = {}  # Populated with AutoDistributedConfig.register()


class AutoDistributedConfig:
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> PretrainedConfig:
        config, kwargs = AutoConfig.from_pretrained(*args, **kwargs, return_unused_kwargs=True)
        if config.model_type not in CONFIG_MAPPING:
            raise ValueError(f"Petals does not support model type {config.model_type}")
        dist_config_class = CONFIG_MAPPING[config.model_type]
        return dist_config_class.from_dict(config.to_dict(), **kwargs)

    @staticmethod
    def register(config_class: Type[PretrainedConfig]) -> None:
        assert issubclass(config_class, PretrainedConfig)
        assert config_class.model_type not in CONFIG_MAPPING

        CONFIG_MAPPING[config_class.model_type] = config_class
