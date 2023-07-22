import os
import re
from dataclasses import dataclass
from typing import Optional, Type, Union

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from petals.utils.hf_auth import always_needs_auth


@dataclass
class _ModelClasses:
    config: Type[PretrainedConfig]
    model: Optional[Type[PreTrainedModel]] = None
    model_for_causal_lm: Optional[Type[PreTrainedModel]] = None
    model_for_sequence_classification: Optional[Type[PreTrainedModel]] = None


_CLASS_MAPPING = {}  # Populated by petals.models.* subpackages with register_model_classes()


def register_model_classes(*, config: Type[PretrainedConfig], **kwargs):
    assert issubclass(config, PretrainedConfig)
    assert config.model_type not in _CLASS_MAPPING, f"Model type {config.model_type} is already registered"

    _CLASS_MAPPING[config.model_type] = _ModelClasses(config=config, **kwargs)


class _AutoDistributedBase:
    _mapping_field = None  # Should be defined in child classes

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike, None], *args, **kwargs) -> PretrainedConfig:
        if (
            always_needs_auth(model_name_or_path)
            and kwargs.get("token") is None
            and kwargs.get("use_auth_token") is None
        ):
            kwargs["use_auth_token"] = True

        config = AutoConfig.from_pretrained(model_name_or_path, *args, **kwargs)
        if config.model_type not in _CLASS_MAPPING:
            raise ValueError(f"Petals does not support model type {config.model_type}")

        proper_cls = getattr(_CLASS_MAPPING[config.model_type], cls._mapping_field)
        if proper_cls is None:
            raise ValueError(f"Petals does not have {cls.__name__} for model type {config.model_type}")

        return proper_cls.from_pretrained(model_name_or_path, *args, **kwargs)


class AutoDistributedConfig(_AutoDistributedBase):
    _mapping_field = "config"


class AutoDistributedModel(_AutoDistributedBase):
    _mapping_field = "model"


class AutoDistributedModelForCausalLM(_AutoDistributedBase):
    _mapping_field = "model_for_causal_lm"


class AutoDistributedModelForSequenceClassification(_AutoDistributedBase):
    _mapping_field = "model_for_sequence_classification"
