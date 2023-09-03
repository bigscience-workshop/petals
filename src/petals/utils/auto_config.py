import os
from dataclasses import dataclass
from typing import Optional, Type, Union

from hivemind import get_logger
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from petals.utils.hf_auth import always_needs_auth

logger = get_logger(__name__)


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


class DefaultRevisionMixin:
    """
    Petals only supports Falcon loaded in the new in-library format (transformers.FalconModel).
    TII models were recently converted to this format but then reverted back due to compatibility issues.
    We chose to support only the new format since HF staff promised to eventually convert these models
    to the new format again, see https://huggingface.co/tiiuae/falcon-40b/discussions/90#64b4d23bf44fd957492f7602
    Until it happens, we override the default `main` revision for the TII repos with the commit
    pointing out to the model in the in-library format.
    """

    DEFAULT_REVISIONS = {
        "tiiuae/falcon-40b": "f1ba7d328c06aa6fbb4a8afd3c756f46d7e6b232",
        "tiiuae/falcon-40b-instruct": "7475ff8cfc36ed9a962b658ae3c33391566a85a5",
        "tiiuae/falcon-7b": "4e2d06f0a7c6370ebabbc30c6f59377ae8f73d76",
        "tiiuae/falcon-7b-instruct": "f8dac3fff96d5debd43edf56fb4e1abcfffbef28",
    }

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, revision: Optional[str] = None, **kwargs
    ):
        if revision is None and model_name_or_path in cls.DEFAULT_REVISIONS:
            revision = cls.DEFAULT_REVISIONS[model_name_or_path]
            logger.info(f"Loading {model_name_or_path}, revision {revision}")
        return super().from_pretrained(model_name_or_path, *args, revision=revision, **kwargs)


class AutoDistributedConfig(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "config"


class AutoDistributedModel(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model"


class AutoDistributedModelForCausalLM(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model_for_causal_lm"


class AutoDistributedModelForSequenceClassification(DefaultRevisionMixin, _AutoDistributedBase):
    _mapping_field = "model_for_sequence_classification"
