"""Domain implementations for multi-domain streaming platform.

Provides base event schemas, protocol definitions, and registry for
pluggable domain implementations. Each domain (covenant, weather, esports,
etc.) implements DomainProtocol and registers with DomainRegistry.
"""

from .base_schemas import (
    BaseAlertEventV1,
    BaseAlertSeverity,
    BaseInputEventV1,
    BasePredictionEventV1,
    decode_base_alert_event,
    decode_base_input_event,
    decode_base_prediction_event,
    encode_base_alert_event,
    encode_base_input_event,
    encode_base_prediction_event,
    make_base_alert_event,
    make_base_input_event,
    make_base_prediction_event,
)
from .protocols import (
    DomainConfig,
    DomainProtocol,
    FeatureExtractorProtocol,
    ModelProtocol,
    make_domain_config,
)
from .registry import DomainRegistry

__all__ = [
    "BaseAlertEventV1",
    "BaseAlertSeverity",
    "BaseInputEventV1",
    "BasePredictionEventV1",
    "DomainConfig",
    "DomainProtocol",
    "DomainRegistry",
    "FeatureExtractorProtocol",
    "ModelProtocol",
    "decode_base_alert_event",
    "decode_base_input_event",
    "decode_base_prediction_event",
    "encode_base_alert_event",
    "encode_base_input_event",
    "encode_base_prediction_event",
    "make_base_alert_event",
    "make_base_input_event",
    "make_base_prediction_event",
    "make_domain_config",
]
