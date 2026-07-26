"""Esports domain for match win prediction."""

from .domain import (
    ESPORTS_ALERT_THRESHOLD,
    ESPORTS_ALERT_TOPIC,
    ESPORTS_DOMAIN_NAME,
    ESPORTS_INPUT_TOPIC,
    ESPORTS_PREDICTION_TOPIC,
    EsportsDomain,
    make_esports_domain,
    make_esports_domain_config,
)
from .features import ESPORTS_FEATURE_NAMES, EsportsFeatureExtractor
from .schemas import (
    MatchEventType,
    MatchEventV1,
    decode_match_event,
    encode_match_event,
    make_match_event,
)

__all__ = [
    "ESPORTS_ALERT_THRESHOLD",
    "ESPORTS_ALERT_TOPIC",
    "ESPORTS_DOMAIN_NAME",
    "ESPORTS_FEATURE_NAMES",
    "ESPORTS_INPUT_TOPIC",
    "ESPORTS_PREDICTION_TOPIC",
    "EsportsDomain",
    "EsportsFeatureExtractor",
    "MatchEventType",
    "MatchEventV1",
    "decode_match_event",
    "encode_match_event",
    "make_esports_domain",
    "make_esports_domain_config",
    "make_match_event",
]
