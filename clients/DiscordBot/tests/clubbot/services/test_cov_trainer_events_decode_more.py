from __future__ import annotations

import logging

from platform_core.json_utils import JSONValue
from platform_discord.trainer.handler import decode_trainer_event

logger = logging.getLogger(__name__)


def _fake_load_json_str_returns_list(s: str) -> JSONValue:
    """Fake load_json_str that returns a list instead of dict."""
    return [1, 2, 3]


def test_trainer_handler_decode_invalid_json_returns_none() -> None:
    """Test that decode_trainer_event returns None for invalid JSON."""
    assert decode_trainer_event("not valid json") is None
    assert decode_trainer_event("[]") is None
    assert decode_trainer_event("{}") is None
