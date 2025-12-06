from __future__ import annotations

import logging

import pytest
from platform_core.digits_metrics_events import try_decode_digits_event
from platform_core.json_utils import JSONValue

logger = logging.getLogger(__name__)


def _fake_load_json_str_returns_list(s: str) -> JSONValue:
    """Fake load_json_str that returns a list instead of dict."""
    return [1, 2, 3]


def test_try_decode_non_dict_after_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that try_decode_digits_event returns None when JSON parses to non-dict."""
    import platform_core.digits_metrics_events as mod

    # Brace check passes but loader returns list -> should yield None
    monkeypatch.setattr(mod, "load_json_str", _fake_load_json_str_returns_list)
    assert try_decode_digits_event("{}") is None
