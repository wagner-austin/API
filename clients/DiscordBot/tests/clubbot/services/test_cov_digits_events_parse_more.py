from __future__ import annotations

import logging

import pytest
from platform_core.digits_metrics_events import decode_digits_event
from platform_core.json_utils import JSONTypeError, JSONValue

logger = logging.getLogger(__name__)


def _fake_load_json_str_returns_list(s: str) -> JSONValue:
    """Fake load_json_str that returns a list instead of dict."""
    return [1, 2, 3]


def test_decode_non_dict_after_loads_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that decode_digits_event raises JSONTypeError when JSON parses to non-dict."""
    import platform_core.digits_metrics_events as mod

    # Monkeypatch load_json_str to return list -> should raise JSONTypeError
    monkeypatch.setattr(mod, "load_json_str", _fake_load_json_str_returns_list)
    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        decode_digits_event("{}")
