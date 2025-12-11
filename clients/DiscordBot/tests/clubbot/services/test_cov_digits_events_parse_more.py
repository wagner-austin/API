from __future__ import annotations

import logging

import pytest
from platform_core.digits_metrics_events import decode_digits_event
from platform_core.json_utils import JSONTypeError

logger = logging.getLogger(__name__)


def test_decode_non_dict_after_loads_raises() -> None:
    """Test that decode_digits_event raises JSONTypeError when JSON parses to non-dict."""
    # Pass JSON that parses to a list instead of a dict
    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        decode_digits_event("[1, 2, 3]")
