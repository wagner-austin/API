from __future__ import annotations

import logging

import pytest
from platform_core.json_utils import JSONTypeError

logger = logging.getLogger(__name__)


def test_digits_decode_non_dict_raises() -> None:
    from platform_core.digits_metrics_events import decode_digits_event

    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        decode_digits_event("[]")
