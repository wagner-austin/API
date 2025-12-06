from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def test_digits_try_decode_non_dict() -> None:
    from platform_core.digits_metrics_events import try_decode_digits_event

    assert try_decode_digits_event("[]") is None
