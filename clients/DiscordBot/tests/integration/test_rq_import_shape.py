from __future__ import annotations

import logging


def test_rq_exposes_retry_top_level() -> None:
    # Enforce the project's chosen RQ import shape
    from rq import Retry

    assert callable(Retry)


logger = logging.getLogger(__name__)
