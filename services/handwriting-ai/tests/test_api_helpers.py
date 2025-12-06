from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode

from handwriting_ai.api.routes.read import _raise_if_too_large
from handwriting_ai.config import (
    AppConfig,
    DigitsConfig,
    SecurityConfig,
    Settings,
    limits_from_settings,
)


def test_raise_if_too_large_raises_with_oversize() -> None:
    app_cfg: AppConfig = {}
    dig_cfg: DigitsConfig = {"max_image_mb": 0}
    sec_cfg: SecurityConfig = {"api_key": ""}
    s: Settings = {"app": app_cfg, "digits": dig_cfg, "security": sec_cfg}
    limits = limits_from_settings(s)
    with pytest.raises(AppError) as ei:
        _raise_if_too_large(b"x" * 1024, limits)
    e = ei.value
    assert e.code is ErrorCode.PAYLOAD_TOO_LARGE and e.http_status == 413


def test_raise_if_too_large_ok_within_limit() -> None:
    app_cfg: AppConfig = {}
    dig_cfg: DigitsConfig = {"max_image_mb": 1}
    sec_cfg: SecurityConfig = {"api_key": ""}
    s: Settings = {"app": app_cfg, "digits": dig_cfg, "security": sec_cfg}
    limits = limits_from_settings(s)
    # 1 byte is well below 1MB
    _raise_if_too_large(b"x", limits)
