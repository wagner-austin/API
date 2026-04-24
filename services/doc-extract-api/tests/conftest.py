"""Pytest configuration and fixtures for doc-extract-api tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.config import _test_hooks as platform_hooks
from platform_core.testing import make_fake_env
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from doc_extract_api import _test_hooks


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore all hooks after each test."""
    original_platform_get_env = platform_hooks.get_env
    original_connect_db = _test_hooks.connect_db
    original_pdfplumber_open = _test_hooks.pdfplumber_open
    original_redis_factory = _test_hooks.redis_factory
    original_read_file = _test_hooks.read_file
    original_ocr_pdf = _test_hooks.ocr_pdf
    original_test_runner = _test_hooks.test_runner

    yield

    platform_hooks.get_env = original_platform_get_env
    _test_hooks.connect_db = original_connect_db
    _test_hooks.pdfplumber_open = original_pdfplumber_open
    _test_hooks.redis_factory = original_redis_factory
    _test_hooks.read_file = original_read_file
    _test_hooks.ocr_pdf = original_ocr_pdf
    _test_hooks.test_runner = original_test_runner


def _load_mcps_env() -> dict[str, str]:
    """Load env vars from the MCPs repo .env file.

    The MCPs repo is a sibling of the api repo under ~/PROJECTS/.

    Returns:
        Dict of env vars from the MCPs .env, empty if not found.
    """
    from pathlib import Path

    # Walk up from tests/ to find the api repo root (has libs/ dir)
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "libs").is_dir():
            # Found api repo root, MCPs is a sibling
            mcps_env = current.parent / "MCPs" / ".env"
            if mcps_env.exists():
                result: dict[str, str] = {}
                for line in mcps_env.read_text(encoding="utf-8").splitlines():
                    if "=" in line and not line.startswith("#"):
                        key, val = line.split("=", 1)
                        result[key.strip()] = val.strip()
                return result
            break
        current = current.parent
    return {}


@pytest.fixture(autouse=True)
def _default_test_env() -> None:
    """Provide default test environment."""
    mcps_env = _load_mcps_env()
    env_vars: dict[str, str] = {
        "REDIS_URL": "redis://test-redis",
        "DATABASE_URL": "postgresql://fake/db",
        "DOC_EXTRACT_TENANT_EMAIL": "test@example.com",
    }
    # Add DATABASE_TEST_URL from MCPs .env if available
    test_db_url = mcps_env.get("DATABASE_TEST_URL", "")
    if len(test_db_url) > 0:
        env_vars["DATABASE_TEST_URL"] = test_db_url

    env = make_fake_env(env_vars)
    platform_hooks.get_env = env

    def _fake_redis(url: str) -> RedisStrProto:
        return FakeRedis()

    _test_hooks.redis_factory = _fake_redis
