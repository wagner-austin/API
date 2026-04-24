"""Tests for doc_extract_api.settings."""

from __future__ import annotations

import pytest
from platform_core.config import _test_hooks as platform_hooks
from platform_core.testing import make_fake_env

from doc_extract_api.settings import get_database_url, get_redis_url, get_tenant_email


class TestSettings:
    def test_get_redis_url(self) -> None:
        platform_hooks.get_env = make_fake_env({"REDIS_URL": "redis://localhost:6379/0"})
        assert get_redis_url() == "redis://localhost:6379/0"

    def test_get_redis_url_missing(self) -> None:
        platform_hooks.get_env = make_fake_env({})
        with pytest.raises(RuntimeError):
            get_redis_url()

    def test_get_database_url(self) -> None:
        platform_hooks.get_env = make_fake_env({"DATABASE_URL": "postgresql://localhost/db"})
        assert get_database_url() == "postgresql://localhost/db"

    def test_get_database_url_missing(self) -> None:
        platform_hooks.get_env = make_fake_env({})
        with pytest.raises(RuntimeError):
            get_database_url()

    def test_get_tenant_email(self) -> None:
        platform_hooks.get_env = make_fake_env({"DOC_EXTRACT_TENANT_EMAIL": "user@example.com"})
        assert get_tenant_email() == "user@example.com"

    def test_get_tenant_email_missing(self) -> None:
        platform_hooks.get_env = make_fake_env({})
        with pytest.raises(RuntimeError):
            get_tenant_email()
