"""Credentials come from the environment, and their absence is two failures.

Two codes rather than one because the operator fixes them in different
places, and a single CONFIG_ERROR would send half the readers to the wrong
one.
"""

from __future__ import annotations

import pytest
from platform_core.error_codes import BoardWatchErrorCode
from platform_core.errors import AppError

from board_watch import _test_hooks
from board_watch.config import (
    API_KEY_VARIABLE,
    DEFAULT_URL,
    TENANT_ID_VARIABLE,
    URL_VARIABLE,
    load_credentials,
)
from tests.conftest import FakeEnv, set_environment


def test_reads_both_secrets_and_defaults_the_url() -> None:
    """The endpoint has a default; the secrets deliberately do not."""
    set_environment()
    credentials = load_credentials()
    assert credentials["api_key"] == "test-key"
    assert credentials["tenant_id"] == "2e137b5f-0000-4000-8000-000000000000"
    assert credentials["url"] == DEFAULT_URL


def test_an_explicit_url_overrides_the_default() -> None:
    """A non-default deployment is configured, not code-changed."""
    environment = set_environment()
    environment.values[URL_VARIABLE] = "http://127.0.0.1:9999/mcp"
    assert load_credentials()["url"] == "http://127.0.0.1:9999/mcp"


@pytest.mark.parametrize("empty", ["", None])
def test_a_missing_api_key_raises_its_own_code(empty: str | None) -> None:
    """Unset and set-to-empty are the same failure and must not differ."""
    values = {TENANT_ID_VARIABLE: "tenant"}
    if empty is not None:
        values[API_KEY_VARIABLE] = empty
    _test_hooks.env = FakeEnv(values)
    with pytest.raises(AppError) as raised:
        load_credentials()
    assert raised.value.code is BoardWatchErrorCode.API_KEY_MISSING
    assert API_KEY_VARIABLE in raised.value.message


@pytest.mark.parametrize("empty", ["", None])
def test_a_missing_tenant_raises_its_own_code(empty: str | None) -> None:
    """The board has no default tenant, so this cannot be defaulted either."""
    values = {API_KEY_VARIABLE: "key"}
    if empty is not None:
        values[TENANT_ID_VARIABLE] = empty
    _test_hooks.env = FakeEnv(values)
    with pytest.raises(AppError) as raised:
        load_credentials()
    assert raised.value.code is BoardWatchErrorCode.TENANT_ID_MISSING
    assert TENANT_ID_VARIABLE in raised.value.message


def test_an_empty_url_falls_back_to_the_default() -> None:
    """An exported-but-blank override is the unset case, not a blank endpoint."""
    environment = set_environment()
    environment.values[URL_VARIABLE] = ""
    assert load_credentials()["url"] == DEFAULT_URL


__all__ = [
    "test_a_missing_api_key_raises_its_own_code",
    "test_a_missing_tenant_raises_its_own_code",
    "test_an_empty_url_falls_back_to_the_default",
    "test_an_explicit_url_overrides_the_default",
    "test_reads_both_secrets_and_defaults_the_url",
]
