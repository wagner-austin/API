"""Credentials come from the environment, and their absence is two failures;
the endpoint comes from the stack's declaration unless overridden.

Two codes rather than one because the operator fixes them in different
places, and a single CONFIG_ERROR would send half the readers to the wrong
one.
"""

from __future__ import annotations

import pytest
from platform_core.error_codes_tooling import BoardWatchErrorCode, StackEndpointErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str
from platform_core.mcp_testing import DECLARED_TASKBOARD_URL, stack_endpoints_text
from platform_core.stack_endpoints import STACK_ENDPOINTS_PATH

from board_watch import _test_hooks
from board_watch.config import (
    API_KEY_VARIABLE,
    TENANT_ID_VARIABLE,
    URL_VARIABLE,
    load_credentials,
)
from tests.conftest import FakeEnv, FakeFiles, set_environment


def _without_override() -> tuple[FakeEnv, FakeFiles]:
    """Bind the environment with no url override and a filesystem holding
    only the stack's endpoint declaration.

    Returns:
        The bound environment and filesystem.
    """
    environment = set_environment()
    del environment.values[URL_VARIABLE]
    files = FakeFiles({STACK_ENDPOINTS_PATH: stack_endpoints_text()})
    _test_hooks.read_text = files.read_text
    return environment, files


def test_reads_both_secrets_and_defaults_the_url_to_the_declared_taskboard() -> None:
    """The endpoint defaults to where the MCPs stack declares the taskboard,
    not to a loopback port the hub no longer forwards (MCPs c6fc4882); the
    secrets deliberately have no default."""
    _without_override()
    credentials = load_credentials()
    assert credentials["api_key"] == "test-key"
    assert credentials["tenant_id"] == "2e137b5f-0000-4000-8000-000000000000"
    assert credentials["url"] == DECLARED_TASKBOARD_URL


def test_a_declaration_without_a_taskboard_refuses_rather_than_guessing() -> None:
    _, files = _without_override()
    files.contents[STACK_ENDPOINTS_PATH] = dump_json_str({"services": {}})
    with pytest.raises(AppError) as raised:
        load_credentials()
    assert raised.value.code is StackEndpointErrorCode.UNDECLARED


def test_an_explicit_url_overrides_the_default_without_reading_the_declaration() -> None:
    """A non-default deployment is configured, not code-changed."""
    environment = set_environment()
    environment.values[URL_VARIABLE] = "http://127.0.0.1:9999/mcp"
    files = FakeFiles()
    _test_hooks.read_text = files.read_text
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


def test_an_empty_url_is_the_unset_case_and_reads_the_declaration() -> None:
    """An exported-but-blank override is the unset case, not a blank endpoint."""
    environment, _ = _without_override()
    environment.values[URL_VARIABLE] = ""
    assert load_credentials()["url"] == DECLARED_TASKBOARD_URL


__all__ = [
    "test_a_declaration_without_a_taskboard_refuses_rather_than_guessing",
    "test_a_missing_api_key_raises_its_own_code",
    "test_a_missing_tenant_raises_its_own_code",
    "test_an_empty_url_is_the_unset_case_and_reads_the_declaration",
    "test_an_explicit_url_overrides_the_default_without_reading_the_declaration",
    "test_reads_both_secrets_and_defaults_the_url_to_the_declared_taskboard",
]
