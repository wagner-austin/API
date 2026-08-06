"""Tests for the service's resolved-port and stop-file contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.service.constants import SERVICE_PORT, health_url, resolve_service_port
from tankpit_bot.service.service_main import resolve_service_stop_file
from tests.conftest import FakeEnv


def test_unset_env_resolves_the_default_port() -> None:
    """No override means the fiesta-proxied default port."""
    original_get_env = _test_hooks.get_env
    try:
        _test_hooks.get_env = FakeEnv({})
        assert resolve_service_port() == SERVICE_PORT
    finally:
        _test_hooks.get_env = original_get_env


def test_env_override_resolves_a_second_instance_port() -> None:
    """A second bot's service binds its own port."""
    original_get_env = _test_hooks.get_env
    try:
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_SERVICE_PORT": "27200"})
        assert resolve_service_port() == 27200
    finally:
        _test_hooks.get_env = original_get_env


def test_bad_ports_are_loud_errors() -> None:
    """Non-integer and out-of-range ports raise instead of binding garbage."""
    original_get_env = _test_hooks.get_env
    try:
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_SERVICE_PORT": "not-a-port"})
        with pytest.raises(ValueError):
            resolve_service_port()
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_SERVICE_PORT": "80"})
        with pytest.raises(ValueError, match="outside"):
            resolve_service_port()
    finally:
        _test_hooks.get_env = original_get_env


def test_health_url_carries_the_port() -> None:
    """The probe URL is loopback + the resolved port + /health."""
    assert health_url(27200) == "http://127.0.0.1:27200/health"


def test_stop_file_is_instance_scoped() -> None:
    """Two services must never share one stop sentinel."""
    original_get_env = _test_hooks.get_env
    try:
        _test_hooks.get_env = FakeEnv({})
        assert resolve_service_stop_file() == Path("runs/state/STOP")
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_INSTANCE": "alpha"})
        assert resolve_service_stop_file() == Path("runs/state/alpha/STOP")
    finally:
        _test_hooks.get_env = original_get_env
