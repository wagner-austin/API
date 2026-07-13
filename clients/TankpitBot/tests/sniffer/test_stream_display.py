"""Tests for the streamed-display helpers in :mod:`tankpit_bot.sniffer.core`.

Covers ``_chrome_stream_display_args`` (env-driven Chromium argument
list), ``_chrome_stream_no_viewport`` (viewport-clamp gate) and
``_maximize_via_cdp`` (post-launch CDP maximise). These paths only
fire when Vibeshine's launcher sets the ``SUNSHINE_STREAM_DISPLAY_*``
env vars, so the tests drive them through the environment hook + a
fake CDP session.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot import _test_hooks
from tankpit_bot.sniffer.core import (
    _chrome_stream_display_args,
    _chrome_stream_no_viewport,
    _maximize_via_cdp,
)
from tests.conftest import FakeCDPSessionSimple, FakeEnv

_VALID_ENV: dict[str, str] = {
    "SUNSHINE_STREAM_DISPLAY_X": "100",
    "SUNSHINE_STREAM_DISPLAY_Y": "200",
    "SUNSHINE_STREAM_DISPLAY_W": "1920",
    "SUNSHINE_STREAM_DISPLAY_H": "1080",
}


def _install_env(env_vars: dict[str, str]) -> None:
    """Install a :class:`FakeEnv` for the streamed-display env vars."""
    _test_hooks.get_env = FakeEnv(dict(env_vars))


class TestChromeStreamDisplayArgs:
    """Cases for ``_chrome_stream_display_args``."""

    def test_returns_position_and_size_when_all_env_vars_set(self) -> None:
        """All four env vars produce the expected two-arg list."""
        _install_env(_VALID_ENV)
        assert _chrome_stream_display_args() == [
            "--window-position=100,200",
            "--window-size=1920,1080",
        ]

    def test_returns_empty_when_x_missing(self) -> None:
        """Missing X env var short-circuits to an empty list."""
        env = dict(_VALID_ENV)
        del env["SUNSHINE_STREAM_DISPLAY_X"]
        _install_env(env)
        assert _chrome_stream_display_args() == []

    def test_returns_empty_when_y_missing(self) -> None:
        """Missing Y env var short-circuits to an empty list."""
        env = dict(_VALID_ENV)
        del env["SUNSHINE_STREAM_DISPLAY_Y"]
        _install_env(env)
        assert _chrome_stream_display_args() == []

    def test_returns_empty_when_w_missing(self) -> None:
        """Missing W env var short-circuits to an empty list."""
        env = dict(_VALID_ENV)
        del env["SUNSHINE_STREAM_DISPLAY_W"]
        _install_env(env)
        assert _chrome_stream_display_args() == []

    def test_returns_empty_when_h_missing(self) -> None:
        """Missing H env var short-circuits to an empty list."""
        env = dict(_VALID_ENV)
        del env["SUNSHINE_STREAM_DISPLAY_H"]
        _install_env(env)
        assert _chrome_stream_display_args() == []

    def test_returns_empty_when_any_var_unparseable(self) -> None:
        """A non-integer env var short-circuits to an empty list."""
        env = dict(_VALID_ENV)
        env["SUNSHINE_STREAM_DISPLAY_W"] = "not_an_integer"
        _install_env(env)
        assert _chrome_stream_display_args() == []

    def test_returns_empty_when_w_is_zero(self) -> None:
        """Zero width is rejected — no positive rect to place the window."""
        env = dict(_VALID_ENV)
        env["SUNSHINE_STREAM_DISPLAY_W"] = "0"
        _install_env(env)
        assert _chrome_stream_display_args() == []

    def test_returns_empty_when_h_is_negative(self) -> None:
        """Negative height is rejected — no positive rect to place the window."""
        env = dict(_VALID_ENV)
        env["SUNSHINE_STREAM_DISPLAY_H"] = "-1"
        _install_env(env)
        assert _chrome_stream_display_args() == []


class TestChromeStreamNoViewport:
    """Cases for ``_chrome_stream_no_viewport``."""

    def test_true_when_display_configured(self) -> None:
        """A configured streamed display disables the default viewport clamp."""
        _install_env(_VALID_ENV)
        assert _chrome_stream_no_viewport() is True

    def test_false_when_display_not_configured(self) -> None:
        """No streamed display env keeps the default viewport clamp."""
        _install_env({})
        assert _chrome_stream_no_viewport() is False


class TestMaximizeViaCDP:
    """Cases for ``_maximize_via_cdp``."""

    def test_dispatches_both_setwindowbounds_commands(self) -> None:
        """A well-formed CDP surface receives both commands in order."""
        cdp = FakeCDPSessionSimple()
        # Only the first send() (Browser.getWindowForTarget) returns a
        # payload the function reads; the second send is fire-and-forget.
        cdp.add_response({"windowId": 42})
        cdp.add_response({})

        _maximize_via_cdp(cdp)

        calls = cdp.get_calls()
        assert calls[0] == ("Browser.getWindowForTarget", None)
        expected_params: JSONObject = {
            "windowId": 42,
            "bounds": {"windowState": "maximized"},
        }
        assert calls[1] == ("Browser.setWindowBounds", expected_params)

    def test_raises_when_windowid_missing_from_response(self) -> None:
        """A CDP surface that drops the ``windowId`` field surfaces loudly."""
        cdp = FakeCDPSessionSimple()
        cdp.add_response({})  # no windowId

        with pytest.raises(JSONTypeError):
            _maximize_via_cdp(cdp)
