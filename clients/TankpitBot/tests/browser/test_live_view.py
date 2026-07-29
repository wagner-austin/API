"""Tests for :mod:`tankpit_bot.browser.live_view`.

Covers the caster-expression builder (token substitution + loud
rejection of unusable cadences), the ensure/stop lifecycle against a
recording CDP fake (binding registered once per session, caster
re-evaluated every demanded tick), and the binding-event frame relay
with its loud drift rejections.
"""

from __future__ import annotations

import base64
from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot import _test_hooks
from tankpit_bot.browser.live_view import (
    BINDING_NAME,
    LiveViewService,
    build_caster_expression,
)
from tests.conftest import FakeEnv


class _RecordingCDP:
    """CDP-session fake that records sends and handler registrations."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, JSONObject | None]] = []
        self.registrations: list[str] = []
        self.handlers: dict[str, Callable[[JSONObject], None]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        self.sent.append((method, params))
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        self.registrations.append(event)
        self.handlers[event] = handler

    def detach(self) -> None:
        raise AssertionError("the live view never detaches the session")


class _FrameSink:
    """Records every frame the service publishes."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def __call__(self, frame: bytes) -> None:
        self.frames.append(frame)


def _make_service(sink: _FrameSink) -> LiveViewService:
    """Build a service with deterministic env-backed cadence."""
    _test_hooks.get_env = FakeEnv(
        {
            "TANKPIT_BOT_VIDEO_FPS": "10",
            "TANKPIT_BOT_VIDEO_QUALITY": "0.7",
        }
    )
    return LiveViewService(publish=sink)


class TestBuildCasterExpression:
    """Token substitution + validation contract."""

    def test_substitutes_cadence_binding_and_quality(self) -> None:
        """The rendered snippet carries the interval, quality, and binding."""
        expression = build_caster_expression(12.0, 0.8)
        assert "}, 83);" in expression  # round(1000 / 12)
        assert 'toDataURL("image/jpeg", 0.8)' in expression
        assert f"window.{BINDING_NAME}(data)" in expression
        assert "__QUALITY__" not in expression
        assert "__INTERVAL_MS__" not in expression
        assert "__BINDING__" not in expression

    def test_rejects_non_positive_fps(self) -> None:
        """A zero fps would divide by zero — refused loudly."""
        with pytest.raises(ValueError, match="fps must be positive"):
            build_caster_expression(0.0, 0.8)

    def test_rejects_out_of_range_quality(self) -> None:
        """Quality outside (0, 1] is a config error, not a clamp."""
        with pytest.raises(ValueError, match="quality must be in"):
            build_caster_expression(12.0, 1.5)


class TestLiveViewLifecycle:
    """ensure/stop against the recording CDP fake."""

    def test_first_ensure_registers_binding_then_evaluates(self) -> None:
        """The first demanded tick wires the binding + installs the caster."""
        service = _make_service(_FrameSink())
        cdp = _RecordingCDP()

        service.ensure(cdp)

        assert service.active is True
        assert cdp.registrations == ["Runtime.bindingCalled"]
        methods = [method for method, _ in cdp.sent]
        assert methods == ["Runtime.addBinding", "Runtime.evaluate"]
        binding_params = cdp.sent[0][1]
        assert binding_params == {"name": BINDING_NAME}

    def test_ensure_reevaluates_every_call_for_navigation_selfheal(self) -> None:
        """Each demanded tick re-runs the idempotent snippet.

        Page navigations (quit-to-lobby, re-login) wipe injected JS;
        the per-tick re-evaluation reinstalls the caster. The binding
        registration itself survives navigations, so it is wired
        exactly once per CDP session.
        """
        service = _make_service(_FrameSink())
        cdp = _RecordingCDP()

        service.ensure(cdp)
        service.ensure(cdp)
        service.ensure(cdp)

        assert cdp.registrations == ["Runtime.bindingCalled"]
        methods = [method for method, _ in cdp.sent]
        assert methods == [
            "Runtime.addBinding",
            "Runtime.evaluate",
            "Runtime.evaluate",
            "Runtime.evaluate",
        ]

    def test_fresh_session_gets_a_fresh_binding(self) -> None:
        """A NEW CDP session (next game session) is wired independently."""
        service = _make_service(_FrameSink())
        first = _RecordingCDP()
        service.ensure(first)
        service.stop(first)

        second = _RecordingCDP()
        service.ensure(second)

        assert second.registrations == ["Runtime.bindingCalled"]
        assert [method for method, _ in second.sent] == [
            "Runtime.addBinding",
            "Runtime.evaluate",
        ]

    def test_stop_evaluates_the_stop_snippet_and_marks_inactive(self) -> None:
        """The last viewer leaving stops the in-page interval."""
        service = _make_service(_FrameSink())
        cdp = _RecordingCDP()
        service.ensure(cdp)

        service.stop(cdp)

        assert service.active is False
        last_params = cdp.sent[-1][1]
        if last_params is None:
            raise AssertionError("Runtime.evaluate always carries params")
        expression = last_params["expression"]
        if not isinstance(expression, str):
            raise AssertionError("expression must be a string")
        assert "window.__botCast.stop()" in expression

    def test_stop_while_inactive_is_a_noop(self) -> None:
        """A stop with no active caster sends nothing."""
        service = _make_service(_FrameSink())
        cdp = _RecordingCDP()

        service.stop(cdp)

        assert cdp.sent == []


class TestBindingFrameRelay:
    """The bindingCalled → publish contract with loud drift rejection."""

    def _relay(self, sink: _FrameSink) -> Callable[[JSONObject], None]:
        """Wire a service to a CDP fake and return the binding handler."""
        service = _make_service(sink)
        cdp = _RecordingCDP()
        service.ensure(cdp)
        return cdp.handlers["Runtime.bindingCalled"]

    def test_frame_payload_is_decoded_and_published(self) -> None:
        """A JPEG data-URL payload lands on the sink as raw bytes."""
        sink = _FrameSink()
        handler = self._relay(sink)
        payload = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8frame").decode()

        handler({"name": BINDING_NAME, "payload": payload})

        assert sink.frames == [b"\xff\xd8frame"]

    def test_foreign_binding_names_are_ignored(self) -> None:
        """Events for other bindings pass through without publishing."""
        sink = _FrameSink()
        handler = self._relay(sink)

        handler({"name": "someOtherBinding", "payload": "irrelevant"})

        assert sink.frames == []

    def test_non_jpeg_payload_fails_loudly(self) -> None:
        """A payload without the JPEG data-URL prefix is caster drift."""
        sink = _FrameSink()
        handler = self._relay(sink)

        with pytest.raises(ValueError, match="not a JPEG data URL"):
            handler({"name": BINDING_NAME, "payload": "data:image/png;base64,QUJD"})
        assert sink.frames == []

    def test_corrupt_base64_fails_loudly(self) -> None:
        """Invalid base64 raises instead of publishing garbage."""
        sink = _FrameSink()
        handler = self._relay(sink)

        with pytest.raises(ValueError, match="invalid base64"):
            handler({"name": BINDING_NAME, "payload": "data:image/jpeg;base64,@@nope@@"})
        assert sink.frames == []
