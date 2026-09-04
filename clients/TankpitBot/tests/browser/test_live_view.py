"""Tests for :mod:`tankpit_bot.browser.live_view`.

Covers the caster-expression builder (token substitution + loud
rejection of unusable cadences), the ensure/stop lifecycle against a
recording CDP fake (binding registered once per session, caster
re-evaluated every demanded tick), and the binding-event frame relay
with its loud drift rejections.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Generator

import pytest
from platform_core.json_utils import JSONObject, narrow_json_to_int, narrow_json_to_str

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import BrowserProtocol, PageProtocol
from tankpit_bot.browser.live_view import (
    BINDING_NAME,
    LiveViewService,
    build_caster_expression,
)
from tests.conftest import FakeEnv


@pytest.fixture(scope="module")
def headless_browser() -> Generator[BrowserProtocol, None, None]:
    """Launch one real headless Chromium shared by this module's caster tests.

    Module-scoped for the reason ``test_lifecycle.py`` gives for its own
    copy: ``launch()`` is the expensive call, and on hosts where a
    filesystem minifilter inspects browser teardown it has been measured
    at tens of seconds. ``--dist loadscope`` keeps a module on one xdist
    worker, so a module-scoped browser is never shared across processes.

    Duplicated rather than imported because a pytest fixture cannot
    travel by import without becoming an unused-name violation at every
    call site, and there is no ``tests/browser/conftest.py`` to hold it.

    Yields:
        A live headless Chromium browser.
    """
    factory = _test_hooks.sync_playwright
    if factory is None:
        factory = _test_hooks.get_sync_playwright()
    if factory is None:
        pytest.skip("Playwright not available")
    with factory() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


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


class TestCasterSuppressesUnchangedFrames:
    """The caster JS, executed in a real headless Chromium.

    Everything else in this file asserts substrings of the expression or
    drives the Python relay. Neither can see what the injected script
    DOES, and what it does is the whole point of this behaviour: the
    tankpit client paints on dirty flags, so at 12 Hz sampling roughly
    71 per cent of captures were byte-identical to the one before and
    the page shipped every one of them.
    """

    @staticmethod
    def _page(browser: BrowserProtocol) -> PageProtocol:
        """Open a page with one canvas and a call-counting binding stand-in.

        The real binding is installed by CDP as a function on ``window``;
        the caster only checks ``typeof window.<name> === "function"``,
        so a plain function of the same name exercises the identical
        path without a CDP round trip.

        Args:
            browser: The shared headless browser.

        Returns:
            The prepared page.
        """
        page = browser.new_context().new_page()
        page.goto(
            "data:text/html,<canvas id=c width=64 height=64></canvas>"
            "<script>"
            "window.__delivered=[];"
            f"window.{BINDING_NAME}=function(d){{window.__delivered.push(d);}};"
            "window.paint=function(v){"
            "const g=document.getElementById('c').getContext('2d');"
            "g.fillStyle='rgb('+v+',20,40)';g.fillRect(0,0,64,64);};"
            "window.paint(10);"
            "</script>"
        )
        return page

    def test_a_still_canvas_delivers_one_frame_not_a_stream_of_copies(
        self, headless_browser: BrowserProtocol
    ) -> None:
        """Nothing moving means nothing sent after the first frame."""
        page = self._page(headless_browser)
        page.evaluate(build_caster_expression(50.0, 0.8))
        page.wait_for_timeout(600)

        assert narrow_json_to_int(page.evaluate("window.__delivered.length")) == 1

    def test_a_repaint_delivers_exactly_one_more_frame(
        self, headless_browser: BrowserProtocol
    ) -> None:
        """A changed canvas resumes delivery, and then stops again.

        The second half is what distinguishes suppression from a broken
        caster: it must send on the change and then fall silent, rather
        than sending once and never again.
        """
        page = self._page(headless_browser)
        page.evaluate(build_caster_expression(50.0, 0.8))
        page.wait_for_timeout(400)
        page.evaluate("window.paint(200)")
        page.wait_for_timeout(600)

        assert narrow_json_to_int(page.evaluate("window.__delivered.length")) == 2
        first = narrow_json_to_str(page.evaluate("window.__delivered[0]"))
        second = narrow_json_to_str(page.evaluate("window.__delivered[1]"))
        assert first != second
        assert first.startswith("data:image/jpeg;base64,")
        assert second.startswith("data:image/jpeg;base64,")
