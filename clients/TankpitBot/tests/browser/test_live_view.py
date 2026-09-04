"""Tests for :mod:`tankpit_bot.browser.live_view`.

Two layers. The expression builder and the ensure/stop lifecycle run
against a recording CDP fake. The caster ITSELF runs in a real headless
Chromium against a real loopback listener, because the thing that
matters about it -- that it posts frames off the Playwright thread, and
posts only when the picture changed -- is invisible to any assertion on
the generated string.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, narrow_json_to_int

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import BrowserProtocol, PageProtocol
from tankpit_bot.browser.live_view import LiveViewService, build_caster_expression
from tankpit_bot.sniffer.chrome_launch import LOOPBACK_POST_ARGS
from tests.conftest import FakeEnv

CAST_URL = "http://127.0.0.1:27100/cast"


class _RecordingCDP:
    """CDP-session fake that records what was sent to it."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, JSONObject | None]] = []
        self.registrations: list[str] = []

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Record one send.

        Args:
            method: CDP method name.
            params: Method parameters.

        Returns:
            An empty result.
        """
        self.sent.append((method, params))
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Record one event registration.

        Args:
            event: CDP event name.
            handler: Callback that would receive it.
        """
        _ = handler
        self.registrations.append(event)

    def detach(self) -> None:
        """Ignore detach."""


class TestBuildCasterExpression:
    def test_the_cast_url_is_embedded_as_json(self) -> None:
        """The URL is quoted by json.dumps, not string-concatenated.

        A hand-quoted URL breaks on the first character that needs
        escaping and produces a caster that fails to parse, which shows
        up as no video rather than as an error.
        """
        expression = build_caster_expression(30.0, 0.8, CAST_URL)

        assert dump_json_str(CAST_URL) in expression
        assert "fetch(" in expression

    def test_the_interval_is_derived_from_fps(self) -> None:
        """60 fps is a 17 ms interval; 12 is 83."""
        assert "}, 17);" in build_caster_expression(60.0, 0.8, CAST_URL)
        assert "}, 83);" in build_caster_expression(12.0, 0.8, CAST_URL)

    def test_the_quality_reaches_todataurl(self) -> None:
        """The encoder argument is the caller's, not a default."""
        assert 'toDataURL("image/jpeg", 0.55)' in build_caster_expression(30.0, 0.55, CAST_URL)

    def test_a_non_positive_fps_is_refused(self) -> None:
        """The interval math would divide by zero."""
        with pytest.raises(ValueError, match="fps must be positive"):
            build_caster_expression(0.0, 0.8, CAST_URL)

    def test_a_quality_outside_the_unit_range_is_refused(self) -> None:
        """toDataURL silently clamps; refusing says what was meant."""
        with pytest.raises(ValueError, match=r"quality must be in \(0, 1\]"):
            build_caster_expression(30.0, 1.5, CAST_URL)

    def test_an_empty_cast_url_is_refused(self) -> None:
        """A caster with nowhere to post is a timer burning encodes.

        Bot passes "" for sessions with no service (``make run``,
        replay, scenarios) and builds no caster at all in that case, so
        reaching here with an empty URL is a wiring bug.
        """
        with pytest.raises(ValueError, match="cast URL must not be empty"):
            build_caster_expression(30.0, 0.8, "")


class TestLiveViewLifecycle:
    def test_ensure_evaluates_the_caster_and_registers_nothing(self) -> None:
        """No CDP binding any more: installing is one evaluate.

        The binding is what coupled frame delivery to the Playwright
        thread. Its absence is the fix, so its absence is asserted.
        """
        service = LiveViewService(CAST_URL)
        cdp = _RecordingCDP()

        service.ensure(cdp)

        assert [method for method, _ in cdp.sent] == ["Runtime.evaluate"]
        assert cdp.registrations == []
        assert service.active is True

    def test_ensure_is_idempotent_across_ticks(self) -> None:
        """Re-evaluating every tick is the self-heal for navigations."""
        service = LiveViewService(CAST_URL)
        cdp = _RecordingCDP()

        service.ensure(cdp)
        service.ensure(cdp)

        assert [method for method, _ in cdp.sent] == ["Runtime.evaluate"] * 2
        assert service.active is True

    def test_stop_turns_the_caster_off_once(self) -> None:
        """The stop expression goes out, and only while active."""
        service = LiveViewService(CAST_URL)
        cdp = _RecordingCDP()
        service.ensure(cdp)

        service.stop(cdp)
        service.stop(cdp)

        assert service.active is False
        assert [method for method, _ in cdp.sent] == ["Runtime.evaluate"] * 2

    def test_stop_before_start_sends_nothing(self) -> None:
        """A session nobody watched never installed anything to remove."""
        service = LiveViewService(CAST_URL)
        cdp = _RecordingCDP()

        service.stop(cdp)

        assert cdp.sent == []

    def test_the_configured_rate_comes_from_the_environment(self) -> None:
        """``TANKPIT_BOT_VIDEO_FPS`` reaches the built expression."""
        original = _test_hooks.get_env
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_VIDEO_FPS": "10"})
        try:
            service = LiveViewService(CAST_URL)
            cdp = _RecordingCDP()
            service.ensure(cdp)
        finally:
            _test_hooks.get_env = original

        _, params = cdp.sent[0]
        if params is None:
            raise AssertionError("Runtime.evaluate carries parameters")
        assert "}, 100);" in str(params["expression"])


@pytest.fixture(scope="module")
def headless_browser() -> Generator[BrowserProtocol, None, None]:
    """Launch one real headless Chromium shared by this module.

    Module-scoped for the reason ``test_lifecycle.py`` gives for its own
    copy: ``launch()`` is the expensive call. Duplicated rather than
    imported because a pytest fixture cannot travel by import without
    becoming an unused-name violation at every call site.

    Yields:
        A live headless Chromium browser.
    """
    factory = _test_hooks.sync_playwright
    if factory is None:
        factory = _test_hooks.get_sync_playwright()
    if factory is None:
        pytest.skip("Playwright not available")
    with factory() as playwright:
        # The SAME args production launches with. Without them Chrome's
        # Local Network Access gate silently swallows every POST, and the
        # suite would be testing a browser this project never runs.
        browser = playwright.chromium.launch(headless=True, args=LOOPBACK_POST_ARGS)
        try:
            yield browser
        finally:
            browser.close()


class _Intake:
    """A loopback listener standing in for the service's ``/cast`` route."""

    def __init__(self) -> None:
        """Start on an ephemeral port."""
        self.bodies: list[bytes] = []
        intake = self

        class _Handler(BaseHTTPRequestHandler):
            # A handler thread must never be able to block forever.
            # The stdlib default is None -- no socket timeout at all --
            # so a connection that is opened and never written to parks
            # a thread in ``rfile.readline()` for the life of the
            # process. Chrome opens exactly such sockets: it keeps
            # spare connections to an origin it is posting to at
            # 50 fps, and they sit silent.
            timeout = 5.0

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                intake.bodies.append(self.rfile.read(length))
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

            def log_message(self, format: str, *args: str) -> None:
                """Swallow the access log.

                The stdlib handler writes a line per request to STDERR,
                and these tests make hundreds. Under PS 5.1 every one
                comes back from ``make check`` wrapped in a
                NativeCommandError record, which reads as a failure in
                an otherwise passing run.

                Args:
                    format: Printf-style template; ignored.
                    args: Template arguments; ignored.
                """
                _ = (format, args)

        # Threading, not the single-threaded HTTPServer: the caster
        # posts at its full rate and a serial handler blocks the page's
        # next fetch behind the previous one, which deadlocks the test
        # rather than measuring anything.
        class _Server(ThreadingHTTPServer):
            # HTTPServer sets this to 1, and on Windows SO_REUSEADDR
            # does not mean "reuse a socket in TIME_WAIT" -- it means
            # the second bind STEALS the port from the first. Sixteen
            # xdist workers each binding an ephemeral port makes a
            # collision a routine event, and with stealing enabled the
            # loser goes silent instead of failing: its page posts
            # frames that another worker's listener accepts, and it
            # reports "0 frames arrived" about a caster that is working
            # perfectly. That is what failed
            # test_a_repaint_posts_exactly_one_more on 2026-09-04 while
            # the two tests either side of it passed. False turns a
            # collision into a loud bind error.
            allow_reuse_address = False

            # ``ThreadingMixIn.server_close`` JOINS every handler
            # thread, because block_on_close defaults to True. That
            # turned the port-release fix below into a deadlock: a
            # thread parked on one of Chrome's silent spare sockets
            # never returns, so the join never returns, and the suite
            # hung at 99 per cent burning a core (2026-09-04, twice).
            # The handler timeout above bounds that thread, and this
            # makes the close not wait for it either -- belt and
            # braces, because a test teardown must not be able to
            # depend on a remote peer's behaviour. Safe because
            # ThreadingHTTPServer already runs handlers as daemons:
            # nothing outlives the worker.
            block_on_close = False

        self._server = _Server(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        """Shut the listener down and RELEASE THE PORT.

        ``shutdown`` only ends the serve loop; without ``server_close``
        the listening socket stays bound for the life of the worker, so
        every test leaks one more port that the next test's ephemeral
        bind can collide with.
        """
        self._server.shutdown()
        self._server.server_close()

    def await_count(self, page: PageProtocol, wanted: int) -> None:
        """Pump the page until ``wanted`` bodies have arrived.

        A fixed sleep is the wrong instrument here and was measurably
        wrong: 700 ms is ample for one POST on an idle machine and not
        ample under a 16-way xdist run sharing the host with a live
        bot, so the arrival assertions failed on load rather than on
        behaviour. Waiting on the CONDITION removes the machine from
        the test.

        The quiet-window assertions still use a real sleep, because
        "nothing more arrived" is a claim about elapsed time and cannot
        be polled for.

        Args:
            page: The page to keep pumping while waiting.
            wanted: Number of bodies to wait for.

        Raises:
            AssertionError: If they have not arrived within 15 seconds,
                which is a caster that is not posting, not a slow host.
        """
        deadline = _test_hooks.get_current_time_ms() + 15_000
        while len(self.bodies) < wanted:
            if _test_hooks.get_current_time_ms() > deadline:
                raise AssertionError(
                    f"expected {wanted} posted frame(s), got {len(self.bodies)} in 15 s"
                )
            page.wait_for_timeout(50)


class TestCasterAgainstARealBrowser:
    """The caster JS, executed for real, posting to a real listener.

    Everything above asserts substrings or CDP calls. Neither can see
    what the injected script DOES, and what it does is the point: post
    off the Playwright thread, and only when the picture changed.
    """

    @staticmethod
    def _casting_page(browser: BrowserProtocol, port: int) -> PageProtocol:
        """Open a page with one repaintable canvas, casting to ``port``.

        Every caller closes the returned page. The browser is shared by
        the whole module, so a page left open keeps its caster interval
        running for the rest of the session — encoding a JPEG every
        20 ms and posting it at a listener the owning test has already
        shut down.

        Args:
            browser: The shared headless browser.
            port: Intake port the caster will post to.

        Returns:
            The prepared page, with the caster already installed.
        """
        page = browser.new_context().new_page()
        page.goto(
            "data:text/html,<canvas id=c width=64 height=64></canvas>"
            "<script>window.paint=function(v){"
            "const g=document.getElementById('c').getContext('2d');"
            "g.fillStyle='rgb('+v+',20,40)';g.fillRect(0,0,64,64);};"
            "window.paint(10);</script>"
        )
        page.evaluate(build_caster_expression(50.0, 0.8, f"http://127.0.0.1:{port}/cast"))
        return page

    def test_it_posts_a_frame_and_then_stays_quiet(self, headless_browser: BrowserProtocol) -> None:
        """A still canvas costs one POST, not a stream of copies.

        At 50 fps a caster that did not suppress would post about 35
        times across the quiet window, so "still 1" is a real claim
        about the dedup rather than a claim about timing.
        """
        intake = _Intake()
        page = self._casting_page(headless_browser, intake.port)
        try:
            intake.await_count(page, 1)
            page.wait_for_timeout(700)
            posted = len(intake.bodies)
        finally:
            page.close()
            intake.stop()

        assert posted == 1

    def test_a_repaint_posts_exactly_one_more(self, headless_browser: BrowserProtocol) -> None:
        """Change resumes posting, and then it falls quiet again.

        The second half distinguishes suppression from a dead caster: it
        must post on the change and then stop, not post once and never
        again.
        """
        intake = _Intake()
        page = self._casting_page(headless_browser, intake.port)
        try:
            intake.await_count(page, 1)
            page.evaluate("window.paint(200)")
            intake.await_count(page, 2)
            page.wait_for_timeout(700)
            bodies = list(intake.bodies)
        finally:
            page.close()
            intake.stop()

        assert len(bodies) == 2
        assert bodies[0] != bodies[1]

    def test_what_arrives_is_a_bare_jpeg(self, headless_browser: BrowserProtocol) -> None:
        """The body IS the frame — no envelope, no base64.

        The binding channel carried a base64 data URL, which inflated
        every frame by a third and had to be decoded on the bot thread.
        The POST body is the image itself.
        """
        intake = _Intake()
        page = self._casting_page(headless_browser, intake.port)
        try:
            intake.await_count(page, 1)
            body = intake.bodies[0]
        finally:
            page.close()
            intake.stop()

        assert body.startswith(b"\xff\xd8\xff")

    def test_it_keeps_posting_while_the_playwright_thread_is_blocked(
        self, headless_browser: BrowserProtocol
    ) -> None:
        """THE WHOLE POINT, and the regression this change exists to stop.

        The binding delivered frames on the connection Playwright owns,
        dispatched by the thread that runs the tick loop. A heavy tick
        queued every frame produced during it and released them in one
        burst that the latest-wins bus collapsed to one, so seconds of
        play arrived as a single picture.

        Here the Python thread spins on pure CPU -- no Playwright call,
        nothing pumping CDP -- while the page repaints. Frames must
        still arrive, because they no longer travel through Playwright
        at all. Under the binding this count was zero.
        """
        intake = _Intake()
        page = self._casting_page(headless_browser, intake.port)
        try:
            page.evaluate(
                "window.__spin = setInterval(function(){"
                "window.paint((Date.now() / 7) % 250);}, 20);"
            )
            # Confirm the caster is actually posting BEFORE the thread
            # goes deaf, so a zero below means blocked delivery rather
            # than a caster that never started.
            intake.await_count(page, 1)
            before = len(intake.bodies)

            deadline = _test_hooks.get_current_time_ms() + 1500
            total = 0
            while _test_hooks.get_current_time_ms() < deadline:
                total += 1

            during = len(intake.bodies) - before
            page.evaluate("clearInterval(window.__spin)")
        finally:
            page.close()
            intake.stop()

        assert narrow_json_to_int(during) >= 5
