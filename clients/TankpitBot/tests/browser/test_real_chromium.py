"""Every case in the suite that needs a real headless Chromium, under one launch.

These cases drive production code against a genuine browser: the static-key
capture reads a script a real page loaded, and the screenshot path decodes
the PNG a real ``Page.captureScreenshot`` returned. Nothing here asserts
anything about launch or close semantics, so they share ONE browser.

WHY ONE MODULE. Until 2026-10-05 they lived in three modules
(``tests/browser/test_lifecycle.py``, ``tests/bot/test_shot_screenshot.py``
and ``tests/diagnostics/test_combat_screenshot.py``), each launching its own
Chromium. ``--dist loadscope`` puts a module on one xdist worker, so a covered
run on austinpc that day started three browsers beside the CPU-bound suite,
and those three workers were three of the four longest: 155 s, 108 s and 84 s
for nine cases whose own work is sub-second, most of it Chromium's teardown
(``browser.close()`` alone was measured from 0.3 s to 219 s on that host). One
module is one worker and one launch. Its cases are module-level functions,
not class methods, because loadscope schedules a class as its own unit and
would hand a second class to a second worker and a second browser.

WHY ONE PAGE. The static-key cases share one page, and the screenshot cases
another; no case opens a page of its own. A page in a fresh context is a
fresh renderer PROCESS, and on austinpc under the full suite's load
``new_page()`` stalled for up to 60.1 s (measured 2026-10-05, 40 fresh
contexts with the rest of the suite running on 15 workers: stalls of 60.1,
15.6, 2.2 and 1.7 s, every other step under 0.03 s); the same 40 cases on one
reused page had no step over 0.03 s. A page per case was what made three of
these cases take about 60 s each in a covered run. Each static-key case sets
the script the shared page is served and navigates it again; routing turns
the HTTP cache off, so every navigation fetches the body its case set.

Three cases were dropped in the move because another case here runs the
same path and asserts more (board task 28e47ae3, A2):
``test_capture_static_key_with_real_headless_browser`` served a key and
checked it came back, which ``test_control_a_served_body_is_saved`` does and
then checks the write; ``test_capture_writes_named_png_when_enabled`` wrote one
named PNG, which ``test_sequence_increments_per_shot`` does twice and checks
both names; ``test_capture_screenshot_png_returns_real_png_bytes`` decoded
one capture, which ``save_screenshot`` performs inside
``test_save_screenshot_writes_real_png_file``, now also checking that the
bytes are more than a PNG signature.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    BrowserProtocol,
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot._test_hooks.cdp import RouteFulfillTarget
from tankpit_bot.bot.base import Bot
from tankpit_bot.browser.lifecycle import _capture_static_key, gather_intel
from tankpit_bot.diagnostics.combat_screenshot import save_screenshot
from tests.conftest import FakeFileSystem

_PAGE_HTML = (
    '<!DOCTYPE html><html><head><script src="/tpclient.js"></script></head><body></body></html>'
)
_TEST_PAGE_URL = "http://localhost:9999/test-page"
_RENDERED_PAGE = "data:text/html,<body style='margin:0;background:#33aa66'>tankpit</body>"
_SHOT_ENV = "TANKPIT_SHOT_SCREENSHOTS"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def chromium() -> Generator[BrowserProtocol, None, None]:
    """Launch the one real headless Chromium this module's cases share.

    Yields:
        A live headless Chromium browser, closed after the module's last case.
    """
    factory = _test_hooks.get_sync_playwright()
    with factory() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


class _IntelPage:
    """The page the static-key cases share, and the script it is served.

    Attributes:
        page: The page, routed to serve ``_PAGE_HTML`` and :attr:`script`.
        cdp: A CDP session attached to it.
        script: The ``tpclient.js`` body the next navigation is served.
    """

    __slots__ = ("cdp", "page", "script")

    def __init__(self, page: PageProtocol, cdp: CDPSessionProtocol) -> None:
        """Hold a routed page and its CDP session, serving no script yet.

        Args:
            page: The routed page.
            cdp: A CDP session attached to it.
        """
        self.page = page
        self.cdp = cdp
        self.script = ""


@pytest.fixture(scope="module")
def intel_page(chromium: BrowserProtocol) -> Generator[_IntelPage, None, None]:
    """Open the one page the static-key cases share, routed once.

    Args:
        chromium: The module's shared browser.

    Yields:
        The page, its CDP session and the script slot each case sets.
    """
    context = chromium.new_context()
    try:
        page = context.new_page()
        shared = _IntelPage(page, context.new_cdp_session(page))

        def _fulfill_page(route: RouteFulfillTarget) -> None:
            route.fulfill(content_type="text/html", body=_PAGE_HTML)

        def _fulfill_tpclient(route: RouteFulfillTarget) -> None:
            route.fulfill(content_type="application/javascript", body=shared.script)

        page.route("**/test-page", _fulfill_page)
        page.route("**/tpclient.js", _fulfill_tpclient)
        yield shared
    finally:
        context.close()


def _serve(intel_page: _IntelPage, js_content: str) -> PageProtocol:
    """Navigate the shared page afresh, serving ``js_content`` as ``tpclient.js``.

    Args:
        intel_page: The module's shared static-key page.
        js_content: Body served for ``tpclient.js`` from this navigation on.

    Returns:
        The page, holding a document loaded with that script.
    """
    intel_page.script = js_content
    intel_page.page.goto(_TEST_PAGE_URL)
    return intel_page.page


@pytest.fixture(scope="module")
def rendered_cdp(chromium: BrowserProtocol) -> Generator[CDPSessionProtocol, None, None]:
    """Attach a CDP session to the one page the screenshot cases share.

    Args:
        chromium: The module's shared browser.

    Yields:
        A live CDP session whose page has painted a solid colour.
    """
    context = chromium.new_context()
    try:
        page = context.new_page()
        page.goto(_RENDERED_PAGE)
        yield context.new_cdp_session(page)
    finally:
        context.close()


def _tpclient_writes(fake_fs: FakeFileSystem) -> list[str]:
    """Name every ``tpclient.js`` the case wrote through the filesystem hook.

    Args:
        fake_fs: The filesystem fixture the case ran under.

    Returns:
        The written paths ending in ``tpclient.js``.
    """
    return [path for path in fake_fs.get_written_files() if path.endswith("tpclient.js")]


@pytest.mark.usefixtures("fake_fs")
def test_gather_intel_with_real_headless_browser(intel_page: _IntelPage) -> None:
    """Real Playwright headless browser runs gather_intel end to end."""
    static_key = "J" * 1000
    page = _serve(intel_page, f'var config = "{static_key}";')
    assert gather_intel(page, intel_page.cdp) == static_key


def test_an_empty_tpclient_body_is_not_saved_over_the_tracked_copy(
    fake_fs: FakeFileSystem,
    intel_page: _IntelPage,
) -> None:
    """Real browser: a tpclient.js that serves nothing is not written.

    The script tag exists and its URL resolves, so the fetch runs and
    legitimately returns the empty string. The checked-in ``tpclient.js`` is
    the reference copy later sessions read, so saving an empty fetch over it
    destroys the artifact -- which is what the old ``else ""`` did with a
    fetch that returned nothing at all.
    """
    page = _serve(intel_page, "")

    assert _capture_static_key(page) is None
    assert _tpclient_writes(fake_fs) == []


def test_control_a_served_body_is_saved(
    fake_fs: FakeFileSystem,
    intel_page: _IntelPage,
) -> None:
    """Control: a served key comes back and its source IS written, so the
    silence above is the check."""
    static_key = "L" * 1000
    page = _serve(intel_page, f'var config = "{static_key}";')

    assert _capture_static_key(page) == static_key
    assert len(_tpclient_writes(fake_fs)) == 1


@pytest.mark.usefixtures("fake_fs")
def test_capture_static_key_no_key_in_content(intel_page: _IntelPage) -> None:
    """Real browser: tpclient.js exists but has no 1000-char string."""
    page = _serve(intel_page, "var x = 1;")
    assert _capture_static_key(page) is None


def test_save_screenshot_writes_real_png_file(
    rendered_cdp: CDPSessionProtocol,
    tmp_path: Path,
) -> None:
    """save_screenshot decodes a real capture and writes it to ``directory/label.png``."""
    result = save_screenshot(rendered_cdp, tmp_path, "shot_0001_x34_y96_id517")

    expected = tmp_path / "shot_0001_x34_y96_id517.png"
    assert result == expected
    written = expected.read_bytes()
    assert written.startswith(_PNG_MAGIC)
    assert len(written) > len(_PNG_MAGIC)


def _env_pointing_to(directory: str | None) -> Callable[[str], str | None]:
    """Build a get_env implementation reporting the screenshot directory.

    Args:
        directory: Value to report for the screenshot env var, or ``None``
            to model the variable being unset.

    Returns:
        A get_env callable returning ``directory`` for the screenshot key
        and ``None`` for every other key.
    """

    def _get(key: str) -> str | None:
        return directory if key == _SHOT_ENV else None

    return _get


@pytest.fixture()
def shot_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Point the shot-screenshot env hook at a temp directory.

    Rebinds the production ``get_env`` hook to report ``tmp_path`` for the
    screenshot variable and restores the prior hook afterwards.

    Yields:
        The temp directory configured as the screenshot output dir.
    """
    previous = _test_hooks.get_env
    _test_hooks.get_env = _env_pointing_to(str(tmp_path))
    try:
        yield tmp_path
    finally:
        _test_hooks.get_env = previous


@pytest.fixture()
def no_shot_env() -> Generator[None, None, None]:
    """Bind the env hook so the shot-screenshot variable reads as unset.

    Yields:
        None; restores the prior hook afterwards.
    """
    previous = _test_hooks.get_env
    _test_hooks.get_env = _env_pointing_to(None)
    try:
        yield
    finally:
        _test_hooks.get_env = previous


def _bot() -> Bot:
    """Build a headless bot instance for screenshot wiring tests.

    Returns:
        A fresh ``Bot`` with no CDP session attached yet.
    """
    return Bot("https://test.tankpit.com/", headless=True)


def test_sequence_increments_per_shot(
    rendered_cdp: CDPSessionProtocol,
    shot_dir: Path,
) -> None:
    """With the env set and a CDP attached, each shot writes its own named PNG."""
    bot = _bot()
    bot._cdp = rendered_cdp

    bot._capture_shot_screenshot(1, 2, 3)
    bot._capture_shot_screenshot(4, 5, 6)

    assert (shot_dir / "shot_0001_x1_y2_id3.png").read_bytes().startswith(_PNG_MAGIC)
    assert (shot_dir / "shot_0002_x4_y5_id6.png").read_bytes().startswith(_PNG_MAGIC)
    assert sorted(path.name for path in shot_dir.glob("*.png")) == [
        "shot_0001_x1_y2_id3.png",
        "shot_0002_x4_y5_id6.png",
    ]


def test_capture_is_noop_when_env_unset(no_shot_env: None) -> None:
    """No screenshot is taken when the opt-in env var is absent."""
    bot = _bot()

    bot._capture_shot_screenshot(1, 2, 3)

    assert bot._shot_screenshot_seq == 0


def test_capture_is_noop_when_cdp_absent(shot_dir: Path) -> None:
    """No screenshot is taken when no CDP session is attached."""
    bot = _bot()
    bot._cdp = None

    bot._capture_shot_screenshot(1, 2, 3)

    assert bot._shot_screenshot_seq == 0
    assert list(shot_dir.glob("*.png")) == []


__all__ = [
    "test_an_empty_tpclient_body_is_not_saved_over_the_tracked_copy",
    "test_capture_is_noop_when_cdp_absent",
    "test_capture_is_noop_when_env_unset",
    "test_capture_static_key_no_key_in_content",
    "test_control_a_served_body_is_saved",
    "test_gather_intel_with_real_headless_browser",
    "test_save_screenshot_writes_real_png_file",
    "test_sequence_increments_per_shot",
]
