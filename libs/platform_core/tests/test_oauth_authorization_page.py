"""Tests for showing the user a desktop sign-in's authorization page."""

from __future__ import annotations

import subprocess
import sys
import webbrowser
from pathlib import Path

from platform_core.oauth import open_authorization_page, open_in_browser

URL = "https://accounts.example.com/o/oauth2/auth?state=abc"


def test_the_url_is_printed_before_the_browser_opens_it() -> None:
    printed: list[str] = []
    opened: list[str] = []

    def open_browser(url: str) -> bool:
        opened.append(url)
        return True

    open_authorization_page(URL, open_browser=open_browser, console_output=printed.append)

    assert printed == [f"Open this URL to authorize:\n{URL}"]
    assert opened == [URL]


def test_a_browser_that_did_not_open_is_said_in_so_many_words() -> None:
    printed: list[str] = []

    open_authorization_page(URL, open_browser=lambda _url: False, console_output=printed.append)

    assert printed == [
        f"Open this URL to authorize:\n{URL}",
        "No browser opened it; open the URL above yourself.",
    ]


def test_open_in_browser_hands_the_url_to_the_preferred_browser(tmp_path: Path) -> None:
    """webbrowser opens the URL with the browser registered first.

    Here that is a real program that records the URL it was given, so the
    host's own browser is never started: a real webbrowser.open on colossus
    started a snap firefox that outlived its fleet run by a week (measured
    2026-10-18, MCPs board task 346abc45). webbrowser keeps the registration
    for the rest of this process; nothing else in the suite opens a browser.
    """
    received = tmp_path / "received"
    recorder = tmp_path / "recorder.py"
    recorder.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')\n",
        encoding="utf-8",
    )
    command = [sys.executable, str(recorder), str(received), "%s"]
    webbrowser.register(
        "platform-core-recorder", None, webbrowser.GenericBrowser(command), preferred=True
    )

    assert open_in_browser(URL) is True
    assert received.read_text(encoding="utf-8") == URL


def test_open_in_browser_says_so_when_the_browser_did_not_open(tmp_path: Path) -> None:
    """A browser program that fails makes open_in_browser answer False.

    The check review of audit ac5813df measured that an opener answering
    True whatever webbrowser.open said passed every case above (MCPs board
    task 41c0a997). webbrowser.open tries each registered browser in turn,
    and on Windows the platform's own default is always among them, so a
    failing browser registered here would fall through to a real one. So the
    case runs in a child interpreter whose try order holds only a program
    that exits 1, and this process's webbrowser is left as it was.
    """
    child = tmp_path / "child.py"
    child.write_text(
        "import sys, webbrowser\n"
        "from platform_core.oauth import open_in_browser\n"
        "failing = webbrowser.GenericBrowser([sys.executable, '-c', 'raise SystemExit(1)', '%s'])\n"
        "webbrowser.register('platform-core-failing', None, failing, preferred=True)\n"
        "webbrowser._tryorder = ['platform-core-failing']\n"
        "print(open_in_browser(sys.argv[1]))\n",
        encoding="utf-8",
    )

    ran = subprocess.run(
        [sys.executable, str(child), URL],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )

    assert (ran.returncode, ran.stdout, ran.stderr) == (0, "False\n", "")
