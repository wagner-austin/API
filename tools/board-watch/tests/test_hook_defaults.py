"""The production hook implementations, exercised for real.

The fakes elsewhere assert what this package ASKS FOR. Nothing there would
notice if the real environment reader returned a blank string, or if the
emitter buffered its output forever -- so these run the real implementations
against a real directory and real standard output.

THE NETWORK SEAM IS NOT RE-TESTED HERE. Its implementation moved to
``platform_core.mcp_client.urllib_mcp_post`` on 2026-09-05 and is exercised
against a real socket in that library's own suite, including the 401 case
that the pass-through error processor exists for. What is still this
package's business is that the seam is BOUND to it -- a hook pointing at
something else would pass every fake-driven test in this directory -- and
that is the one assertion below.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.config import config_test_hooks
from platform_core.mcp_client import urllib_mcp_post

from board_watch import _test_hooks


def test_the_network_seam_is_bound_to_the_shared_poster() -> None:
    """Production binds the real thing; a test binds a fake and restores this.

    Asserted by identity rather than by behaviour because behaviour is the
    library's to prove. What can go wrong HERE is the binding: a seam left
    pointing at a fake, or at a second poster somebody added locally.
    """
    assert _test_hooks.http_post is urllib_mcp_post


def test_the_real_environment_reader_normalises_blank_to_unset() -> None:
    """It delegates to the monorepo's one permitted environment reader.

    Rebinding that reader's own hook rather than setting a real variable is
    what keeps this package from growing a second ``os.environ`` access, and
    it exercises the delegation rather than assuming it.
    """
    config_test_hooks.get_env = {"SET": "present", "BLANK": "   "}.get
    assert _test_hooks.env("SET") == "present"
    assert _test_hooks.env("BLANK") is None
    assert _test_hooks.env("ABSENT") is None


def test_the_real_emitter_writes_a_line_and_flushes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Monitor reads this stream, so a buffered line is an event that has not
    happened yet as far as the subscriber is concerned."""
    _test_hooks.emit("one")
    _test_hooks.emit("two")
    assert capsys.readouterr().out == "one\ntwo\n"


def test_the_real_writer_creates_missing_parents(tmp_path: pathlib.Path) -> None:
    """The default state directory does not exist on a fresh machine."""
    target = tmp_path / "a" / "b" / "c.json"
    _test_hooks.write_text(target, "body")
    assert target.read_text(encoding="utf-8") == "body"
    assert _test_hooks.read_text(target) == "body"


def test_the_real_existence_check_distinguishes_a_directory(
    tmp_path: pathlib.Path,
) -> None:
    """A directory at the cursor's path is not a cursor document."""
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    assert _test_hooks.file_exists(directory) is False
    assert _test_hooks.file_exists(tmp_path / "absent.json") is False


__all__ = [
    "test_the_network_seam_is_bound_to_the_shared_poster",
    "test_the_real_emitter_writes_a_line_and_flushes",
    "test_the_real_environment_reader_normalises_blank_to_unset",
    "test_the_real_existence_check_distinguishes_a_directory",
    "test_the_real_writer_creates_missing_parents",
]
