"""The production hook bindings.

The file and report implementations are platform_core's and are tested
against the real filesystem and stdout in its suite
(``test_journal_cursor.py``, ``test_report_line.py``). What is this
package's own is that production binds exactly those, that a reset restores
them after a test rebinds, and the ssh seam's real process runner, which is
run here against a real process.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from platform_core.journal_cursor import file_is_present, read_file_bytes, write_file_text
from platform_core.mcp_client import urllib_mcp_post
from platform_core.report_line import emit_line

from lock_wake import _test_hooks
from tests.conftest import FakeSsh, install_ssh


class TestDefaults:
    def test_the_production_bindings_are_platform_cores(self) -> None:
        def _discard(line: str) -> None:
            del line

        _test_hooks.emit = _discard
        install_ssh(FakeSsh("diphtheria", {}, None))
        _test_hooks.reset_hooks()
        assert _test_hooks.http_post is urllib_mcp_post
        assert _test_hooks.read_bytes is read_file_bytes
        assert _test_hooks.write_text is write_file_text
        assert _test_hooks.file_exists is file_is_present
        assert _test_hooks.emit is emit_line
        assert _test_hooks.run_ssh is _test_hooks.run_ssh_command


class TestRunSshCommand:
    def test_captures_bytes_and_returns_a_nonzero_status_without_raising(self) -> None:
        completed = _test_hooks.run_ssh_command(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'12\\n'); "
                "sys.stderr.buffer.write(b'refused'); sys.exit(255)",
            ],
            60,
        )
        assert completed.returncode == 255
        assert completed.stdout == b"12\n"
        assert completed.stderr == b"refused"

    def test_a_process_past_its_deadline_is_killed_and_raises(self) -> None:
        with pytest.raises(subprocess.TimeoutExpired):
            _test_hooks.run_ssh_command([sys.executable, "-c", "import time; time.sleep(30)"], 1)
