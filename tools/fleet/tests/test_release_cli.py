"""A sealed release's own node CLI (MCPs board task 3e9c99af): how its command
is composed, how much of a failed run a detail keeps, and the close shape."""

from __future__ import annotations

import pathlib

from fleet.contracts.dispatch import ClosingStatus
from fleet.core import _test_hooks
from fleet.core.release_cli import (
    DETAIL_TAIL_CHARS,
    ReleaseCliOutcome,
    output_tail,
    release_cli_argv,
)
from tests.conftest import failed

MODULE = pathlib.Path("/srv/releases/fleet-audit/abc/MCPs/fleet-mcp/dist/src/inspect-cli.js")


class TestTheCommand:
    def test_a_cli_needing_nothing_more_is_entered_with_the_process_alone(self) -> None:
        # The netboot CLI's importer, byte for byte as netboot_arm composed it
        # before the lift.
        assert release_cli_argv("/usr/bin/node", MODULE, (), "arm", "--node", "pendragon") == (
            "/usr/bin/node",
            "--input-type=module",
            "-e",
            "const m = await import('file:///srv/releases/fleet-audit/abc/MCPs/fleet-mcp/dist/"
            "src/inspect-cli.js'); await m.runFromProcess(process);",
            "arm",
            "--node",
            "pendragon",
        )

    def test_what_the_entry_is_handed_goes_after_the_process_as_json_strings(self) -> None:
        root = '/srv/releases/it\'s "quoted"'
        argv = release_cli_argv("node", MODULE, (root, "second"), "--probe", "os")

        assert argv[3] == (
            "const m = await import('file:///srv/releases/fleet-audit/abc/MCPs/fleet-mcp/dist/"
            "src/inspect-cli.js'); await m.runFromProcess(process, "
            '"/srv/releases/it\'s \\"quoted\\"", "second");'
        )
        assert argv[4:] == ("--probe", "os")


class TestTheTail:
    def test_a_short_failure_keeps_stdout_then_stderr_stripped(self) -> None:
        result = _test_hooks.CommandResult(
            returncode=1, stdout="  started\n", stderr="the error\n", timed_out=False
        )

        assert output_tail(result) == "started\nthe error"

    def test_a_long_failure_keeps_only_its_last_characters(self) -> None:
        tail = output_tail(failed(1, "x" * (DETAIL_TAIL_CHARS + 50) + "END"))

        assert len(tail) == DETAIL_TAIL_CHARS
        assert tail.endswith("END")
        assert DETAIL_TAIL_CHARS == 600


def test_an_outcome_is_the_three_fields_a_close_reports() -> None:
    assert ReleaseCliOutcome(status=ClosingStatus.FAILED, exit_code=1, detail="d") == {
        "status": ClosingStatus.FAILED,
        "exit_code": 1,
        "detail": "d",
    }
