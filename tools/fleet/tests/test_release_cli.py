"""A sealed release's own node CLI (MCPs board task 3e9c99af): how its command
is composed, how much of a failed run a detail keeps, and the close shape."""

from __future__ import annotations

import pathlib
import shutil

import pytest
from platform_core.json_utils import load_json_str

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
        # before the lift, then the -- ending node's own options (test claim
        # f6051b29, upheld).
        assert release_cli_argv("/usr/bin/node", MODULE, (), "arm", "--node", "pendragon") == (
            "/usr/bin/node",
            "--input-type=module",
            "-e",
            "const m = await import('file:///srv/releases/fleet-audit/abc/MCPs/fleet-mcp/dist/"
            "src/inspect-cli.js'); await m.runFromProcess(process);",
            "--",
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
        # The CLI's arguments follow the -- that ends node's options (test
        # claim b753494c, upheld).
        assert argv[4:] == ("--", "--probe", "os")


class TestRealNode:
    """The composed command run by real node (MCPs board task c8978826).

    Job 0aad55fd's fleet-inspect exited 9 on ``node: bad option:
    --workspace``: no case had run the argv with node, only with a stand-in
    process. A module recording what ``runFromProcess`` is handed is run
    here with the node the fleet runs MCPs' TypeScript checks with, which
    every node carries.
    """

    @pytest.mark.parametrize(
        "arguments",
        [
            ("--workspace", "/home/corvis/PROJECTS/MCPs", "--node", "pendragon", "--probe", "os"),
            ("arm", "--node", "pendragon"),
        ],
    )
    def test_node_hands_the_cli_its_arguments_whether_a_flag_or_a_command_comes_first(
        self, tmp_path: pathlib.Path, arguments: tuple[str, ...]
    ) -> None:
        node = shutil.which("node")
        assert node is not None, "node runs every MCPs TypeScript check, so every node has it"
        module = tmp_path / "cli.mjs"
        module.write_text(
            "export async function runFromProcess(proc, root) {\n"
            "  proc.stdout.write(JSON.stringify({ argv: proc.argv.slice(1), root }) + '\\n');\n"
            "}\n",
            encoding="utf-8",
        )

        result = _test_hooks.run(
            release_cli_argv(node, module, ("/srv/releases/fleet-audit/abc",), *arguments),
            timeout_seconds=30,
        )

        assert (result["returncode"], result["stderr"]) == (0, "")
        assert load_json_str(result["stdout"]) == {
            "argv": list(arguments),
            "root": "/srv/releases/fleet-audit/abc",
        }


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
