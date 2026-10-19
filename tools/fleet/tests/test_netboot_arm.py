"""One netboot arm's pieces (MCPs board task 83102d9d): the key read from a
real env file, the arm's command line, and how each way an arm can end
closes its job. ``test_agent_netboot.py`` runs them together through a tick.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest

from fleet.contracts.dispatch import ClosingStatus
from fleet.core import _test_hooks, netboot_arm
from fleet.core.sealed_release import SealedRelease
from tests.conftest import failed, ok, timed_out

RELEASE: Final = SealedRelease(
    release="7fcc96d6ca7f9057183df144ad8ea2bd",
    revision="bade610b3a91eba2a7bab589770754d61e63a0e5",
    root=pathlib.Path("/home/corvis/.local/share/corvis/netboot-releases/7fcc96d6/MCPs"),
)

OPENING: Final = (
    "fleet-netboot arm of lavender from sealed release 7fcc96d6ca7f9057183df144ad8ea2bd "
    "(revision bade610b3a91eba2a7bab589770754d61e63a0e5)"
)

ARMED: Final = "NETBOOT ARMED host=lavender mac=1c:69:7a:fe:78:9a arm=ded2eb05 key=kTEST1234CNTRL"


def env_file(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    """Write an env file.

    Args:
        tmp_path: The test's temporary directory.
        text: Its contents.

    Returns:
        Its path.
    """
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def outcome(result: _test_hooks.CommandResult) -> netboot_arm.ArmOutcome:
    """Read an arm of lavender from :data:`RELEASE`.

    Args:
        result: How the arm ended.

    Returns:
        The job's close.
    """
    return netboot_arm.arm_outcome(result, RELEASE, "lavender")


class TestTheKey:
    def test_the_one_key_line_is_read_among_the_files_others(self, tmp_path: pathlib.Path) -> None:
        path = env_file(tmp_path, "A=1\nTAILSCALE_API_KEY=tskey-api-abc\r\nB=2\n")

        assert netboot_arm.read_tailnet_key(path) == netboot_arm.TailnetKey(value="tskey-api-abc")

    @pytest.mark.parametrize(
        ("text", "count"),
        [
            ("VM_HOST=lavender-wsl\n", 0),
            ("TAILSCALE_API_KEY=\n", 1),
            ("TAILSCALE_API_KEY=tskey-api-a\nTAILSCALE_API_KEY=tskey-api-b\n", 2),
            # A commented or exported line is not the line the lane reads.
            ("# TAILSCALE_API_KEY=tskey-api-a\nexport TAILSCALE_API_KEY=tskey-api-b\n", 0),
        ],
    )
    def test_anything_but_one_key_that_is_not_empty_is_refused_without_echoing_a_line(
        self, tmp_path: pathlib.Path, text: str, count: int
    ) -> None:
        path = env_file(tmp_path, text)

        detail = netboot_arm.read_tailnet_key(path)

        assert detail == (
            f"{netboot_arm.NO_KEY_CODE}: {path} carries {count} TAILSCALE_API_KEY line(s), and "
            "the arm mints with exactly one that is not empty; nothing ran"
        )
        assert "tskey" not in detail

    def test_an_absent_env_file_is_refused(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "missing.env"

        assert netboot_arm.read_tailnet_key(path) == (
            f"{netboot_arm.NO_KEY_CODE}: {path} does not exist, so no key can be minted; "
            "nothing ran"
        )


class TestTheCommandLine:
    def test_the_arm_runs_the_releases_cli_with_its_definition_roster_and_revision(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Absolute on the platform the suite runs on, as the hub runner's
        # {release} always is: the CLI is imported by its file URI.
        root = tmp_path / "netboot-releases" / RELEASE["release"] / "MCPs"
        served = tmp_path / "srv" / "corvis-netboot"
        place = netboot_arm.NetbootPlace(
            release=root,
            nodejs="/usr/bin/node",
            served_root=served,
            env_file=tmp_path / "MCPs" / ".env",
        )

        release = SealedRelease(release=RELEASE["release"], revision=RELEASE["revision"], root=root)

        argv = netboot_arm.arm_argv(place, release, "lavender")

        cli = (root / "fleet-mcp" / "dist" / "src" / "netboot-cli.js").as_uri()
        assert argv == (
            "/usr/bin/node",
            "--input-type=module",
            "-e",
            f"const m = await import('{cli}'); await m.runFromProcess(process);",
            "arm",
            "--definition",
            str(root / "fleet-mcp" / "netboot"),
            "--root",
            str(served),
            "--cache",
            str(served / "cache"),
            "--commit",
            RELEASE["revision"],
            "--node",
            "lavender",
            "--registry",
            str(root / "fleet-mcp" / "fleet-nodes.json"),
        )
        # The env file's path is the agent's to read; the arm is never told it.
        assert not any(".env" in token for token in argv)


class TestTheOutcome:
    def test_an_arm_that_printed_its_line_passes_with_it(self) -> None:
        assert outcome(ok(f"NETBOOT BUILT digest=4be1\n{ARMED}\n")) == netboot_arm.ArmOutcome(
            status=ClosingStatus.PASSED, exit_code=0, detail=f"{OPENING}: {ARMED}"
        )

    @pytest.mark.parametrize("stdout", ["NETBOOT BUILT digest=4be1\n", f"{ARMED}\n{ARMED}\n"])
    def test_an_exit_0_without_exactly_one_armed_line_is_not_reported_armed(
        self, stdout: str
    ) -> None:
        count = stdout.count("NETBOOT ARMED")

        assert outcome(ok(stdout)) == netboot_arm.ArmOutcome(
            status=ClosingStatus.REFUSED,
            exit_code=None,
            detail=(
                f"{netboot_arm.ANSWER_MALFORMED_CODE}: {OPENING} exited 0 and printed {count} "
                "'NETBOOT ARMED' line(s) where it prints exactly one, so this runner does not "
                "report it armed"
            ),
        )

    @pytest.mark.parametrize(
        ("result", "status", "exit_code"),
        [
            (ok(f"{ARMED}\nkey tskey-auth-kTEST-secret\n"), ClosingStatus.REFUSED, None),
            (failed(1, "fleet-netboot: denied for tskey-api-abc\n"), ClosingStatus.FAILED, 1),
        ],
    )
    def test_output_carrying_a_key_is_never_copied(
        self, result: _test_hooks.CommandResult, status: ClosingStatus, exit_code: int | None
    ) -> None:
        closed = outcome(result)

        assert closed["status"] is status
        assert closed["exit_code"] == exit_code
        assert closed["detail"].startswith(f"{netboot_arm.KEY_IN_OUTPUT_CODE}: {OPENING} exited ")
        assert "tskey-auth" not in closed["detail"]
        assert "tskey-api" not in closed["detail"]

    @pytest.mark.parametrize("code", netboot_arm.ARM_REFUSALS)
    def test_a_refusal_raised_before_minting_closes_refused_with_the_arms_text(
        self, code: str
    ) -> None:
        stderr = f"progress\nfleet-netboot: {code}: lavender, as the arm says\n"

        assert outcome(failed(1, stderr)) == netboot_arm.ArmOutcome(
            status=ClosingStatus.REFUSED,
            exit_code=None,
            detail=f"{code}: lavender, as the arm says",
        )

    def test_any_other_failure_closes_failed_with_the_outputs_tail(self) -> None:
        stderr = "fleet-netboot: NETBOOT_ISO_FETCH_FAILED: " + "x" * 700 + "\n"

        closed = outcome(failed(1, stderr))

        assert closed["status"] is ClosingStatus.FAILED
        assert closed["exit_code"] == 1
        assert closed["detail"].startswith(f"{OPENING} exited 1: ")
        assert len(closed["detail"]) == len(f"{OPENING} exited 1: ") + 600

    def test_a_usage_exit_with_no_cli_error_closes_failed(self) -> None:
        usage = _test_hooks.CommandResult(
            returncode=2, stdout="usage: fleet-netboot build ...\n", stderr="", timed_out=False
        )

        assert outcome(usage) == netboot_arm.ArmOutcome(
            status=ClosingStatus.FAILED,
            exit_code=2,
            detail=f"{OPENING} exited 2: usage: fleet-netboot build ...",
        )

    def test_an_arm_ended_at_its_deadline_closes_failed_saying_so(self) -> None:
        assert outcome(timed_out(netboot_arm.ARM_TIMEOUT_SECONDS)) == netboot_arm.ArmOutcome(
            status=ClosingStatus.FAILED,
            exit_code=_test_hooks.TIMED_OUT_RETURNCODE,
            detail=(f"{OPENING} exited {_test_hooks.TIMED_OUT_RETURNCODE}: timed out after 1800 s"),
        )
