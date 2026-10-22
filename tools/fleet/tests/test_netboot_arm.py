"""One netboot arm's pieces (MCPs board task 83102d9d): the key read from a
real env file, the arm's command line, and how each way an arm can end
closes its job. ``test_agent_netboot.py`` runs them together through a tick.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest

from fleet.cli import agent
from fleet.contracts.dispatch import ClosingStatus
from fleet.core import _test_hooks, netboot_arm
from fleet.core.sealed_release import SealedRelease
from tests._netboot_tick import (
    WRITER_ID,
    WRITER_SECRET,
    netboot_place,
    place,
    sealed,
    verify_reply,
)
from tests.conftest import FakeRun, failed, ok, timed_out

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


#: Both writer lines, as the netboot host's env file carries them.
WRITER_LINES: Final = (
    "TAILNET_WRITER_CLIENT_ID=kW\nTAILNET_WRITER_CLIENT_SECRET=tskey-client-kW-s\n"
)


class TestTheWriterClient:
    def test_the_two_writer_lines_are_read_among_the_files_others(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = env_file(
            tmp_path,
            "A=1\nTAILNET_WRITER_CLIENT_ID=kW\r\nTAILSCALE_API_KEY=tskey-api-retired\n"
            "TAILNET_WRITER_CLIENT_SECRET=tskey-client-kW-s\nB=2\n",
        )

        assert netboot_arm.read_tailnet_writer(path) == netboot_arm.TailnetWriter(
            client_id="kW", client_secret="tskey-client-kW-s"
        )
        assert netboot_arm.WRITER_NAMES == (
            "TAILNET_WRITER_CLIENT_ID",
            "TAILNET_WRITER_CLIENT_SECRET",
        )

    def test_the_last_assignment_wins_as_in_the_files_one_reader(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = env_file(tmp_path, WRITER_LINES + "TAILNET_WRITER_CLIENT_SECRET=tskey-client-kW-b\n")

        assert netboot_arm.read_tailnet_writer(path) == netboot_arm.TailnetWriter(
            client_id="kW", client_secret="tskey-client-kW-b"
        )

    @pytest.mark.parametrize(
        ("text", "missing"),
        [
            ("VM_HOST=lavender-wsl\n", "TAILNET_WRITER_CLIENT_ID or TAILNET_WRITER_CLIENT_SECRET"),
            # The retired personal key alone is no credential.
            (
                "TAILSCALE_API_KEY=tskey-api-a\n",
                "TAILNET_WRITER_CLIENT_ID or TAILNET_WRITER_CLIENT_SECRET",
            ),
            (
                "TAILNET_WRITER_CLIENT_ID=kW\nTAILNET_WRITER_CLIENT_SECRET=\n",
                "TAILNET_WRITER_CLIENT_SECRET",
            ),
            # A commented or exported line does not set the id.
            (
                "# TAILNET_WRITER_CLIENT_ID=kA\nexport TAILNET_WRITER_CLIENT_ID=kB\n"
                "TAILNET_WRITER_CLIENT_SECRET=tskey-client-kW-s\n",
                "TAILNET_WRITER_CLIENT_ID",
            ),
        ],
    )
    def test_a_file_that_sets_no_writer_line_is_refused_naming_it_without_echoing_a_line(
        self, tmp_path: pathlib.Path, text: str, missing: str
    ) -> None:
        path = env_file(tmp_path, text)

        detail = netboot_arm.read_tailnet_writer(path)

        assert detail == (
            f"{netboot_arm.NO_KEY_CODE}: {path} sets no {missing}, or sets it empty, and the "
            "netboot duty's commands run with the tailnet writer client; nothing ran"
        )
        assert "tskey" not in detail

    def test_a_line_the_reader_cannot_split_is_refused_by_number_without_echoing_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = env_file(tmp_path, "A=1\ntskey-client-pasted-alone\n")

        detail = netboot_arm.read_tailnet_writer(path)

        assert detail == (
            f"{netboot_arm.NO_KEY_CODE}: line 2 of {path} is not a KEY=VALUE line; only those, "
            "blank lines and # comments are read, and skipping one would drop whichever key it "
            "sets; nothing ran"
        )
        assert "tskey" not in detail

    def test_an_absent_env_file_is_refused(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "missing.env"

        assert netboot_arm.read_tailnet_writer(path) == (
            f"{netboot_arm.NO_KEY_CODE}: {path} does not exist, so no key can be minted; "
            "nothing ran"
        )


#: Both Wi-Fi lines, a made-up network that is never the fleet's own
#: (MCPs board task f78f7e33).
WIFI_LINES: Final = (
    "FLEET_WIFI_SSID=corvis test net\nFLEET_WIFI_PASSPHRASE=made-up-passphrase-0001\n"
)


class TestTheFleetWifi:
    def test_both_lines_are_read_among_the_files_others(self, tmp_path: pathlib.Path) -> None:
        path = env_file(tmp_path, f"A=1\n{WRITER_LINES}{WIFI_LINES}B=2\n")

        assert netboot_arm.read_fleet_wifi(path) == netboot_arm.FleetWifi(
            ssid="corvis test net", passphrase="made-up-passphrase-0001"
        )
        assert netboot_arm.WIFI_NAMES == ("FLEET_WIFI_SSID", "FLEET_WIFI_PASSPHRASE")

    def test_a_file_that_sets_neither_line_gives_no_wifi(self, tmp_path: pathlib.Path) -> None:
        assert netboot_arm.read_fleet_wifi(env_file(tmp_path, WRITER_LINES)) is None
        empty = env_file(tmp_path, f"{WRITER_LINES}FLEET_WIFI_SSID=\nFLEET_WIFI_PASSPHRASE=\n")
        assert netboot_arm.read_fleet_wifi(empty) is None

    @pytest.mark.parametrize(
        ("text", "missing"),
        [
            ("FLEET_WIFI_SSID=corvis test net\n", "FLEET_WIFI_PASSPHRASE"),
            ("FLEET_WIFI_PASSPHRASE=made-up-passphrase-0001\n", "FLEET_WIFI_SSID"),
        ],
    )
    def test_one_line_without_the_other_is_refused_naming_it_without_echoing_a_line(
        self, tmp_path: pathlib.Path, text: str, missing: str
    ) -> None:
        path = env_file(tmp_path, WRITER_LINES + text)

        detail = netboot_arm.read_fleet_wifi(path)

        assert detail == (
            f"{netboot_arm.WIFI_UNUSABLE_CODE}: {path} sets one of FLEET_WIFI_SSID and "
            f"FLEET_WIFI_PASSPHRASE but no {missing}, or sets it empty, so no arm is given half "
            "a network; nothing ran"
        )
        assert "made-up" not in detail
        assert "corvis test net" not in detail

    def test_an_absent_or_unreadable_file_is_refused(self, tmp_path: pathlib.Path) -> None:
        absent = tmp_path / "missing.env"
        assert netboot_arm.read_fleet_wifi(absent) == (
            f"{netboot_arm.WIFI_UNUSABLE_CODE}: {absent} does not exist; nothing ran"
        )
        path = env_file(tmp_path, "A=1\nmade-up-passphrase-pasted-alone\n")
        assert netboot_arm.read_fleet_wifi(path) == (
            f"{netboot_arm.WIFI_UNUSABLE_CODE}: line 2 of {path} is not a KEY=VALUE line; only "
            "those, blank lines and # comments are read, and skipping one would drop whichever "
            "key it sets; nothing ran"
        )

    def test_ready_gives_the_wifi_and_refuses_half_a_network(self, tmp_path: pathlib.Path) -> None:
        paths = place(tmp_path)
        env = paths[agent.ENV_FILE_FLAG]
        release = paths[agent.NETBOOT_RELEASE_FLAG]
        where = netboot_place(paths, tmp_path / "registry")
        env.write_text(env.read_text(encoding="utf-8") + WIFI_LINES, encoding="utf-8")
        _test_hooks.run = FakeRun([verify_reply(release)])

        assert netboot_arm.ready(where) == netboot_arm.Ready(
            release=sealed(paths),
            writer=netboot_arm.TailnetWriter(client_id=WRITER_ID, client_secret=WRITER_SECRET),
            wifi=netboot_arm.FleetWifi(
                ssid="corvis test net", passphrase="made-up-passphrase-0001"
            ),
        )
        env.write_text(f"{WRITER_LINES}FLEET_WIFI_SSID=corvis test net\n", encoding="utf-8")
        _test_hooks.run = FakeRun([verify_reply(release)])
        assert netboot_arm.ready(where) == (
            f"{netboot_arm.WIFI_UNUSABLE_CODE}: {env} sets one of FLEET_WIFI_SSID and "
            "FLEET_WIFI_PASSPHRASE but no FLEET_WIFI_PASSPHRASE, or sets it empty, so no arm is "
            "given half a network; nothing ran"
        )

    def test_the_wifi_reaches_the_commands_environment_after_the_writer_client(self) -> None:
        runner = FakeRun([ok("ok\n"), ok("ok\n")])
        _test_hooks.run = runner
        writer = netboot_arm.TailnetWriter(client_id="kW", client_secret="tskey-client-kW-s")
        wifi = netboot_arm.FleetWifi(ssid="corvis test net", passphrase="made-up-passphrase-0001")

        netboot_arm.run_cli(("node",), writer, wifi=wifi, timeout_seconds=5)
        netboot_arm.run_cli(("node",), writer, wifi=None, timeout_seconds=5)

        writer_env = (
            ("TAILNET_WRITER_CLIENT_ID", "kW"),
            ("TAILNET_WRITER_CLIENT_SECRET", "tskey-client-kW-s"),
        )
        assert runner.set_env == [
            (
                *writer_env,
                ("FLEET_WIFI_SSID", "corvis test net"),
                ("FLEET_WIFI_PASSPHRASE", "made-up-passphrase-0001"),
            ),
            writer_env,
        ]
        assert runner.calls == [("node",), ("node",)]


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
            registry=tmp_path / "API" / "tools" / "fleet" / "registry",
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
            str(tmp_path / "API" / "tools" / "fleet" / "registry"),
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
