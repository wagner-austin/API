"""The adopt pass's pieces (MCPs board task 49d366ac): its command line, and
how each way ``adopt`` can end reads into the lines a tick logs.
``test_agent_netboot_adopt.py`` runs them together through a tick.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest

from fleet.core import _test_hooks, netboot_adopt, netboot_arm
from fleet.core.sealed_release import SealedRelease
from tests.conftest import failed, ok, timed_out

RELEASE: Final = SealedRelease(
    release="7fcc96d6ca7f9057183df144ad8ea2bd",
    revision="bade610b3a91eba2a7bab589770754d61e63a0e5",
    root=pathlib.Path("/home/corvis/.local/share/corvis/netboot-releases/7fcc96d6/MCPs"),
)

OPENING: Final = (
    "fleet-netboot adopt from sealed release 7fcc96d6ca7f9057183df144ad8ea2bd "
    "(revision bade610b3a91eba2a7bab589770754d61e63a0e5)"
)

ADOPTED: Final = (
    "NETBOOT ADOPTED host=pendragon arm=ded2eb05 device=22 address=100.74.0.62 "
    "readdressed=true retired=11"
)

#: Output that broke the contract and printed a minted key.
LEAKED: Final = f"{ADOPTED} key=tskey-auth-kSECRET\n"


def report(result: _test_hooks.CommandResult) -> tuple[str, ...]:
    """Read an ``adopt`` run from :data:`RELEASE`.

    Args:
        result: How it ended.

    Returns:
        The lines the tick logs.
    """
    return netboot_adopt.adopt_report(result, RELEASE)


class TestTheCommandLine:
    def test_adopt_runs_the_releases_cli_over_the_served_root_and_the_loaded_roster(
        self, tmp_path: pathlib.Path
    ) -> None:
        root = tmp_path / "netboot-releases" / RELEASE["release"] / "MCPs"
        served = tmp_path / "srv" / "corvis-netboot"
        registry = tmp_path / "API" / "tools" / "fleet" / "registry"
        place = netboot_arm.NetbootPlace(
            release=root,
            nodejs="/usr/bin/node",
            served_root=served,
            env_file=tmp_path / "MCPs" / ".env",
            registry=registry,
        )
        release = SealedRelease(release=RELEASE["release"], revision=RELEASE["revision"], root=root)

        argv = netboot_adopt.adopt_argv(place, release)

        cli = (root / "fleet-mcp" / "dist" / "src" / "netboot-cli.js").as_uri()
        assert argv == (
            "/usr/bin/node",
            "--input-type=module",
            "-e",
            f"const m = await import('{cli}'); await m.runFromProcess(process);",
            "adopt",
            "--root",
            str(served),
            "--registry",
            str(registry),
        )
        # The key travels in the environment; the env file is never named.
        assert not any(".env" in token for token in argv)


class TestTheReport:
    def test_every_line_printed_is_reported_with_the_release_and_blank_ones_dropped(
        self,
    ) -> None:
        nothing = "NETBOOT ADOPT nothing pending: 0 of 3 arm(s) checked in within the window"

        assert report(ok(f"{ADOPTED}\n\n{nothing}\n")) == (
            f"{OPENING}: {ADOPTED}",
            f"{OPENING}: {nothing}",
        )

    def test_an_exit_0_that_printed_nothing_says_so(self) -> None:
        assert report(ok(" \n")) == (
            f"{netboot_adopt.ANSWER_EMPTY_CODE}: {OPENING} exited 0 and printed nothing, where "
            "it prints a line for each adoption or one saying none is pending",
        )

    def test_a_failure_is_one_line_with_its_exit_and_the_outputs_tail(self) -> None:
        refusal = "fleet-netboot: TAILNET_DEVICE_DELETE_FAILED: DELETE answered 403"

        assert report(failed(1, f"{refusal}\n")) == (f"{OPENING} exited 1: {refusal}",)

    def test_a_run_ended_at_its_deadline_says_so(self) -> None:
        assert report(timed_out(netboot_adopt.ADOPT_TIMEOUT_SECONDS)) == (
            f"{OPENING} exited {_test_hooks.TIMED_OUT_RETURNCODE}: timed out after 300 s",
        )

    @pytest.mark.parametrize(("result", "returncode"), [(ok(LEAKED), 0), (failed(1, LEAKED), 1)])
    def test_output_carrying_a_key_is_never_copied_whatever_the_exit(
        self, result: _test_hooks.CommandResult, returncode: int
    ) -> None:
        lines = report(result)

        assert lines == (
            f"{netboot_adopt.KEY_IN_OUTPUT_CODE}: {OPENING} exited {returncode} and its "
            "output carries a tailnet key (tskey-...), so none of it is logged; the served "
            "root's adoption records on the netboot host are the record",
        )
        assert "kSECRET" not in "".join(lines)
