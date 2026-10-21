"""One netboot arm retirement's pieces (MCPs board task 7b7570c5): the
retirement's command line, and how each way it can end closes its job.
``test_agent_netboot_retire.py`` runs them together through a tick.
"""

from __future__ import annotations

import pathlib
from typing import Final

from fleet.contracts.dispatch import ClosingStatus
from fleet.core import _test_hooks, netboot_arm, netboot_retire
from fleet.core.sealed_release import SealedRelease
from tests.conftest import failed, ok, timed_out

RELEASE: Final = SealedRelease(
    release="7fcc96d6ca7f9057183df144ad8ea2bd",
    revision="bade610b3a91eba2a7bab589770754d61e63a0e5",
    root=pathlib.Path("/home/corvis/.local/share/corvis/netboot-releases/7fcc96d6/MCPs"),
)

OPENING: Final = (
    "fleet-netboot retire of pendragon's arm from sealed release "
    "7fcc96d6ca7f9057183df144ad8ea2bd (revision bade610b3a91eba2a7bab589770754d61e63a0e5)"
)

#: What the command prints for pendragon's retired arm: the key's id, never a key.
RETIRED: Final = (
    "NETBOOT RETIRED host=pendragon mac=d8:5e:d3:84:53:d4 arm=8e6726b1 from=installing "
    'key=kTEST1234CNTRL revoked=2026-10-21T07:26:30Z reason="its ISO download was cut"'
)


def outcome(result: _test_hooks.CommandResult) -> netboot_arm.ArmOutcome:
    """Read a retirement of pendragon's arm from :data:`RELEASE`.

    Args:
        result: How the retirement ended.

    Returns:
        The job's close.
    """
    return netboot_retire.retire_outcome(result, RELEASE, "pendragon")


class TestTheCommandLine:
    def test_the_retirement_runs_the_releases_cli_with_the_root_roster_machine_and_reason(
        self,
    ) -> None:
        place = netboot_arm.NetbootPlace(
            release=RELEASE["root"],
            nodejs="/usr/bin/node",
            served_root=pathlib.Path("/srv/corvis-netboot"),
            env_file=pathlib.Path("/home/corvis/PROJECTS/MCPs/.env"),
            registry=pathlib.Path("/home/corvis/PROJECTS/API/tools/fleet/registry"),
        )
        reason = 'its ISO download was cut; "quoted" and $(not a shell)'

        argv = netboot_retire.retire_argv(place, RELEASE, "pendragon", reason)

        assert argv == netboot_arm.cli_argv(
            place,
            RELEASE,
            "retire",
            "--root",
            "/srv/corvis-netboot",
            "--registry",
            "/home/corvis/PROJECTS/API/tools/fleet/registry",
            "--node",
            "pendragon",
            "--reason",
            reason,
        )
        # The reason is one argv element, exactly as the job carried it.
        assert argv[-1] == reason


class TestTheOutcome:
    def test_a_retirement_that_printed_its_line_passes_with_it(self) -> None:
        assert outcome(ok(f"{RETIRED}\n")) == netboot_arm.ArmOutcome(
            status=ClosingStatus.PASSED, exit_code=0, detail=f"{OPENING}: {RETIRED}"
        )

    def test_every_retired_line_is_carried_when_it_retired_more_than_one_arm(self) -> None:
        second = RETIRED.replace("arm=8e6726b1", "arm=55555555")

        closed = outcome(ok(f"{RETIRED}\n{second}\n"))

        assert closed["status"] is ClosingStatus.PASSED
        assert closed["detail"] == f"{OPENING}: {RETIRED}; {second}"

    def test_an_exit_0_without_a_retired_line_is_not_reported_retired(self) -> None:
        assert outcome(ok("nothing to see\n")) == netboot_arm.ArmOutcome(
            status=ClosingStatus.REFUSED,
            exit_code=None,
            detail=(
                f"{netboot_retire.ANSWER_MALFORMED_CODE}: {OPENING} exited 0 and printed no "
                "'NETBOOT RETIRED' line where it prints one per arm retired, so this runner does "
                "not report it retired"
            ),
        )

    def test_output_carrying_a_key_is_never_copied(self) -> None:
        for result, status, code in [
            (ok("tskey-auth-kTEST-secret\n"), ClosingStatus.REFUSED, None),
            (failed(1, "fleet-netboot: tskey-client-x\n"), ClosingStatus.FAILED, 1),
        ]:
            closed = outcome(result)
            assert (closed["status"], closed["exit_code"]) == (status, code)
            assert closed["detail"].startswith(f"{netboot_retire.KEY_IN_OUTPUT_CODE}: {OPENING}")
            assert "tskey-auth" not in closed["detail"]
            assert "tskey-client" not in closed["detail"]

    def test_a_refusal_raised_before_tailscale_closes_refused_with_the_commands_text(self) -> None:
        for code in netboot_retire.RETIRE_REFUSALS:
            refusal = f"{code}: pendragon (d8:5e:d3:84:53:d4) has no waiting or installing arm"
            assert outcome(failed(1, f"fleet-netboot: {refusal}\n")) == netboot_arm.ArmOutcome(
                status=ClosingStatus.REFUSED, exit_code=None, detail=refusal
            )

    def test_any_other_failure_closes_failed_with_the_outputs_tail(self) -> None:
        error = "fleet-netboot: TAILNET_KEY_REVOKE_FAILED: DELETE answered HTTP 403\n"

        assert outcome(failed(1, error)) == netboot_arm.ArmOutcome(
            status=ClosingStatus.FAILED,
            exit_code=1,
            detail=f"{OPENING} exited 1: {error.strip()}",
        )

    def test_a_retirement_ended_at_its_deadline_closes_failed_saying_so(self) -> None:
        assert outcome(timed_out(netboot_retire.RETIRE_TIMEOUT_SECONDS)) == netboot_arm.ArmOutcome(
            status=ClosingStatus.FAILED,
            exit_code=_test_hooks.TIMED_OUT_RETURNCODE,
            detail=f"{OPENING} exited {_test_hooks.TIMED_OUT_RETURNCODE}: timed out after 300 s",
        )
