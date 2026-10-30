"""The rebuild: every stage in order, the reboot it waits out, and its refusals.

The fakes answer through ``_test_hooks.run``, ``now`` and ``sleep`` exactly
as :mod:`tests.test_core_io` describes. What each test asserts is the
sequence the rebuild drives and the bytes it ships, composed from the same
renderers production uses; the renderers' own behaviour has its own tests.
"""

from __future__ import annotations

import time

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import FileAsset, HostRunnerSpec
from fleet.core import _test_hooks, remote, runner_audit, runner_rebuild, runner_render
from tests._runner_fixtures import RESERVED_CORES, a_host, an_install, quiet_rebuild_answers
from tests.conftest import FakeClock, FakeRun, FakeSleep, failed, ok

#: Where the base lands on the host.
_BASE_PATH = "/home/corvis/fleet/runner-payloads/fleet-rebuild-base.sh"

#: The base's output when a package asked for a reboot.
_ASKS = ok(f"{runner_render.REBOOT_MARKER}\nbase ready\n")


def _host() -> HostRunnerSpec:
    """A lavender-shaped host: two runners of API and one of chat.

    Returns:
        The spec, with one manual asset. Two installs share a repository, as
        lavender's API runners do, so one token serves both.
    """
    return a_host(
        [
            an_install("wagner-austin/API"),
            an_install("wagner-austin/API", "lavender-2", slot=2),
            an_install("wagner-austin/chat"),
        ],
        assets=[
            FileAsset(
                path="/opt/corvis/rw-game/game-lib.jar",
                sha256=None,
                writable=False,
                reason="the licensed engine",
                manual=True,
                provision_command=None,
            )
        ],
    )


def _clean_transcript(spec: HostRunnerSpec) -> str:
    """One OK line per expected check.

    Args:
        spec: The host.

    Returns:
        The transcript.
    """
    return (
        "\n".join(f"CHECK {check['check_id']} OK" for check in runner_audit.expected_checks(spec))
        + "\n"
    )


def _quiet(spec: HostRunnerSpec) -> list[_test_hooks.CommandResult]:
    """The answers of a rebuild over a host already built.

    Args:
        spec: The host.

    Returns:
        :func:`tests._runner_fixtures.quiet_rebuild_answers` for its two
        repositories and a clean audit.
    """
    return quiet_rebuild_answers(_clean_transcript(spec), spec, repos=2)


def _clock() -> FakeSleep:
    """Bind a fake clock and sleep, so a reboot's wait passes no time.

    Returns:
        The sleep, which records what was waited.
    """
    clock = FakeClock(1_000)
    sleeper = FakeSleep(clock)
    _test_hooks.now = clock
    _test_hooks.sleep = sleeper
    return sleeper


class TestAQuietRebuild:
    """A host whose base is already laid: every stage runs and changes nothing."""

    def test_every_stage_runs_in_order_and_the_audit_is_the_verdict(self) -> None:
        spec = _host()
        runner = FakeRun(_quiet(spec))
        _test_hooks.run = runner
        report = runner_rebuild.rebuild(spec, RESERVED_CORES)
        assert report["steps"] == [
            "base: laid, and nothing asked for a reboot",
            "tokens: minted for 2 repositories",
            "provision: ran for 3 runners",
        ]
        assert report["manual_steps"] == runner_render.manual_steps(spec)
        assert len(report["manual_steps"]) == 1
        assert [finding["ok"] for finding in report["findings"]] == [True] * (
            len(runner_audit.expected_checks(spec)) + 1
        )
        # The base's two, the two mints, the provision's two, the audit's two,
        # then its dropped-job row's two gh lists and its Worker read.
        assert len(runner.calls) == 11

    def test_the_stages_ship_the_rendered_scripts_with_fresh_tokens(self) -> None:
        spec = _host()
        runner = FakeRun(_quiet(spec))
        _test_hooks.run = runner
        runner_rebuild.rebuild(spec, RESERVED_CORES)
        assert runner.stdin[0] == runner_render.render_base_script(spec).encode()
        assert tuple(runner.calls[1][-4:]) == (*remote.ROOT_BASH, _BASE_PATH)
        assert runner.calls[2][4] == "repos/wagner-austin/API/actions/runners/registration-token"
        assert runner.calls[3][4] == "repos/wagner-austin/chat/actions/runners/registration-token"
        tokens = {"wagner-austin/API": "TOKEN0", "wagner-austin/chat": "TOKEN1"}
        assert (
            runner.stdin[4]
            == runner_render.render_provision_script(spec, tokens, RESERVED_CORES).encode()
        )
        assert runner.stdin[6] == runner_audit.render_audit_script(spec).encode()

    def test_the_long_stages_carry_the_stage_deadline(self) -> None:
        spec = _host()
        runner = FakeRun(_quiet(spec))
        _test_hooks.run = runner
        runner_rebuild.rebuild(spec, RESERVED_CORES)
        stage = runner_render.STAGE_TIMEOUT_SECONDS
        short = remote.SSH_TIMEOUT_SECONDS
        assert [runner.timeouts[i] for i in (0, 1, 4, 5, 6, 7)] == [
            short,
            stage,
            short,
            stage,
            short,
            runner_audit.AUDIT_TIMEOUT_SECONDS,
        ]


class TestTheReboot:
    """The base's reboot, waited out by boot id."""

    def test_a_requested_reboot_is_taken_and_waited_out(self) -> None:
        spec = _host()
        sleeper = _clock()
        runner = FakeRun(
            [
                ok(""),
                _ASKS,
                ok("boot-a\n"),  # the boot before
                ok(""),  # the reboot scheduled
                failed(255, "Connection refused"),  # poll 1: down
                ok("boot-a\n"),  # poll 2: not gone down yet
                ok("boot-b\n"),  # poll 3: a new boot
                ok(""),
                ok("base ready\n"),  # the base again, nothing to do
                *_quiet(spec)[2:],
            ]
        )
        _test_hooks.run = runner
        report = runner_rebuild.rebuild(spec, RESERVED_CORES)
        assert report["steps"][0] == "base: laid, and the host rebooted for it (boot boot-b)"
        assert sleeper.slept == [runner_rebuild.REBOOT_POLL_SECONDS] * 3
        assert tuple(runner.calls[2][-2:]) == ("cat", runner_rebuild.BOOT_ID_PATH)
        assert tuple(runner.calls[3][-6:]) == (
            *remote.ROOT_PREFIX,
            "systemd-run",
            f"--on-active={runner_rebuild.REBOOT_DELAY_SECONDS}",
            "/usr/bin/systemctl",
            "reboot",
        )
        assert runner.stdin[7] == runner.stdin[0]

    def test_a_host_that_never_comes_back_is_unreachable_with_the_way_on(self) -> None:
        _clock()
        polls = runner_rebuild.REBOOT_DEADLINE_SECONDS // runner_rebuild.REBOOT_POLL_SECONDS
        _test_hooks.run = FakeRun(
            [
                ok(""),
                _ASKS,
                ok("boot-a\n"),
                ok(""),
                *(failed(255, "Connection timed out") for _ in range(polls)),
            ]
        )
        with pytest.raises(AppError) as fault:
            runner_rebuild.rebuild(_host(), RESERVED_CORES)
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert fault.value.message == (
            "lavender did not answer from a new boot within 1200s of the rebuild's restart "
            "(boot boot-a); re-run --rebuild once it is back, and every stage already done is "
            "skipped"
        )

    def test_a_second_reboot_request_is_refused_rather_than_looped(self) -> None:
        _clock()
        _test_hooks.run = FakeRun(
            [
                ok(""),
                _ASKS,
                ok("boot-a\n"),
                ok(""),
                ok("boot-b\n"),
                ok(""),
                ok(f"installed nvidia-dkms\n{runner_render.REBOOT_MARKER}\n"),
            ]
        )
        with pytest.raises(AppError) as fault:
            runner_rebuild.rebuild(_host(), RESERVED_CORES)
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "asked for a second reboot" in fault.value.message
        assert fault.value.message.endswith(
            f"Its output: installed nvidia-dkms\n{runner_render.REBOOT_MARKER}"
        )


class TestTheRealSleep:
    """The seam's production binding, which the reboot poll waits on."""

    def test_sleep_waits_on_the_real_clock(self) -> None:
        """One second asked for is at least one second passed, measured on
        the monotonic clock so a wall-clock step cannot fake the wait."""
        started = time.monotonic()
        _test_hooks._default_sleep(1)
        assert time.monotonic() - started >= 1.0


class TestTheVerdict:
    """The audit closes the rebuild, and one that cannot run is an error."""

    def test_an_unreachable_audit_raises(self) -> None:
        spec = _host()
        answers = _quiet(spec)
        answers[6:] = [failed(255, "No route to host")]
        _test_hooks.run = FakeRun(answers)
        with pytest.raises(AppError) as fault:
            runner_rebuild.rebuild(spec, RESERVED_CORES)
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert fault.value.message.startswith("the post-rebuild audit could not reach lavender: ")
        assert fault.value.message.endswith(
            "Every stage ran; re-run the audit before trusting the host."
        )

    def test_a_failed_stage_raises_with_its_own_words(self) -> None:
        _test_hooks.run = FakeRun([ok(""), failed(1, "E: Unable to locate package golang-go")])
        with pytest.raises(AppError) as fault:
            runner_rebuild.rebuild(_host(), RESERVED_CORES)
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "E: Unable to locate package golang-go" in fault.value.message
