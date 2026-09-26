"""The rebuild: every stage in order, the reboot it waits out, and its refusals.

The fakes answer through ``_test_hooks.run``, ``now`` and ``sleep`` exactly
as :mod:`tests.test_core_io` describes. What each test asserts is the
sequence the rebuild drives and the bytes it ships, composed from the same
renderers production uses; the renderers' own content has its own tests.
"""

from __future__ import annotations

import time

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import FileAsset, HostRunnerSpec, RunnerInstall
from fleet.core import (
    _test_hooks,
    remote,
    runner_audit,
    runner_base_render,
    runner_rebuild,
    runner_render,
)
from tests._runner_fixtures import a_base, quiet_rebuild_answers
from tests.conftest import FakeClock, FakeRun, failed, ok


def _host() -> HostRunnerSpec:
    """A lavender-shaped host: two WSL installs of one repo, one Windows install.

    Returns:
        The spec, with one manual asset. Two installs share a repository, as
        lavender's two API runners do, so one token serves both.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=None,
        wslconfig_min_memory_gb=None,
        scratch_dir="C:/fleet/stage",
        gpu_required=False,
        systemd_timers=[],
        installs=[
            RunnerInstall(
                repo="wagner-austin/API",
                runner_name="lavender-wsl",
                side="wsl",
                service="actions.runner.wagner-austin-API.lavender-wsl.service",
                workdir="/home/gharunner/actions-runner-api-1/_work",
                labels=["lavender-wsl"],
                python_toolcache=[],
            ),
            RunnerInstall(
                repo="wagner-austin/API",
                runner_name="lavender-wsl-2",
                side="wsl",
                service="actions.runner.wagner-austin-API.lavender-wsl-2.service",
                workdir="/home/gharunner/actions-runner-api-2/_work",
                labels=["lavender-wsl"],
                python_toolcache=[],
            ),
            RunnerInstall(
                repo="wagner-austin/chat",
                runner_name="lavender",
                side="windows",
                service="actions.runner.wagner-austin-chat.lavender",
                workdir="C:/actions-runner-chat/_work",
                labels=["lavender"],
                python_toolcache=["3.11.9"],
            ),
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
        base=a_base(),
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


class FakeSleep:
    """A sleep that moves a :class:`FakeClock` instead of waiting.

    Satisfies :class:`~fleet.core._test_hooks.SleepProtocol`.

    Attributes:
        clock: The clock each sleep advances.
        slept: Every wait asked for, in order.
    """

    clock: FakeClock
    slept: list[int]

    def __init__(self, clock: FakeClock) -> None:
        """Bind the clock.

        Args:
            clock: The clock to advance.
        """
        self.clock = clock
        self.slept = []

    def __call__(self, seconds: int) -> None:
        """Advance the clock by the wait.

        Args:
            seconds: How long the caller asked to wait.
        """
        self.slept.append(seconds)
        self.clock.seconds += seconds


class TestAQuietRebuild:
    """A host whose base is already laid: every stage runs and changes nothing."""

    def test_every_stage_runs_in_order_and_the_audit_is_the_verdict(self) -> None:
        spec = _host()
        runner = FakeRun(quiet_rebuild_answers(_clean_transcript(spec), repos=2))
        _test_hooks.run = runner
        report = runner_rebuild.rebuild(spec)
        assert report["steps"] == [
            "windows base: in place",
            "distro: already registered",
            "wsl.conf: in place",
            "linux base: linux base ready",
            "tokens: minted for 2 repositories",
            "provision.ps1: ran for 1 Windows-side repositories",
            "provision.sh: ran for 1 WSL-side repositories",
        ]
        assert report["manual_steps"] == runner_render.render_provision(spec)["manual_steps"]
        assert all(finding["ok"] for finding in report["findings"])
        assert len(runner.calls) == 19

    def test_the_stages_ship_the_rendered_scripts_with_fresh_tokens(self) -> None:
        spec = _host()
        runner = FakeRun(quiet_rebuild_answers(_clean_transcript(spec), repos=2))
        _test_hooks.run = runner
        runner_rebuild.rebuild(spec)
        rendered = runner_render.render_provision(spec)
        assert runner.stdin[0] == runner_base_render.render_windows_base_script(spec).encode()
        assert runner.stdin[2] == runner_base_render.render_import_script(spec).encode()
        assert runner.stdin[4] == runner_base_render.render_wslconf_script().encode()
        assert runner.stdin[7] == runner_base_render.render_linux_base_script(spec).encode()
        assert runner.calls[10][:3] == ("gh", "api", "-X")
        assert (
            runner.stdin[12]
            == "\n".join(["$env:RUNNER_TOKEN_CHAT = 'TOKEN1'", rendered["windows_script"]]).encode()
        )
        assert (
            runner.stdin[14]
            == "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "export RUNNER_TOKEN_API='TOKEN0'",
                    rendered["linux_script"],
                ]
            ).encode()
        )

    def test_the_long_stages_carry_the_stage_deadline(self) -> None:
        spec = _host()
        runner = FakeRun(quiet_rebuild_answers(_clean_transcript(spec), repos=2))
        _test_hooks.run = runner
        runner_rebuild.rebuild(spec)
        stage = runner_rebuild.STAGE_TIMEOUT_SECONDS
        short = remote.SSH_TIMEOUT_SECONDS
        # Each send is short; the runs of the windows base, the import, the
        # linux base and both provision scripts are long; wsl.conf is short.
        assert runner.timeouts[1] == stage
        assert runner.timeouts[3] == stage
        assert runner.timeouts[6] == short
        assert runner.timeouts[9] == stage
        assert runner.timeouts[13] == stage
        assert runner.timeouts[16] == stage
        assert runner.timeouts[0] == short

    def test_an_imported_distro_is_named_in_its_step(self) -> None:
        spec = _host()
        answers = quiet_rebuild_answers(_clean_transcript(spec), repos=2)
        answers[3] = ok("imported Ubuntu 24.04-20240423\n")
        _test_hooks.run = FakeRun(answers)
        report = runner_rebuild.rebuild(spec)
        assert report["steps"][1] == "distro: imported Ubuntu 24.04-20240423"


class TestWslConf:
    """A rewritten wsl.conf restarts the distro before anything runs in it."""

    def test_a_changed_wsl_conf_terminates_the_distro(self) -> None:
        spec = _host()
        answers = quiet_rebuild_answers(_clean_transcript(spec), repos=2)
        answers[6] = ok(runner_base_render.WSLCONF_CHANGED_MARKER + "\n")
        answers[7:7] = [ok(""), ok("")]
        runner = FakeRun(answers)
        _test_hooks.run = runner
        report = runner_rebuild.rebuild(spec)
        assert report["steps"][2] == "wsl.conf: written, and the distro restarted into systemd"
        assert runner.stdin[7] == b"wsl --terminate 'Ubuntu'\nexit $LASTEXITCODE\n"


class TestTheReboot:
    """The Windows base's restart, waited out by boot instant."""

    def test_a_requested_reboot_is_taken_and_waited_out(self) -> None:
        spec = _host()
        clock = FakeClock(1_000)
        sleeper = FakeSleep(clock)
        _test_hooks.now = clock
        _test_hooks.sleep = sleeper
        quiet = quiet_rebuild_answers(_clean_transcript(spec), repos=2)
        answers = [
            ok(""),
            ok(runner_base_render.REBOOT_MARKER + "\n"),  # windows base asks
            ok(""),
            ok("2026-09-26T13:00:00.0000000Z\n"),  # boot instant before
            ok(""),
            ok(""),  # restart scheduled
            failed(255, "Connection refused"),  # poll 1: down
            ok(""),
            ok("2026-09-26T13:00:00.0000000Z\n"),  # poll 2: not gone down yet
            ok(""),
            ok("2026-09-26T13:03:00.0000000Z\n"),  # poll 3: a new boot
            ok(""),
            ok(""),  # windows base again, nothing to do
            *quiet[2:],
        ]
        runner = FakeRun(answers)
        _test_hooks.run = runner
        report = runner_rebuild.rebuild(spec)
        assert report["steps"][0] == (
            "windows base: laid, and the host restarted for it "
            "(booted 2026-09-26T13:03:00.0000000Z)"
        )
        assert sleeper.slept == [runner_rebuild.REBOOT_POLL_SECONDS] * 3
        assert runner.stdin[4] == (
            b"shutdown.exe /r /t 10 /c 'fleet-runners --rebuild'\nexit $LASTEXITCODE\n"
        )

    def test_a_host_that_never_comes_back_is_unreachable_with_the_way_on(self) -> None:
        spec = _host()
        clock = FakeClock(1_000)
        _test_hooks.now = clock
        _test_hooks.sleep = FakeSleep(clock)
        polls = runner_rebuild.REBOOT_DEADLINE_SECONDS // runner_rebuild.REBOOT_POLL_SECONDS
        _test_hooks.run = FakeRun(
            [
                ok(""),
                ok(runner_base_render.REBOOT_MARKER + "\n"),
                ok(""),
                ok("2026-09-26T13:00:00.0000000Z\n"),
                ok(""),
                ok(""),
                *(failed(255, "Connection timed out") for _ in range(polls)),
            ]
        )
        with pytest.raises(AppError) as fault:
            runner_rebuild.rebuild(spec)
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "did not answer from a new boot" in fault.value.message
        assert "re-run --rebuild" in fault.value.message

    def test_a_second_reboot_request_is_refused_rather_than_looped(self) -> None:
        spec = _host()
        clock = FakeClock(1_000)
        _test_hooks.now = clock
        _test_hooks.sleep = FakeSleep(clock)
        _test_hooks.run = FakeRun(
            [
                ok(""),
                ok(runner_base_render.REBOOT_MARKER + "\n"),
                ok(""),
                ok("2026-09-26T13:00:00.0000000Z\n"),
                ok(""),
                ok(""),
                ok(""),
                ok("2026-09-26T13:03:00.0000000Z\n"),
                ok(""),
                ok("enabled Windows feature X\n" + runner_base_render.REBOOT_MARKER + "\n"),
            ]
        )
        with pytest.raises(AppError) as fault:
            runner_rebuild.rebuild(spec)
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "second restart" in fault.value.message
        assert "enabled Windows feature X" in fault.value.message


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
        answers = quiet_rebuild_answers(_clean_transcript(spec), repos=2)
        answers[-2:] = [failed(255, "No route to host")]
        _test_hooks.run = FakeRun(answers)
        with pytest.raises(AppError) as fault:
            runner_rebuild.rebuild(spec)
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "post-rebuild audit" in fault.value.message

    def test_a_failed_stage_raises_with_its_own_words(self) -> None:
        _test_hooks.run = FakeRun([ok(""), failed(1, "rootfs sha256 00 does not match the pin")])
        with pytest.raises(AppError) as fault:
            runner_rebuild.rebuild(_host())
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "does not match the pin" in fault.value.message
