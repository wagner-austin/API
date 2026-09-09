"""The audit: script rendering, transcript scoring, and the ssh seam.

The fakes here answer through ``_test_hooks.run`` exactly as
:mod:`tests.test_core_io` describes: they satisfy the real protocol, and the
transcript each test scores is one the rendered script's own Emit lines would
produce -- ids taken from :func:`expected_checks`, never retyped.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import (
    FileAsset,
    HostRunnerSpec,
    RunnerInstall,
)
from fleet.core import _test_hooks, runner_audit
from tests.conftest import FakeRun, failed, ok


def _host(
    *,
    keepalive_task: str | None = "wsl-keepalive",
    wslconfig_min_memory_gb: int | None = 26,
    gpu_required: bool = True,
    systemd_timers: list[str] | None = None,
    assets: list[FileAsset] | None = None,
) -> HostRunnerSpec:
    """A host spec with one install, shaped per test.

    Args:
        keepalive_task: The keepalive declaration.
        wslconfig_min_memory_gb: The memory floor declaration.
        gpu_required: Whether the GPU check is declared.
        systemd_timers: The timer declarations, default one.
        assets: The asset declarations, default one pinned writable pair.

    Returns:
        The spec.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=keepalive_task,
        wslconfig_min_memory_gb=wslconfig_min_memory_gb,
        scratch_dir="C:/fleet/stage",
        gpu_required=gpu_required,
        systemd_timers=["ci-clean.timer"] if systemd_timers is None else systemd_timers,
        installs=[
            RunnerInstall(
                repo="wagner-austin/API",
                runner_name="lavender-wsl",
                side="wsl",
                service="actions.runner.wagner-austin-API.lavender-wsl.service",
                workdir="/home/gharunner/actions-runner-api-1/_work",
                labels=["lavender-wsl"],
            )
        ],
        assets=[
            FileAsset(
                path="/opt/corvis/rw-game/game-lib.jar",
                sha256="8a" * 32,
                writable=False,
                reason="the provenance jar",
                manual=True,
                provision_command=None,
            ),
            FileAsset(
                path="/data",
                sha256=None,
                writable=True,
                reason="checkpoint writes",
                manual=False,
                provision_command=None,
            ),
        ]
        if assets is None
        else assets,
    )


def _clean_transcript(spec: HostRunnerSpec) -> str:
    """The transcript of a host that satisfies every check.

    Args:
        spec: The host.

    Returns:
        One OK line per expected check, in order.
    """
    return (
        "\n".join(f"CHECK {check['check_id']} OK" for check in runner_audit.expected_checks(spec))
        + "\n"
    )


class TestExpectedChecks:
    """What the roster promises the script will report."""

    def test_every_declared_concept_yields_its_checks_in_order(self) -> None:
        ids = [check["check_id"] for check in runner_audit.expected_checks(_host())]
        assert ids == [
            "keepalive:wsl-keepalive",
            "memory-floor:26gb",
            "gpu:nvidia-smi",
            "timer:ci-clean.timer",
            "service:wsl:actions.runner.wagner-austin-API.lavender-wsl.service",
            "workdir:wagner-austin/API:wsl:lavender-wsl",
            "asset:/opt/corvis/rw-game/game-lib.jar",
            "sha256:/opt/corvis/rw-game/game-lib.jar",
            "asset:/data",
            "writable:/data",
        ]

    def test_undeclared_concepts_yield_no_checks(self) -> None:
        spec = _host(
            keepalive_task=None,
            wslconfig_min_memory_gb=None,
            gpu_required=False,
            systemd_timers=[],
            assets=[],
        )
        ids = [check["check_id"] for check in runner_audit.expected_checks(spec)]
        assert ids == [
            "service:wsl:actions.runner.wagner-austin-API.lavender-wsl.service",
            "workdir:wagner-austin/API:wsl:lavender-wsl",
        ]


class TestRenderAuditScript:
    """The rendered driver."""

    def test_the_script_emits_every_expected_check_in_order(self) -> None:
        spec = _host()
        script = runner_audit.render_audit_script(spec)
        positions = [
            script.index(f"'{check['check_id']}'") for check in runner_audit.expected_checks(spec)
        ]
        assert positions == sorted(positions)

    def test_the_script_forces_wsl_output_to_utf8(self) -> None:
        assert "$env:WSL_UTF8 = '1'" in runner_audit.render_audit_script(_host())

    def test_the_script_measures_the_vm_not_the_config_file(self) -> None:
        script = runner_audit.render_audit_script(_host(wslconfig_min_memory_gb=26))
        assert "free -m" in script
        assert "-ge 26000" in script
        assert ".wslconfig" not in script

    def test_the_pin_appears_verbatim(self) -> None:
        assert "8a" * 32 in runner_audit.render_audit_script(_host())

    def test_the_script_ends_by_exiting_zero_because_drift_is_data(self) -> None:
        assert runner_audit.render_audit_script(_host()).rstrip().endswith("exit 0")

    @pytest.mark.parametrize("bad", ["it's", 'say "hi"', "a\nb", "tick`", "dollar$"])
    def test_an_unscriptable_roster_value_is_refused(self, bad: str) -> None:
        spec = _host(systemd_timers=[bad])
        with pytest.raises(ValueError, match="cannot be embedded"):
            runner_audit.render_audit_script(spec)


class TestParseAuditTranscript:
    """Scoring, and every way a transcript can fail to be scorable."""

    def test_a_clean_transcript_scores_every_check_ok(self) -> None:
        spec = _host()
        findings = runner_audit.parse_audit_transcript(spec, _clean_transcript(spec))
        assert [finding["ok"] for finding in findings] == [True] * 10
        assert findings[0]["reason"].startswith("the scheduled task")

    def test_a_drift_line_carries_its_detail_and_reason(self) -> None:
        spec = _host()
        lines = _clean_transcript(spec).splitlines()
        lines[2] = "CHECK gpu:nvidia-smi DRIFT nvidia-smi said: "
        findings = runner_audit.parse_audit_transcript(spec, "\n".join(lines))
        drifted = findings[2]
        assert drifted["ok"] is False
        assert drifted["detail"] == "nvidia-smi said: "
        assert drifted["reason"] == "runner jobs on this host digest a real GPU"

    def test_blank_lines_are_not_checks(self) -> None:
        spec = _host()
        transcript = "\n\n" + _clean_transcript(spec) + "\n"
        assert len(runner_audit.parse_audit_transcript(spec, transcript)) == 10

    def test_a_non_check_line_is_unparsable(self) -> None:
        spec = _host()
        with pytest.raises(AppError) as fault:
            runner_audit.parse_audit_transcript(spec, "some stray output\n")
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE

    def test_a_drift_line_with_no_detail_is_unparsable(self) -> None:
        spec = _host()
        lines = _clean_transcript(spec).splitlines()
        lines[0] = "CHECK keepalive:wsl-keepalive DRIFT"
        with pytest.raises(AppError) as fault:
            runner_audit.parse_audit_transcript(spec, "\n".join(lines))
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
        assert "carries no detail" in fault.value.message

    def test_a_truncated_transcript_is_a_script_that_died_midway(self) -> None:
        spec = _host()
        truncated = "\n".join(_clean_transcript(spec).splitlines()[:4])
        with pytest.raises(AppError) as fault:
            runner_audit.parse_audit_transcript(spec, truncated)
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
        assert "died midway" in fault.value.message

    def test_a_reordered_transcript_is_unparsable(self) -> None:
        spec = _host()
        lines = _clean_transcript(spec).splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        with pytest.raises(AppError) as fault:
            runner_audit.parse_audit_transcript(spec, "\n".join(lines))
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE


class TestAttemptAuditHost:
    """The ssh seam: send, run, score."""

    def test_a_reachable_clean_host_yields_findings(self) -> None:
        spec = _host()
        runner = FakeRun([ok(""), ok(_clean_transcript(spec))])
        _test_hooks.run = runner
        outcome = runner_audit.attempt_audit_host(spec)
        assert outcome["reason"] == ""
        assert outcome["findings"] == [
            runner_audit.AuditFinding(
                check_id=check["check_id"], ok=True, detail="", reason=check["reason"]
            )
            for check in runner_audit.expected_checks(spec)
        ]
        sent_argv = runner.calls[0]
        assert sent_argv[0] == "ssh"
        assert "C:/fleet/stage/" + runner_audit.AUDIT_SCRIPT_NAME in " ".join(sent_argv)
        assert runner.stdin[0] == runner_audit.render_audit_script(spec).encode("utf-8")

    def test_an_unreachable_host_is_a_value_not_a_crash(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "No route to host")])
        outcome = runner_audit.attempt_audit_host(_host())
        assert outcome["findings"] is None
        assert "No route to host" in outcome["reason"]

    def test_a_script_that_ran_and_lied_raises_rather_than_scores(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok("nonsense\n")])
        with pytest.raises(AppError) as fault:
            runner_audit.attempt_audit_host(_host())
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
