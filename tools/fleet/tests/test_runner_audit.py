"""The audit: which rows a host owes, the verdict each reaches, scoring, and the ssh seam.

The rendered script runs here whole, under a real bash with every command it
asks a stand-in that answers as each case says (tests/_audit_rows.py), against
a host whose work tree, runner ``.path`` and assets are real files in the
case's directory. The scoring and seam cases answer through
``_test_hooks.run`` exactly as :mod:`tests.test_core_io` describes, their
transcripts built from :func:`expected_checks`, never retyped.
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Mapping

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import FileAsset, HostRunnerSpec
from fleet.core import _test_hooks, runner_audit, runner_audit_disk, runner_dropped_check
from fleet.core.linux_isolated_build import EXEC_PATH
from fleet.core.runner_audit_types import AuditFinding, AuditOutcome
from fleet.core.runner_job_scope_render import JOB_SCOPE_PATH
from fleet.core.runner_reaper_render import REAPER_PATH
from tests._audit_rows import run_rows
from tests._host_bash import bash_path, bash_says
from tests._runner_fixtures import a_host, an_install, no_dropped_jobs
from tests.conftest import FakeRun, failed, ok

#: The pinned asset's bytes.
_JAR_BYTES = b"the provenance jar"

#: Every command the audit asks, answering as a clean host does.
_CLEAN: Mapping[str, str] = {
    "free": 'echo "       total   used"; echo "Mem:      31      4"',
    "df": 'echo "Used"; echo " 20G"',
    "du": 'echo "1G	$3"',
    "nvidia-smi": 'echo "NVIDIA GeForce RTX 3090"',
    "systemctl": 'case "$1" in is-enabled) echo enabled ;; is-active) echo active ;; esac',
    "runuser": 'shift 3; "$@"',
    JOB_SCOPE_PATH: "echo placed",
    REAPER_PATH: "echo 0",
}


@pytest.fixture(name="base")
def _base(tmp_path: pathlib.Path) -> str:
    """The case's directory as the host's bash spells it.

    Args:
        tmp_path: The case's directory.

    Returns:
        :func:`tests._host_bash.bash_path` of it, the form every path in
        the host and its rows takes.
    """
    return bash_path(tmp_path)


def _host(tmp_path: pathlib.Path) -> HostRunnerSpec:
    """A host declaring every kind of row, its files laid out in ``tmp_path``.

    Args:
        tmp_path: The case's directory, which holds the install's work tree,
            the ``.path`` its runner gives its jobs (the stand-ins, then the
            system directories their ``env bash`` shebang is found in), and
            both assets.

    Returns:
        One install with a GPU, one timer, a pinned jar and a writable
        directory, every path spelled as the host's bash spells it.
    """
    base = bash_path(tmp_path)
    root = tmp_path / "actions-runner-api-1"
    (root / "_work").mkdir(parents=True)
    (root / ".path").write_text(f"{base}/stand-ins:{EXEC_PATH}")
    (tmp_path / "game-lib.jar").write_bytes(_JAR_BYTES)
    (tmp_path / "data").mkdir()
    install = an_install()
    install["workdir"] = f"{base}/actions-runner-api-1/_work"
    return a_host(
        [install],
        gpu_required=True,
        systemd_timers=["ci-clean.timer"],
        assets=[
            FileAsset(
                path=f"{base}/game-lib.jar",
                sha256=hashlib.sha256(_JAR_BYTES).hexdigest(),
                writable=False,
                reason="the provenance jar",
                manual=True,
                provision_command=None,
            ),
            FileAsset(
                path=f"{base}/data",
                sha256=None,
                writable=True,
                reason="checkpoint writes",
                manual=False,
                provision_command="mkdir -p /data",
            ),
        ],
    )


def _audit(spec: HostRunnerSpec, tmp_path: pathlib.Path, **answers: str) -> dict[str, str]:
    """Run the host's whole audit, each command answering as a clean host's but those named.

    Args:
        spec: The host.
        tmp_path: The case's directory.
        **answers: Stand-in bodies replacing the clean ones, by command name.

    Returns:
        Each ``CHECK`` line's id to the rest of the line, in the order printed.
    """
    rows = run_rows(
        tmp_path, runner_audit.render_audit_script(spec).splitlines(), {**_CLEAN, **answers}
    )
    return {row.split(" ")[1]: row.split(" ", 2)[2] for row in rows}


class TestExpectedChecks:
    """What the roster promises the script will report."""

    def test_every_declared_concept_yields_its_checks_in_order(
        self, tmp_path: pathlib.Path, base: str
    ) -> None:
        ids = [check["check_id"] for check in runner_audit.expected_checks(_host(tmp_path))]
        assert ids == [
            "memory:31gb",
            "disk:/:ceiling-150gb:baseline-18gb@2026-10-20",
            "cache:/home/gharunner/.cache:ceiling-60gb",
            f"cache:{base}/actions-runner-api-1/_work:ceiling-15gb",
            "gpu:wagner-austin/API:lavender",
            "timer:ci-clean.timer",
            "service:actions.runner.wagner-austin-API.lavender.service",
            "workdir:wagner-austin/API:lavender",
            "placement:wagner-austin/API:lavender",
            "orphans:wagner-austin/API:lavender",
            f"asset:{base}/game-lib.jar",
            f"sha256:{base}/game-lib.jar",
            f"asset:{base}/data",
            f"writable:{base}/data",
        ]

    def test_undeclared_concepts_yield_no_checks(self) -> None:
        ids = [check["check_id"] for check in runner_audit.expected_checks(a_host())]
        assert ids == [
            "memory:31gb",
            "disk:/:ceiling-150gb:baseline-18gb@2026-10-20",
            "cache:/home/gharunner/.cache:ceiling-60gb",
            "cache:/home/gharunner/actions-runner-api-1/_work:ceiling-15gb",
            "service:actions.runner.wagner-austin-API.lavender.service",
            "workdir:wagner-austin/API:lavender",
            "placement:wagner-austin/API:lavender",
            "orphans:wagner-austin/API:lavender",
        ]

    def test_the_disk_row_carries_the_ceiling_and_the_idle_baseline(self) -> None:
        assert runner_audit_disk.disk_check_id(a_host()) == (
            "disk:/:ceiling-150gb:baseline-18gb@2026-10-20"
        )

    def test_the_memory_row_carries_the_declared_size(self) -> None:
        assert runner_audit.memory_check_id(a_host()) == "memory:31gb"


class TestTheScriptRunsForReal:
    """The whole audit, under bash, each command a stand-in."""

    def test_a_clean_host_passes_every_row_in_the_declared_order(
        self, tmp_path: pathlib.Path
    ) -> None:
        spec = _host(tmp_path)
        rows = _audit(spec, tmp_path)
        assert list(rows) == [check["check_id"] for check in runner_audit.expected_checks(spec)]
        assert set(rows.values()) == {"OK"}

    def test_a_host_short_of_its_memory_drifts_with_what_free_said(
        self, tmp_path: pathlib.Path
    ) -> None:
        spec = _host(tmp_path)
        rows = _audit(spec, tmp_path, free='echo "Mem:      15      4"')
        assert rows["memory:31gb"] == (
            "DRIFT the host reports 15 GB, below the 30 GB a 31 GB host reports; "
            "free said: Mem:      15      4 (exit 0)"
        )

    def test_a_memory_answer_that_is_not_a_size_never_passes(self, tmp_path: pathlib.Path) -> None:
        spec = _host(tmp_path)
        rows = _audit(spec, tmp_path, free='echo "free: /proc/meminfo: denied"; exit 1')
        assert rows["memory:31gb"].startswith(
            "DRIFT the host reports an unreadable figure GB, below the 30 GB"
        )

    def test_a_root_past_its_ceiling_drifts(self, tmp_path: pathlib.Path) -> None:
        spec = _host(tmp_path)
        rows = _audit(spec, tmp_path, df='echo "Used"; echo " 151G"')
        assert rows[runner_audit_disk.disk_check_id(spec)] == (
            "DRIFT the root uses 151 GB against a ceiling of 150 GB; an idle rebuilt host used "
            "18 GB on 2026-10-20; df said: Used  151G (exit 0)"
        )

    @pytest.mark.parametrize(
        "smi",
        ["exit 0", 'echo "NVIDIA-SMI has failed: no devices were found"; exit 9'],
        ids=["silent", "failed"],
    )
    def test_a_runner_path_with_no_card_drifts(self, tmp_path: pathlib.Path, smi: str) -> None:
        spec = _host(tmp_path)
        rows = _audit(spec, tmp_path, **{"nvidia-smi": smi})
        assert rows["gpu:wagner-austin/API:lavender"].startswith(
            "DRIFT nvidia-smi on the runner PATH said: "
        )

    def test_a_disabled_timer_and_an_inactive_runner_drift(self, tmp_path: pathlib.Path) -> None:
        spec = _host(tmp_path)
        rows = _audit(
            spec,
            tmp_path,
            systemctl='case "$1" in is-enabled) echo disabled ;; *) echo inactive; exit 3 ;; esac',
        )
        assert rows["timer:ci-clean.timer"] == (
            "DRIFT systemctl is-enabled said: disabled (exit 0)"
        )
        assert rows["service:actions.runner.wagner-austin-API.lavender.service"] == (
            "DRIFT systemctl is-active said: inactive (exit 3)"
        )

    def test_a_missing_work_tree_drifts(self, tmp_path: pathlib.Path, base: str) -> None:
        spec = _host(tmp_path)
        (tmp_path / "actions-runner-api-1" / "_work").rmdir()
        rows = _audit(spec, tmp_path)
        assert rows["workdir:wagner-austin/API:lavender"] == (
            f"DRIFT test -d {base}/actions-runner-api-1/_work:  (exit 1)"
        )

    def test_an_unplaced_unit_and_a_leftover_drift(self, tmp_path: pathlib.Path) -> None:
        spec = _host(tmp_path)
        rows = _audit(
            spec, tmp_path, **{JOB_SCOPE_PATH: "echo unplaced; exit 1", REAPER_PATH: "echo 2"}
        )
        assert rows["placement:wagner-austin/API:lavender"].startswith("DRIFT ")
        assert rows["placement:wagner-austin/API:lavender"].endswith(
            "the helper read: unplaced (exit 1)"
        )
        assert rows["orphans:wagner-austin/API:lavender"].endswith("the reaper counted: 2 (exit 0)")

    def test_a_missing_asset_drifts_on_presence_and_on_its_pin(
        self, tmp_path: pathlib.Path, base: str
    ) -> None:
        spec = _host(tmp_path)
        (tmp_path / "game-lib.jar").unlink()
        rows = _audit(spec, tmp_path)
        assert rows[f"asset:{base}/game-lib.jar"] == (
            f"DRIFT test -e {base}/game-lib.jar:  (exit 1)"
        )
        assert rows[f"sha256:{base}/game-lib.jar"].startswith("DRIFT sha256sum said: ")

    def test_an_asset_of_other_bytes_drifts_on_its_pin(
        self, tmp_path: pathlib.Path, base: str
    ) -> None:
        spec = _host(tmp_path)
        (tmp_path / "game-lib.jar").write_bytes(b"another jar")
        rows = _audit(spec, tmp_path)
        said = bash_says('sha256sum "$1"', f"{base}/game-lib.jar")
        assert said.startswith(f"{hashlib.sha256(b'another jar').hexdigest()} ")
        assert rows[f"asset:{base}/game-lib.jar"] == "OK"
        assert rows[f"sha256:{base}/game-lib.jar"] == f"DRIFT sha256sum said: {said} (exit 0)"

    def test_an_asset_the_runner_account_cannot_write_drifts(
        self, tmp_path: pathlib.Path, base: str
    ) -> None:
        spec = _host(tmp_path)
        rows = _audit(spec, tmp_path, runuser='echo "$*" > "$(dirname "$0")/asked"; exit 1')
        assert rows[f"writable:{base}/data"] == (
            f"DRIFT test -w {base}/data as gharunner:  (exit 1)"
        )
        assert (tmp_path / "stand-ins" / "asked").read_text() == (
            f"-u gharunner -- test -w {base}/data\n"
        )

    @pytest.mark.parametrize("bad", ["it's", 'say "hi"', "a\nb", "tick`", "dollar$"])
    def test_an_unscriptable_roster_value_is_refused(self, bad: str) -> None:
        with pytest.raises(ValueError, match="cannot be embedded"):
            runner_audit.render_audit_script(a_host(systemd_timers=[bad]))


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


class TestAttemptAuditHost:
    """The ssh seam: send, run as root, score, then the hub's dropped-job row."""

    def test_a_reachable_clean_host_yields_findings(self) -> None:
        spec = a_host()
        runner = FakeRun([ok(""), ok(_clean_transcript(spec)), *no_dropped_jobs(spec)])
        _test_hooks.run = runner
        outcome = runner_audit.attempt_audit_host(spec)
        dropped = AuditFinding(
            check_id="dropped-jobs:lavender",
            ok=True,
            detail="examined 1 runner(s) against GitHub; none holds a Runner.Worker over "
            "300 s that GitHub reads not busy or unlisted, or that a newer Worker "
            "superseded",
            reason=runner_dropped_check.DROPPED_REASON,
        )
        assert outcome == AuditOutcome(
            findings=[
                *(
                    AuditFinding(
                        check_id=check["check_id"], ok=True, detail="", reason=check["reason"]
                    )
                    for check in runner_audit.expected_checks(spec)
                ),
                dropped,
            ],
            reason="",
        )
        path = f"/home/corvis/fleet/runner-payloads/{runner_audit.AUDIT_SCRIPT_STEM}.sh"
        assert runner.stdin[0] == runner_audit.render_audit_script(spec).encode("utf-8")
        assert path in " ".join(runner.calls[0])
        assert tuple(runner.calls[1][-4:]) == ("sudo", "-n", "/bin/bash", path)

    def test_an_unreachable_host_is_a_value_not_a_crash(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "No route to host")])
        outcome = runner_audit.attempt_audit_host(a_host())
        assert outcome["findings"] is None
        assert "No route to host" in outcome["reason"]

    def test_a_host_lost_before_its_worker_read_is_unreachable_not_half_scored(self) -> None:
        """The script answered, then ssh could not reach the host for the
        dropped-job row's read: reporting the script's rows alone would read
        as a host whose every row was examined."""
        spec = a_host()
        _test_hooks.run = FakeRun(
            [ok(""), ok(_clean_transcript(spec)), ok(""), failed(255, "Connection reset")]
        )
        outcome = runner_audit.attempt_audit_host(spec)
        assert outcome["findings"] is None
        assert "Connection reset" in outcome["reason"]

    def test_a_script_that_ran_and_lied_raises_rather_than_scores(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok("nonsense\n")])
        with pytest.raises(AppError) as fault:
            runner_audit.attempt_audit_host(a_host())
        assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
