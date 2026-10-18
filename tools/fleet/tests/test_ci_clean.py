"""ci-clean, the CI runner hosts' daily hygiene (:mod:`fleet.core.ci_clean`).

Its Python half is executed for real against a home this suite lays out:
virtualenvs whose recorded sources are gone or alive, and cache generations
that runner installs' markers name or do not, because a string assertion
would pass a script that deletes everything or nothing.
"""

from __future__ import annotations

import pathlib
import shlex
import subprocess
import time
from typing import Final

import pytest

from fleet.core import ci_clean, names
from fleet.core.runner_tool_caches import RUNNER_VIRTUALENVS
from tests._host_bash import host_bash

#: One day in seconds.
DAY: Final = 86400

#: What each laid-out cache file holds: one MiB.
MIB: Final = 1024 * 1024


def test_no_pass_waits_for_the_runners_to_be_idle() -> None:
    # The gate this replaced skipped the whole Python half while any
    # Runner.Worker ran, and lavender always runs one (MCPs board task
    # ecb13d3d): 30 days of 'skipping this pass' and no reclaim.
    assert "Runner.Worker" not in ci_clean.CI_CLEAN_SCRIPT
    assert "skipping this pass" not in ci_clean.CI_CLEAN_SCRIPT


def test_the_orphan_rule_runs_over_the_runner_account_s_virtualenvs() -> None:
    rule = "\n".join(ci_clean.orphan_sweep_lines(f'"{RUNNER_VIRTUALENVS}"'))
    assert rule in ci_clean.CI_CLEAN_SCRIPT
    assert "unknown provenance: keep" in ci_clean.CI_CLEAN_SCRIPT


def test_the_download_caches_go_only_by_whole_generations() -> None:
    """MCPs board task 4b405239: no pass removes a single cache file, which
    could part a poetry metadata file from its body."""
    caches = ci_clean.CI_CLEAN_CACHES
    assert caches.index(ci_clean.GENERATIONS_SWEEP) < caches.index(ci_clean.GENERATION_READERS)
    assert caches.endswith(ci_clean.GENERATION_READERS + "\n")
    assert ci_clean.GENERATIONS_SWEEP.splitlines()[-1] == (
        'remove_unused "$HOME/.cache/fleet-tools" "$HOME/.cache/fleet-jobs"'
    )
    assert "-atime" not in ci_clean.CI_CLEAN_SCRIPT


def test_the_python_half_follows_the_docker_prunes_inside_the_reported_group() -> None:
    lines = ci_clean.CI_CLEAN_SCRIPT.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert lines.index("set -euo pipefail") < lines.index("{")
    group = lines[lines.index("{") + 1 : lines.index('} | tee "$report.pending"')]
    caches = ci_clean.CI_CLEAN_CACHES.rstrip("\n").splitlines()
    assert group[len(group) - len(caches) :] == caches
    prunes = [index for index, line in enumerate(group) if line.startswith("docker ")]
    assert len(prunes) == 5
    assert max(prunes) < len(group) - len(caches)


def _commands(lines: list[str]) -> list[str]:
    """The lines of a script that are neither blank nor comments.

    Args:
        lines: The script's lines.

    Returns:
        Those lines, in order.
    """
    return [line for line in lines if line and not line.startswith("#")]


def test_every_run_reports_its_digest_its_start_and_the_build_cache_around_the_prunes() -> None:
    """MCPs board tasks 46c7b11b and ecb13d3d, A2: the host case reads
    which copy ran, when, and the cache before the prunes and after them."""
    lines = ci_clean.CI_CLEAN_SCRIPT.splitlines()
    head = _commands(lines[: lines.index("{")])
    assert head[-5:] == [
        'report="${STATE_DIRECTORY:?ci-clean runs as ci-clean.service}/last-run"',
        "started=$(date +%s)",
        'digest=$(sha256sum "$0")',
        "build_cache_url='http://localhost/system/df?type=build-cache'",
        'build_cache_before=$(curl -sf --unix-socket /var/run/docker.sock "$build_cache_url")',
    ]
    tail = lines[lines.index('} | tee "$report.pending"') + 1 :]
    assert _commands(tail) == [
        'build_cache=$(curl -sf --unix-socket /var/run/docker.sock "$build_cache_url")',
        'printf \'%s\\n\' "started: $started" "script-sha256: ${digest%% *}" \\',
        '    | tee -a "$report.pending"',
        'printf \'%s\\n\' "build-cache-before: $build_cache_before" >> "$report.pending"',
        'printf \'%s\\n\' "build-cache: $build_cache" >> "$report.pending"',
        'mv -f "$report.pending" "$report"',
        'echo "ci-clean: reported to $report"',
    ]


def test_the_report_lies_in_the_state_directory_systemd_makes_for_the_service() -> None:
    # systemd makes StateDirectory= under /var/lib for a system service,
    # owned by its User=, and hands the path to it as STATE_DIRECTORY.
    assert ci_clean.CI_CLEAN_SERVICE.splitlines()[5:8] == [
        "User=gharunner",
        "StateDirectory=ci-clean",
        "ExecStart=/usr/local/bin/ci-clean",
    ]
    assert ci_clean.CI_CLEAN_REPORT == "/var/lib/ci-clean/last-run"


def test_the_build_cache_is_bounded_by_age_then_by_size() -> None:
    # Without these two lines no pass reached BuildKit's cache, and
    # lavender's daemon held 98.55 GB of it reclaimable on 2026-10-18 (MCPs
    # board task 46c7b11b). The age pass runs first, so the cap removes least
    # recently used records only from what a week kept.
    docker_lines = [
        line
        for line in ci_clean.CI_CLEAN_SCRIPT.splitlines()
        if line.startswith("docker builder prune")
    ]
    assert docker_lines == [
        "docker builder prune -f --filter until=168h",
        "docker builder prune -f --max-used-space 60gb",
    ]


def test_no_prune_takes_named_volumes_or_tagged_images() -> None:
    # --all widens volume prune to named volumes and image prune to every
    # unused tagged image, the pull cache each job would pay back.
    docker_lines = [
        line for line in ci_clean.CI_CLEAN_SCRIPT.splitlines() if line.startswith("docker ")
    ]
    assert len(docker_lines) == 5
    assert [line for line in docker_lines if "--all" in line or " -a" in line] == []


def test_the_timer_fires_daily_and_catches_a_missed_start() -> None:
    assert "OnCalendar=*-*-* 04:00" in ci_clean.CI_CLEAN_TIMER
    assert "Persistent=true" in ci_clean.CI_CLEAN_TIMER


def _generation(tools: pathlib.Path, *, days_ago: int) -> str:
    """Lay out a cache generation started some days ago, holding one MiB.

    Args:
        tools: The generations directory.
        days_ago: How long ago it started.

    Returns:
        Its name.
    """
    name = str(int(time.time()) - days_ago * DAY)
    artifacts = tools / name / "pypoetry" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "torch.whl").write_bytes(bytes(MIB))
    return name


def _virtualenv(venvs: pathlib.Path, name: str, sources: list[str]) -> pathlib.Path:
    """Lay out one poetry virtualenv recording these source paths.

    Args:
        venvs: The virtualenvs directory.
        name: Its folder.
        sources: Its ``.pth`` lines.

    Returns:
        Its directory.
    """
    site = venvs / name / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    (site / "weights.bin").write_bytes(bytes(2 * MIB))
    (site / "project.pth").write_text("".join(f"{line}\n" for line in sources), "utf-8")
    return venvs / name


def _run_caches(tmp_path: pathlib.Path, home: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Execute the Python half under the script's own shell options, for one home.

    Args:
        tmp_path: Where the script is written.
        home: The home it cleans, set in the script ahead of the half.

    Returns:
        The finished run.
    """
    script = tmp_path / "caches.sh"
    header = f"set -euo pipefail\nHOME={shlex.quote(str(home))}\n"
    script.write_text(header + ci_clean.CI_CLEAN_CACHES, encoding="utf-8")
    return subprocess.run(
        [host_bash(), str(script)], capture_output=True, text=True, check=False, timeout=120
    )


@pytest.mark.host_linux
def test_the_python_half_removes_only_orphans_and_generations_no_runner_reads(
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    venvs = home / ".cache" / "pypoetry" / "virtualenvs"
    checkout = tmp_path / "runner" / "_work" / "src"
    checkout.mkdir(parents=True)
    orphan = _virtualenv(venvs, "gone-py3.11", [str(tmp_path / "retired")])
    serving = _virtualenv(venvs, "alive-py3.11", [str(checkout)])
    unknown = _virtualenv(venvs, "unknown-py3.11", ["import _virtualenv"])
    tools = home / ".cache" / "fleet-tools"
    unread = _generation(tools, days_ago=30)
    marked = _generation(tools, days_ago=16)
    current = _generation(tools, days_ago=1)
    (tools / names.CURRENT_GENERATION_NAME).write_text(f"{current}\n", encoding="utf-8")
    idle_runner = home / ".cache" / "fleet-jobs" / "actions-runner-treebot-1"
    idle_runner.mkdir(parents=True)
    (idle_runner / names.GENERATION_MARKER_NAME).write_text(f"{marked}\n", encoding="utf-8")
    busy_runner = home / ".cache" / "fleet-jobs" / "actions-runner-api-1"
    busy_runner.mkdir(parents=True)
    (busy_runner / names.GENERATION_MARKER_NAME).write_text(f"{current}\n", encoding="utf-8")
    npm = tools / current / "npm" / "actions-runner-api-1" / "_cacache"
    npm.mkdir(parents=True)
    (npm / "index").write_bytes(bytes(3000))

    ran = _run_caches(tmp_path, home)

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.splitlines() == [
        "venv-sweep: removed 1 of 3 (2 MB)",
        f"cache-generations: {tools}: generation {current} current, removed 1 of 3 (1 MB)",
        f"cache-reader: actions-runner-api-1 read generation {current}",
        f"cache-reader: actions-runner-treebot-1 read generation {marked}",
        f"cache-tool: generation {current} holds npm (3000 bytes of files)",
        f"cache-tool: generation {current} holds npm/actions-runner-api-1 (3000 bytes of files)",
        f"cache-tool: generation {current} holds pypoetry ({MIB} bytes of files)",
    ]
    assert not orphan.exists()
    assert [serving.exists(), unknown.exists()] == [True, True]
    assert sorted(entry.name for entry in tools.iterdir()) == sorted(
        [marked, current, names.CURRENT_GENERATION_NAME]
    )
    assert not (tools / unread).exists()


@pytest.mark.host_linux
def test_a_host_no_job_has_run_on_says_it_has_no_generation(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    (home / ".cache" / "pypoetry" / "virtualenvs").mkdir(parents=True)

    ran = _run_caches(tmp_path, home)

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.splitlines() == [
        "venv-sweep: removed 0 of 0 (0 MB)",
        f"cache-generations: {home}/.cache/fleet-tools: none yet",
        "cache-reader: no generation yet",
    ]


@pytest.mark.host_linux
def test_the_python_half_stops_on_a_pth_it_cannot_read(tmp_path: pathlib.Path) -> None:
    # Read as recording nothing, an unreadable .pth kept its virtualenv with
    # no word said (MCPs board task 346abc45, A2).
    home = tmp_path / "home"
    venv = _virtualenv(home / ".cache" / "pypoetry" / "virtualenvs", "locked-py3.11", ["/gone"])
    pth = venv / "lib" / "python3.11" / "site-packages" / "project.pth"
    pth.chmod(0)
    try:
        ran = _run_caches(tmp_path, home)
    finally:
        pth.chmod(0o644)

    assert (ran.returncode, ran.stderr) == (1, f"venv-sweep: cannot read {pth}\n")
    assert venv.exists()
