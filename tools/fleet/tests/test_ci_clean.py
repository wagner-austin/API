"""ci-clean, the CI runner hosts' daily hygiene (:mod:`fleet.core.ci_clean`).

Its Python half is executed for real against a home this suite lays out,
with atimes and mtimes set where each rule draws its line, because the rules
are find predicates whose meaning only GNU find on Linux settles; a string
assertion would pass a script that deletes everything or nothing.
"""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import time
from typing import Final

import pytest

from fleet.core import ci_clean, venv_sweep
from tests._host_bash import host_bash

#: One day in seconds.
DAY: Final = 86400

#: What each laid-out cache file holds: one MiB, which ext4 counts as 1024 KB.
MIB: Final = 1024 * 1024


def test_no_pass_waits_for_the_runners_to_be_idle() -> None:
    # The gate this replaced skipped the whole Python half while any
    # Runner.Worker ran, and lavender always runs one (MCPs board task
    # ecb13d3d): 30 days of 'skipping this pass' and no reclaim.
    assert "Runner.Worker" not in ci_clean.CI_CLEAN_SCRIPT
    assert "skipping this pass" not in ci_clean.CI_CLEAN_SCRIPT


def test_the_orphan_rule_is_the_node_sweeps_own_lines() -> None:
    # One rule for both caches (MCPs board task 346abc45): the lines are
    # rendered from venv_sweep, so changing them there changes both.
    rule = "\n".join(venv_sweep.orphan_sweep_lines('"$POETRY/virtualenvs"'))
    assert rule in ci_clean.CI_CLEAN_SCRIPT
    assert "\n".join(venv_sweep.orphan_sweep_lines("'/n'")[1:]) in venv_sweep.linux_script("/n")
    assert "unknown provenance: keep" in ci_clean.CI_CLEAN_SCRIPT


def test_the_python_half_follows_the_docker_prunes() -> None:
    assert ci_clean.CI_CLEAN_SCRIPT.endswith(ci_clean.CI_CLEAN_CACHES)
    assert ci_clean.CI_CLEAN_SCRIPT.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail\n" in ci_clean.CI_CLEAN_SCRIPT


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


def _file(path: pathlib.Path, *, read_days_ago: int, changed_days_ago: int) -> pathlib.Path:
    """Write one MiB at a path, last read and last changed so many days ago.

    Args:
        path: The file.
        read_days_ago: Its atime, in days before now.
        changed_days_ago: Its mtime, in days before now.

    Returns:
        The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x01" * MIB)
    now = time.time()
    os.utime(path, (now - read_days_ago * DAY, now - changed_days_ago * DAY))
    return path


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
def test_the_python_half_removes_only_orphans_unread_files_and_stale_fragments(
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    poetry = home / ".cache" / "pypoetry"
    checkout = tmp_path / "runner" / "_work" / "src"
    checkout.mkdir(parents=True)
    orphan = _virtualenv(poetry / "virtualenvs", "gone-py3.11", [str(tmp_path / "retired")])
    serving = _virtualenv(poetry / "virtualenvs", "alive-py3.11", [str(checkout)])
    unknown = _virtualenv(poetry / "virtualenvs", "unknown-py3.11", ["import _virtualenv"])
    pypi = poetry / "cache" / "repositories" / "PyPI"
    pip = home / ".cache" / "pip" / "http"
    unread = [
        _file(poetry / "artifacts" / "a" / "old.whl", read_days_ago=20, changed_days_ago=60),
        _file(pypi / "old", read_days_ago=20, changed_days_ago=20),
        _file(pip / "old", read_days_ago=20, changed_days_ago=20),
    ]
    read = [
        _file(poetry / "artifacts" / "b" / "used.whl", read_days_ago=1, changed_days_ago=60),
        _file(pypi / "used", read_days_ago=0, changed_days_ago=30),
        _file(pip / "used", read_days_ago=2, changed_days_ago=2),
    ]
    stale_fragment = _file(poetry / "artifacts" / "tmpab12", read_days_ago=0, changed_days_ago=2)
    live_fragment = _file(poetry / "artifacts" / "tmpcd34", read_days_ago=0, changed_days_ago=0)
    empty_old = poetry / "cache" / "empty-old"
    empty_old.mkdir()
    os.utime(empty_old, (time.time() - 20 * DAY, time.time() - 20 * DAY))
    empty_new = poetry / "cache" / "empty-new"
    empty_new.mkdir()

    ran = _run_caches(tmp_path, home)

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.splitlines() == [
        "venv-sweep: removed 1 of 3 (2 MB)",
        f"ci-clean: {poetry}/artifacts: 1024KB unread for 14 days",
        f"ci-clean: {poetry}/cache: 1024KB unread for 14 days",
        f"ci-clean: {home}/.cache/pip: 1024KB unread for 14 days",
        "ci-clean: reclaimed 6MB this pass.",
    ]
    assert not orphan.exists()
    assert [serving.exists(), unknown.exists()] == [True, True]
    assert [path.exists() for path in unread] == [False, False, False]
    assert [path.exists() for path in read] == [True, True, True]
    assert [stale_fragment.exists(), live_fragment.exists()] == [False, True]
    assert [empty_old.exists(), empty_new.exists()] == [False, True]


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
