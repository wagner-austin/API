"""The action archive cache (MCPs board task ce2b92e0): both pieces run for real.

The fill script runs under a real bash over checkouts laid out as the runners
leave them, ``<workdir>/<repo>/<repo>/.github/...``, with ``git`` and ``curl``
defined as stand-ins that answer from a table and record what was fetched, so
nothing reaches GitHub. The drop-in lines run the same way with ``systemctl``
a recording stand-in and the three absolute roots they read, ``/etc/systemd/
system``, ``/proc`` and ``/sys/fs/cgroup``, rewritten to the test's own
directories: the write, the reload, the read of the running process's
environment and the busy check all run as rendered.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import time
from typing import Literal

import pytest

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import runner_action_cache
from tests._host_bash import host_bash
from tests._runner_fixtures import a_base, a_ci_slice

#: The unit the drop-in cases run.
UNIT = "actions.runner.wagner-austin-MCPs.lavender-wsl-4.service"

#: What actions/checkout's v7 resolved to in run 38709479412.
CHECKOUT_SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"

#: An annotated tag: the tag object, then the commit it peels to.
NODE_TAG = "a" * 40
NODE_SHA = "b" * 40

#: A branch ref.
CODEQL_SHA = "c" * 40

#: A ref pinned by its full sha.
CACHE_SHA = "d" * 40

#: ``git`` answers ``ls-remote`` with the repository's whole table, as the
#: script picks exact names out of it; ``curl`` writes the URL it was asked
#: for as the archive's bytes, or fails as curl -f does on an HTTP error.
FILL_STAND_INS = """
git() {
    [ "$1" = ls-remote ] || return 9
    local repo=${2#https://github.com/}
    cat "$TABLE/refs/${repo/\\//_}" 2>/dev/null || return 128
}
curl() {
    local out url
    while [ $# -gt 0 ]; do
        case $1 in
            -o) out=$2; shift 2 ;;
            -*) shift ;;
            *) url=$1; shift ;;
        esac
    done
    echo "$url" >> "$TABLE/fetched"
    if [ -e "$TABLE/refuse" ]; then
        printf 'partial' > "$out"
        return 22
    fi
    printf '%s' "$url" > "$out"
}
"""

#: The workflow a checkout holds: a lightweight tag, a quoted annotated tag
#: in a list item, a subpath action on a branch, a pinned sha, a local
#: composite action, a docker image and a comment, the last three never
#: fetched.
WORKFLOW = f"""name: Check
jobs:
  test:
    steps:
      - uses: actions/checkout@v7
      - uses: "actions/setup-node@v7"
      - uses: github/codeql-action/init@v3  # a subpath, on a branch
      - uses: actions/cache@{CACHE_SHA}
      - uses: ./.github/actions/setup-workspace
      - uses: docker://alpine:3
      # uses: actions/never@v1 is a comment and is not read
"""

#: A composite action naming one action the workflow also names.
COMPOSITE = """runs:
  using: composite
  steps:
    - uses: actions/checkout@v7
"""


def _install(
    service: str,
    side: Literal["wsl", "windows"] = "wsl",
    workdir: str = "/home/gharunner/r/_work",
) -> RunnerInstall:
    """One install.

    Args:
        service: The unit or service name.
        side: ``wsl`` or ``windows``.
        workdir: Its work tree.

    Returns:
        The install.
    """
    return RunnerInstall(
        repo="wagner-austin/MCPs",
        runner_name="lavender-wsl-4",
        side=side,
        service=service,
        workdir=workdir,
        labels=["lavender-wsl"],
        python_toolcache=[],
    )


def _host(installs: list[RunnerInstall]) -> HostRunnerSpec:
    """lavender with the given installs.

    Args:
        installs: Its installs.

    Returns:
        The roster entry.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task="wsl-keepalive",
        wslconfig_min_memory_gb=26,
        scratch_dir="C:/fleet/stage",
        gpu_required=True,
        systemd_timers=["ci-clean.timer"],
        job_timeout_minutes=360,
        installs=installs,
        assets=[],
        base=a_base(),
        ci_slice=a_ci_slice(),
    )


class Fill:
    """A laid-out host: a table of refs, checkouts under workdirs, a cache."""

    def __init__(self, root: pathlib.Path) -> None:
        """Lay out the table and an empty cache.

        Args:
            root: The test's directory.
        """
        self.root = root
        self.table = root / "table"
        (self.table / "refs").mkdir(parents=True)
        self.cache = root / "cache"
        self.refs("actions/checkout", f"{CHECKOUT_SHA}\trefs/tags/v7\n")
        self.refs(
            "actions/setup-node", f"{NODE_TAG}\trefs/tags/v7\n{NODE_SHA}\trefs/tags/v7^{{}}\n"
        )
        self.refs("github/codeql-action", f"{CODEQL_SHA}\trefs/heads/v3\n")
        script = (
            f"TABLE='{self.table.as_posix()}'\n" + FILL_STAND_INS + runner_action_cache.FILL_SCRIPT
        )
        (root / "fill.sh").write_bytes(script.encode())

    def refs(self, repo: str, listing: str) -> None:
        """What ``git ls-remote`` answers for one repository.

        Args:
            repo: ``owner/repo``.
            listing: The lines, ``<sha>\\t<ref>``.
        """
        (self.table / "refs" / repo.replace("/", "_")).write_text(listing, encoding="utf-8")

    def checkout(self, workdir: str, repo: str, workflow: str, composite: str | None) -> None:
        """A checkout the way a runner leaves it under its work tree.

        Args:
            workdir: The work tree's name under the test's directory.
            repo: The checked-out repository's name.
            workflow: Its one workflow.
            composite: Its composite setup action, or None for none.
        """
        github = self.root / workdir / repo / repo / ".github"
        (github / "workflows").mkdir(parents=True)
        (github / "workflows" / "check.yml").write_text(workflow, encoding="utf-8")
        if composite is not None:
            (github / "actions" / "setup-workspace").mkdir(parents=True)
            (github / "actions" / "setup-workspace" / "action.yml").write_text(
                composite, encoding="utf-8"
            )

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run the fill.

        Args:
            arguments: Its arguments.

        Returns:
            The finished process.
        """
        return subprocess.run(
            [host_bash(), "fill.sh", *arguments],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

    def fetched(self) -> list[str]:
        """Every URL curl was asked for, in order.

        Returns:
            The URLs, none when curl never ran.
        """
        log = self.table / "fetched"
        return log.read_text(encoding="utf-8").split() if log.exists() else []

    def archives(self) -> list[str]:
        """Every file in the cache, relative to it.

        Returns:
            The sorted ``<owner>_<repo>/<file>`` paths.
        """
        return sorted(
            p.relative_to(self.cache).as_posix() for p in self.cache.rglob("*") if p.is_file()
        )


def _summary(
    examined: int,
    workdirs: int,
    cached: int,
    held: int,
    failed: int,
    pruned: int,
    archive_format: str = "tar.gz",
) -> str:
    """One pass's last line, into the cases' ``cache``.

    Args:
        examined: References read.
        workdirs: Workdirs given.
        cached: Archives fetched.
        held: Archives already there.
        failed: References that resolved to nothing or failed to fetch.
        pruned: Archives removed.
        archive_format: The pass's format.

    Returns:
        The line, with its newline.
    """
    return (
        f"fleet-action-cache-fill: {archive_format} into cache: {examined} action reference(s) "
        f"in the checkouts of {workdirs} workdir(s): {cached} cached now, {held} already held, "
        f"{failed} failed; {pruned} unnamed for {runner_action_cache.PRUNE_DAYS} days removed\n"
    )


class TestTheFillRunsForReal:
    """One archive per action version, fetched once, kept, and pruned only when long unnamed."""

    def test_every_action_is_fetched_once_and_a_second_pass_fetches_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        host.checkout("one", "MCPs", WORKFLOW, COMPOSITE)
        host.checkout("two", "API", "steps:\n  - uses: actions/checkout@v7\n", None)
        first = host.run("tar.gz", "cache", "one", "two")
        assert first.returncode == 0, first.stderr
        codeload = "https://codeload.github.com"
        assert host.fetched() == [
            f"{codeload}/actions/cache/tar.gz/{CACHE_SHA}",
            f"{codeload}/actions/checkout/tar.gz/{CHECKOUT_SHA}",
            f"{codeload}/actions/setup-node/tar.gz/{NODE_SHA}",
            f"{codeload}/github/codeql-action/tar.gz/{CODEQL_SHA}",
        ]
        assert host.archives() == [
            f"actions_cache/{CACHE_SHA}.tar.gz",
            f"actions_checkout/{CHECKOUT_SHA}.tar.gz",
            f"actions_setup-node/{NODE_SHA}.tar.gz",
            f"github_codeql-action/{CODEQL_SHA}.tar.gz",
        ]
        archive = host.cache / "actions_checkout" / f"{CHECKOUT_SHA}.tar.gz"
        assert (
            archive.read_text(encoding="utf-8")
            == f"{codeload}/actions/checkout/tar.gz/{CHECKOUT_SHA}"
        )
        assert (
            first.stdout.splitlines()[1]
            == f"fleet-action-cache-fill: cached actions/checkout@v7 at {CHECKOUT_SHA}"
        )
        assert first.stdout.endswith(_summary(4, 2, 4, 0, 0, 0))
        second = host.run("tar.gz", "cache", "one", "two")
        assert second.returncode == 0, second.stderr
        assert len(host.fetched()) == 4
        assert second.stdout == _summary(4, 2, 0, 4, 0, 0)

    def test_the_windows_side_gets_zip_archives_from_the_zip_path(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        host.checkout("win", "MCPs", "steps:\n  - uses: actions/checkout@v7\n", None)
        ran = host.run("zip", "cache", "win")
        assert ran.returncode == 0, ran.stderr
        assert host.fetched() == [
            f"https://codeload.github.com/actions/checkout/zip/{CHECKOUT_SHA}"
        ]
        assert host.archives() == [f"actions_checkout/{CHECKOUT_SHA}.zip"]

    def test_a_ref_naming_nothing_fails_the_pass_after_the_others_are_cached(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        workflow = (
            "steps:\n"
            "  - uses: actions/checkout@v7\n"
            "  - uses: actions/setup-node@v99\n"
            "  - uses: nobody/gone@v1\n"
        )
        host.checkout("one", "MCPs", workflow, None)
        ran = host.run("tar.gz", "cache", "one")
        assert ran.returncode == 1
        assert ran.stdout.splitlines()[1:] == [
            "fleet-action-cache-fill: actions/setup-node@v99 names no tag, branch or commit "
            "at https://github.com/actions/setup-node",
            "fleet-action-cache-fill: nobody/gone@v1 names no tag, branch or commit "
            "at https://github.com/nobody/gone",
            _summary(3, 1, 1, 0, 2, 0).rstrip("\n"),
        ]
        assert host.archives() == [f"actions_checkout/{CHECKOUT_SHA}.tar.gz"]

    def test_a_refused_download_leaves_no_partial_archive_and_fails_the_pass(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        host.checkout("one", "MCPs", "steps:\n  - uses: actions/checkout@v7\n", None)
        (host.table / "refuse").write_text("", encoding="utf-8")
        ran = host.run("tar.gz", "cache", "one")
        assert ran.returncode == 1
        assert ran.stdout == (
            f"fleet-action-cache-fill: could not fetch actions/checkout@v7 at {CHECKOUT_SHA} "
            "from https://codeload.github.com\n" + _summary(1, 1, 0, 0, 1, 0)
        )
        assert host.archives() == []

    def test_only_an_archive_unnamed_past_the_prune_age_is_removed(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        host.checkout("one", "MCPs", "steps:\n  - uses: actions/checkout@v7\n", None)
        days = 86400
        old = time.time() - (runner_action_cache.PRUNE_DAYS + 10) * days
        named = host.cache / "actions_checkout" / f"{CHECKOUT_SHA}.tar.gz"
        stale = host.cache / "actions_checkout" / f"{'e' * 40}.tar.gz"
        recent = host.cache / "actions_setup-node" / f"{'f' * 40}.tar.gz"
        other_format = host.cache / "actions_setup-node" / f"{'f' * 40}.zip"
        for archive, when in (
            (named, old),
            (stale, old),
            (recent, time.time() - 5 * days),
            (other_format, old),
        ):
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_text("held", encoding="utf-8")
            os.utime(archive, (when, when))
        ran = host.run("tar.gz", "cache", "one")
        assert ran.returncode == 0, ran.stderr
        assert host.fetched() == []
        assert ran.stdout == _summary(1, 1, 0, 1, 0, 1)
        assert host.archives() == [
            f"actions_checkout/{CHECKOUT_SHA}.tar.gz",
            f"actions_setup-node/{'f' * 40}.tar.gz",
            f"actions_setup-node/{'f' * 40}.zip",
        ]
        # Named this pass, so its age starts again.
        assert named.stat().st_mtime > time.time() - days

    def test_a_host_with_no_checkouts_examines_nothing_and_removes_nothing_recent(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        (tmp_path / "empty").mkdir()
        held = host.cache / "actions_checkout" / f"{CHECKOUT_SHA}.tar.gz"
        held.parent.mkdir(parents=True)
        held.write_text("held", encoding="utf-8")
        ran = host.run("tar.gz", "cache", "empty")
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == _summary(0, 1, 0, 0, 0, 0)
        assert host.archives() == [f"actions_checkout/{CHECKOUT_SHA}.tar.gz"]

    def test_a_failed_pass_does_not_stop_the_next_and_fails_the_fill(
        self, tmp_path: pathlib.Path
    ) -> None:
        # lavender's first run, 2026-10-15T12:41Z: codeload refused one
        # archive, and the Windows pass, a second ExecStart, never ran.
        host = Fill(tmp_path)
        host.checkout("one", "MCPs", "steps:\n  - uses: nobody/gone@v1\n", None)
        host.checkout("win", "MCPs", "steps:\n  - uses: actions/checkout@v7\n", None)
        ran = host.run("tar.gz", "cache", "one", "--", "zip", "cache", "win")
        assert ran.returncode == 1
        assert ran.stdout.splitlines()[1] == _summary(1, 1, 0, 0, 1, 0).rstrip("\n")
        assert ran.stdout.endswith(_summary(1, 1, 1, 0, 0, 0, "zip"))
        assert host.archives() == [f"actions_checkout/{CHECKOUT_SHA}.zip"]

    @pytest.mark.parametrize(
        "arguments",
        [
            ("tar.gz", "cache"),
            ("tar", "cache", "one"),
            ("tar.gz", "cache", "one", "--", "zip", "cache"),
        ],
    )
    def test_a_short_or_unknown_pass_is_refused_with_its_usage_before_any_runs(
        self, tmp_path: pathlib.Path, arguments: tuple[str, ...]
    ) -> None:
        host = Fill(tmp_path)
        host.checkout("one", "MCPs", "steps:\n  - uses: actions/checkout@v7\n", None)
        ran = host.run(*arguments)
        assert ran.returncode == 2
        assert ran.stderr == (
            "fleet-action-cache-fill: usage: fleet-action-cache-fill tar.gz|zip CACHE WORKDIR... "
            "[-- tar.gz|zip CACHE WORKDIR...]...\n"
        )
        assert host.fetched() == []


class TestTheFillUnits:
    """The service passes each side's workdirs to its own cache."""

    def test_a_host_with_both_sides_runs_one_pass_per_side(self) -> None:
        service = runner_action_cache.render_fill_service(
            _host(
                [
                    _install(UNIT),
                    _install(
                        "actions.runner.wagner-austin-API.lavender-wsl.service",
                        workdir="/home/gharunner/a/_work",
                    ),
                    _install(
                        "actions.runner.wagner-austin-MCPs.lavender",
                        "windows",
                        "C:/actions-runner/_work",
                    ),
                ]
            )
        )
        assert [line for line in service.splitlines() if line.startswith("ExecStart=")] == [
            "ExecStart=/usr/local/sbin/fleet-action-cache-fill tar.gz /var/cache/fleet-actions "
            "/home/gharunner/r/_work /home/gharunner/a/_work -- "
            "zip /mnt/c/fleet/action-cache /mnt/c/actions-runner/_work",
        ]

    def test_a_host_with_one_side_runs_one_pass(self) -> None:
        service = runner_action_cache.render_fill_service(_host([_install(UNIT)]))
        assert [line for line in service.splitlines() if line.startswith("ExecStart=")] == [
            "ExecStart=/usr/local/sbin/fleet-action-cache-fill tar.gz /var/cache/fleet-actions "
            "/home/gharunner/r/_work"
        ]

    def test_a_windows_workdir_without_a_drive_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a forward-slashed drive path"):
            runner_action_cache.render_fill_service(
                _host([_install(UNIT, "windows", "actions-runner/_work")])
            )

    def test_the_lines_install_the_script_both_units_and_both_caches_then_start_a_pass(
        self,
    ) -> None:
        lines = runner_action_cache.render_fill_lines(_host([_install(UNIT)]))
        assert lines[:3] == [
            "cat > /usr/local/sbin/fleet-action-cache-fill <<'FILL_EOF'",
            runner_action_cache.FILL_SCRIPT.rstrip("\n"),
            "FILL_EOF",
        ]
        assert lines[-4:] == [
            "install -d -m 0755 /var/cache/fleet-actions /mnt/c/fleet/action-cache",
            "systemctl daemon-reload",
            "systemctl enable --now fleet-action-cache-fill.timer",
            "systemctl start --no-block fleet-action-cache-fill.service",
        ]
        assert "OnUnitActiveSec=15min" in runner_action_cache.FILL_TIMER


class Unit:
    """A runner unit laid out as files: its drop-ins, its main process, its cgroup."""

    def __init__(self, root: pathlib.Path) -> None:
        """Lay out a running unit whose main process lacks the variable.

        Args:
            root: The test's directory.
        """
        self.root = root
        (root / "proc" / "100").mkdir(parents=True)
        (root / "proc" / "100" / "environ").write_bytes(b"HOME=/home/gharunner\0LANG=C.UTF-8\0")
        (root / "proc" / "100" / "cmdline").write_bytes(b"/bin/bash\0runsvc.sh\0")
        self.cgroup = root / "cgroup" / "system.slice" / UNIT
        self.cgroup.mkdir(parents=True)
        (self.cgroup / "cgroup.procs").write_text("100\n", encoding="utf-8")
        body = "\n".join(runner_action_cache.render_cache_environment_lines(_install(UNIT)))
        for real, laid_out in (
            ("/etc/systemd/system", "units"),
            ("/proc/", "proc/"),
            ("/sys/fs/cgroup", "cgroup"),
        ):
            body = body.replace(real, laid_out)
        # restart gives the unit's main process the drop-in's environment, as
        # systemd does when it starts the unit again.
        stand_ins = (
            "systemctl() {\n"
            '    echo "$*" >> calls.log\n'
            '    case "$1 $3" in\n'
            '        "show MainPID") echo 100 ;;\n'
            f'        "show ControlGroup") echo /system.slice/{UNIT} ;;\n'
            "    esac\n"
            '    if [ "$1" = restart ]; then\n'
            "        printf 'HOME=/home/gharunner\\0"
            f"{runner_action_cache.CACHE_VARIABLE}={runner_action_cache.WSL_CACHE_DIR}\\0' "
            "> proc/100/environ\n"
            "    fi\n"
            "}\n"
        )
        (root / "cache.sh").write_bytes((stand_ins + body + "\n").encode())

    def busy(self) -> None:
        """Put a job's Worker in the unit's cgroup."""
        (self.root / "proc" / "200").mkdir()
        (self.root / "proc" / "200" / "cmdline").write_bytes(
            b"/home/gharunner/bin/Runner.Worker\0spawnclient\0"
        )
        (self.cgroup / "cgroup.procs").write_text("100\n200\n", encoding="utf-8")

    def run(self) -> subprocess.CompletedProcess[str]:
        """Run the lines once.

        Returns:
            The finished process.
        """
        return subprocess.run(
            [host_bash(), "cache.sh"], cwd=self.root, capture_output=True, text=True, check=False
        )

    def calls(self) -> list[str]:
        """Every systemctl call, in order.

        Returns:
            The calls' arguments.
        """
        return (self.root / "calls.log").read_text(encoding="utf-8").splitlines()


class TestTheDropInRunsForReal:
    """The variable lands once, and reaches the unit only by a restart of an idle unit."""

    def test_an_idle_unit_is_given_the_variable_and_restarted_once(
        self, tmp_path: pathlib.Path
    ) -> None:
        unit = Unit(tmp_path)
        first = unit.run()
        assert first.returncode == 0, first.stderr
        assert first.stdout == f"action cache set for {UNIT}: restarted while idle\n"
        drop_in = tmp_path / "units" / f"{UNIT}.d" / runner_action_cache.CACHE_DROP_IN_NAME
        assert drop_in.read_bytes() == runner_action_cache.CACHE_DROP_IN.encode()
        second = unit.run()
        assert second.returncode == 0, second.stderr
        assert second.stdout == ""
        assert unit.calls() == [
            "daemon-reload",
            f"show -p MainPID --value {UNIT}",
            f"show -p ControlGroup --value {UNIT}",
            f"restart {UNIT}",
            f"show -p MainPID --value {UNIT}",
        ]

    def test_a_unit_running_a_job_is_left_for_the_next_run(self, tmp_path: pathlib.Path) -> None:
        unit = Unit(tmp_path)
        unit.busy()
        ran = unit.run()
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == (
            f"action cache set for {UNIT} at its next restart: it is running a job; "
            "run provision again to restart it\n"
        )
        assert f"restart {UNIT}" not in unit.calls()

    def test_an_unscriptable_unit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be embedded in a rendered script verbatim"):
            runner_action_cache.render_cache_environment_lines(_install("bad'unit"))
