"""The provision's retirement of runner units the roster no longer declares (board task 49d1a916).

The lines run here under a real bash. ``systemctl``, ``sleep`` and ``rm``
are stand-ins that record what they were asked (tests/_stand_ins.py), each
retiring unit's ``svc.sh`` is a real script in the case's directory that
records its verb, and the cgroup root the lines read job scopes under is the
case's own directory.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.runner_slice import CI_SLICE_NAME
from fleet.contracts.runners import HostRunnerSpec
from fleet.core.runner_proc_bash import JOB_SCOPE_PREFIX
from fleet.core.runner_retire_render import CGROUP_ROOT, RETIRE_POLL_SECONDS, render_retire_lines
from tests._host_bash import bash_path
from tests._runner_fixtures import a_host, an_install
from tests._stand_ins import run_with_stand_ins

#: The unit the host keeps.
KEPT = "actions.runner.wagner-austin-MCPs.lavender.service"
#: A unit the roster dropped, as lavender's MCPs lavender-3 was.
DROPPED = "actions.runner.wagner-austin-MCPs.lavender-3.service"
#: The listing the lines ask systemd for.
LISTING = "systemctl list-units actions.runner.* --all --no-legend --plain"


def _host() -> HostRunnerSpec:
    """Lavender keeping its first MCPs runner alone."""
    return a_host([an_install("wagner-austin/MCPs", "lavender")])


def _root(tmp_path: pathlib.Path, unit: str) -> pathlib.Path:
    """A retiring unit's install directory, its ``svc.sh`` recording each verb.

    Args:
        tmp_path: The case's directory, holding ``asked``.
        unit: The unit, which names the directory.

    Returns:
        The directory.
    """
    root = tmp_path / unit.removesuffix(".service")
    root.mkdir()
    svc = root / "svc.sh"
    record = f'echo "svc.sh $*" >> "{bash_path(tmp_path)}/asked"'
    svc.write_bytes(f"#!/usr/bin/env bash\n{record}\n".encode())
    svc.chmod(0o755)
    return root


def _scope(tmp_path: pathlib.Path, unit: str, pid: int) -> pathlib.Path:
    """A live job scope of ``unit`` under the case's cgroup root.

    Args:
        tmp_path: The case's directory.
        unit: The unit whose job it is.
        pid: The job's Worker pid.

    Returns:
        The scope's directory, made.
    """
    name = f"{JOB_SCOPE_PREFIX}{unit.removesuffix('.service')}-{pid}.scope"
    scope = tmp_path / "cgroup" / CI_SLICE_NAME / name
    scope.mkdir(parents=True)
    return scope


def _retire(
    tmp_path: pathlib.Path,
    spec: HostRunnerSpec,
    loaded: list[str],
    *,
    sleep: str = "exit 0",
) -> subprocess.CompletedProcess[str]:
    """Run the rendered retirement over a host that has ``loaded`` units.

    Args:
        tmp_path: The case's directory.
        spec: The host's roster entry.
        loaded: The runner units systemd lists, each working in its own
            directory under ``tmp_path``.
        sleep: What the ``sleep`` stand-in does after recording.

    Returns:
        The finished run.
    """
    asked = f"{bash_path(tmp_path)}/asked"
    listing = "".join(
        f"echo '{unit} loaded active running GitHub Actions Runner'\n" for unit in loaded
    )
    systemctl = (
        f'echo "systemctl $*" >> "{asked}"\n'
        'case "$1" in\n'
        f"    list-units) {listing or ':'} ;;\n"
        f'    show) echo "{bash_path(tmp_path)}/${{5%.service}}" ;;\n'
        "esac"
    )
    lines = [
        line.replace(CGROUP_ROOT, f"{bash_path(tmp_path)}/cgroup")
        for line in render_retire_lines(spec)
    ]
    return run_with_stand_ins(
        tmp_path,
        lines,
        {
            "systemctl": systemctl,
            "sleep": f'echo "sleep $*" >> "{asked}"\n{sleep}',
            "rm": f'echo "rm $*" >> "{asked}"',
        },
        options=("-e", "-u", "-o", "pipefail"),
    )


def _asked(tmp_path: pathlib.Path) -> list[str]:
    """Every command line the stand-ins and ``svc.sh`` recorded, in order."""
    asked = tmp_path / "asked"
    return asked.read_text(encoding="utf-8").splitlines() if asked.exists() else []


def _retired(tmp_path: pathlib.Path, unit: str) -> list[str]:
    """What retiring an idle ``unit`` asks, after the listing."""
    root = f"{bash_path(tmp_path)}/{unit.removesuffix('.service')}"
    return [
        f"systemctl show -p WorkingDirectory --value {unit}",
        "svc.sh stop",
        "svc.sh uninstall",
        f"rm -rf /etc/systemd/system/{unit}.d {root}",
        "systemctl daemon-reload",
    ]


def test_an_undeclared_idle_unit_is_stopped_uninstalled_and_removed(tmp_path: pathlib.Path) -> None:
    root = _root(tmp_path, DROPPED)

    ran = _retire(tmp_path, _host(), [KEPT, DROPPED])

    assert ran.returncode == 0, ran.stderr
    assert _asked(tmp_path) == [LISTING, *_retired(tmp_path, DROPPED)]
    assert ran.stdout == (
        f"runner retired: {DROPPED} and its install {bash_path(root)}; "
        "fleet-runners --prune yes deletes its registration\n"
    )


def test_a_host_running_only_declared_units_is_left_alone(tmp_path: pathlib.Path) -> None:
    ran = _retire(tmp_path, _host(), [KEPT])

    assert ran.returncode == 0, ran.stderr
    assert (_asked(tmp_path), ran.stdout) == ([LISTING], "")


def test_a_unit_running_a_job_is_retired_once_its_job_ends(tmp_path: pathlib.Path) -> None:
    _root(tmp_path, DROPPED)
    scope = _scope(tmp_path, DROPPED, 4242)

    ran = _retire(tmp_path, _host(), [DROPPED], sleep=f"rmdir '{bash_path(scope)}'")

    assert ran.returncode == 0, ran.stderr
    assert _asked(tmp_path) == [
        LISTING,
        f"sleep {RETIRE_POLL_SECONDS}",
        *_retired(tmp_path, DROPPED),
    ]


def test_another_runner_s_job_never_holds_a_unit_whose_name_it_extends(
    tmp_path: pathlib.Path,
) -> None:
    """``...lavender-2-4242.scope`` begins as ``...lavender``'s scopes would,
    and is lavender-2's."""
    retiring = "actions.runner.wagner-austin-API.lavender.service"
    _root(tmp_path, retiring)
    _scope(tmp_path, "actions.runner.wagner-austin-API.lavender-2.service", 4242)

    ran = _retire(tmp_path, _host(), [retiring])

    assert ran.returncode == 0, ran.stderr
    assert _asked(tmp_path) == [LISTING, *_retired(tmp_path, retiring)]


def test_a_job_outlasting_the_host_s_timeout_is_refused_and_its_unit_kept(
    tmp_path: pathlib.Path,
) -> None:
    spec = _host()
    spec["job_timeout_minutes"] = 1
    _root(tmp_path, DROPPED)
    _scope(tmp_path, DROPPED, 4242)

    ran = _retire(tmp_path, spec, [DROPPED])

    assert ran.returncode == 1
    assert ran.stderr == f"runner retire refused: {DROPPED} still runs a job after 60 s\n"
    assert _asked(tmp_path) == [LISTING, *[f"sleep {RETIRE_POLL_SECONDS}"] * 6]


def test_a_unit_whose_working_directory_holds_no_svc_sh_is_refused(tmp_path: pathlib.Path) -> None:
    ran = _retire(tmp_path, _host(), [DROPPED])

    root = f"{bash_path(tmp_path)}/{DROPPED.removesuffix('.service')}"
    assert ran.returncode == 1
    assert ran.stderr == (
        f"runner retire refused: {DROPPED} has no svc.sh in its working directory '{root}'\n"
    )
    assert _asked(tmp_path) == [LISTING, f"systemctl show -p WorkingDirectory --value {DROPPED}"]


def test_an_unscriptable_declared_unit_is_refused() -> None:
    install = an_install("wagner-austin/MCPs", "lavender")
    install["service"] = "bad'unit.service"
    with pytest.raises(ValueError, match="service"):
        render_retire_lines(a_host([install]))
