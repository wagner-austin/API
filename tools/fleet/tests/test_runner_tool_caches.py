"""The runner job hook's cache generation (MCPs board task 4b405239).

The hook's lines are executed for real under this host's bash with a home,
a runner install and a ``GITHUB_ENV`` file laid out under ``tmp_path``, as
the Actions runner hands them to the hook (measured on lavender-wsl on
2026-10-18): the job is given the generation's paths in ``GITHUB_ENV``, the
generation is named in its runner install's own marker, and a
``RUNNER_TEMP`` outside an install fails the job by name.
"""

from __future__ import annotations

import pathlib
import shlex
import subprocess

from fleet.core import names, runner_tool_caches
from tests._host_bash import host_bash


def _hook(tmp_path: pathlib.Path, runner_temp: str) -> subprocess.CompletedProcess[str]:
    """Run the hook's cache lines once, as the runner runs the hook.

    Args:
        tmp_path: Holds the home and the hook's script.
        runner_temp: The ``RUNNER_TEMP`` the runner hands the hook.

    Returns:
        The finished run.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    header = (
        "set -euo pipefail\n"
        f"HOME={shlex.quote(home.as_posix())}\n"
        f"RUNNER_TEMP={shlex.quote(runner_temp)}\n"
        f"GITHUB_ENV={shlex.quote((tmp_path / 'set_env').as_posix())}\n"
    )
    script = tmp_path / "hook.sh"
    script.write_text(
        header + "".join(f"{line}\n" for line in runner_tool_caches.HOOK_LINES), encoding="utf-8"
    )
    return subprocess.run(
        [host_bash(), script.as_posix()], capture_output=True, text=True, check=False, timeout=60
    )


def test_a_job_is_handed_its_generation_and_its_install_marks_it(tmp_path: pathlib.Path) -> None:
    install = tmp_path / "home" / "actions-runner-api-1"

    ran = _hook(tmp_path, f"{install.as_posix()}/_work/_temp")

    assert ran.returncode == 0, ran.stderr
    home = (tmp_path / "home").as_posix()
    tools = tmp_path / "home" / ".cache" / "fleet-tools"
    generation = (tools / names.CURRENT_GENERATION_NAME).read_text(encoding="utf-8").strip()
    marker = tmp_path / "home" / ".cache" / "fleet-jobs" / "actions-runner-api-1"
    assert (marker / names.GENERATION_MARKER_NAME).read_text(encoding="utf-8") == f"{generation}\n"
    cache = f"{home}/.cache/fleet-tools/{generation}"
    assert (tmp_path / "set_env").read_text(encoding="utf-8").splitlines() == [
        f"POETRY_CACHE_DIR={cache}/pypoetry",
        f"POETRY_VIRTUALENVS_PATH={home}/.cache/pypoetry/virtualenvs",
        f"PIP_CACHE_DIR={cache}/pip",
        f"npm_config_cache={cache}/npm",
    ]
    assert ran.stdout == (
        f"fleet-runner-job-started: npm, pip and poetry read cache generation {generation}\n"
    )


def test_a_runner_temp_outside_a_runner_install_fails_the_job_by_name(
    tmp_path: pathlib.Path,
) -> None:
    ran = _hook(tmp_path, "/tmp/elsewhere")

    assert ran.returncode == 1
    assert ran.stderr == (
        "FLEET_RUNNER_CACHES_REFUSED: RUNNER_TEMP /tmp/elsewhere is no runner install's "
        "_work/_temp\n"
    )
    assert not (tmp_path / "set_env").exists()
