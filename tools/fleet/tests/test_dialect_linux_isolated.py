"""A docker project's build runs as execdocker, never as the runner.

MCPs board task a8ee9b21. On diphtheria the runner's user is in the docker
group, so a docker project built as that user would reach the production
stack's daemon. :mod:`fleet.core.linux_isolated_build` runs such a build as
execdocker against its rootless daemon instead. The text cases pin the
properties that make that an isolation; the host case runs inside
tools/fleet-execution-linux's own isolated build, which the docker tag
routes through that script, and proves from there that it runs as
execdocker, sees a rootless daemon, is refused by the stack's socket and
cannot escalate. Two more host cases read, from the same build, that the
venv the lane made for it runs the fleet's pinned Python (MCPs board task
53b4725d) and that the interpreter's Python.h is on the node (MCPs board
task bcb7c6ef).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import sysconfig

import pytest

from fleet.contracts.project import ProjectConfig
from fleet.contracts.source import InstallStep
from fleet.contracts.tags import NodeTag
from fleet.contracts.toolchain import REQUIRED_PYTHON
from fleet.core import dialect, names, recipe, tool_caches
from fleet.core.ci_load import CI_CPU_STAT, sh_ci_load_end_lines
from fleet.core.dialect import PROLOGUE
from fleet.core.linux_isolated_build import EXEC_HOME_LINE, EXEC_LANE_LOCK, EXEC_USER
from fleet.core.node_programs import EXEC_PATH
from fleet.core.phase_markers import SH_UTC_STAMP
from fleet.core.run_summary import sh_run_summary_line
from tests.conftest import DEMO_PROJECT

TARGET = "/home/corvis/fleet/stage/slime-execution-1790600000"


NPM_CI = InstallStep(phase="install", argv=("npm", "ci"))


def _isolated(
    *, target: str = TARGET, path: str = DEMO_PROJECT, install: tuple[InstallStep, ...] = ()
) -> str:
    """Render the isolated build for one run.

    Args:
        target: The dispatch directory.
        path: The recipe's directory inside the export.
        install: The install steps.

    Returns:
        The script's text.
    """
    return dialect.build_script(
        target=target,
        path=path,
        workers=3,
        install=install,
        cache_root="/s/cache",
        isolated_docker=True,
        agent="opus-demo-0929",
        loopback_unit=None,
    )


def _plan(required_tags: tuple[NodeTag, ...]) -> ProjectConfig:
    """A project declaration requiring these tags.

    Args:
        required_tags: What the suite requires of a node.

    Returns:
        The project.
    """
    return ProjectConfig(
        worker_ram_gb=1.0,
        minimum_workers=1,
        expected_minutes=5,
        exclusive_resources=(),
        external_paths=(),
        required_tags=required_tags,
        source=None,
        node=None,
        loopback_only=False,
        worker_cores=1,
        worker_cores_measurement=None,
    )


class TestIsolatedText:
    def test_it_starts_with_the_prologue_and_copies_the_tree_into_execdocker_s_home(
        self,
    ) -> None:
        lines = _isolated().splitlines()

        assert _isolated().startswith(PROLOGUE)
        copy = lines.index(f"sudo -n cp -a '{TARGET}/.' \"$exec_root/\"")
        uid = lines.index(f'exec_uid="$(id -u {EXEC_USER})"')
        assert lines[uid + 1 : uid + 5] == [
            EXEC_HOME_LINE,
            'exec_root="$exec_home/fleet/stage/slime-execution-1790600000"',
            'exec_registry="$exec_home/fleet/stage/slime-execution-1790600000.stage/registry"',
            'exec_cache="$exec_home/fleet/cache"',
        ]
        owner = lines.index(
            f'sudo -n chown -R {EXEC_USER}:{EXEC_USER} "$exec_root" "$exec_registry" "$exec_cache"'
        )
        assert lines.index('sudo -n rm -rf "$exec_root" "$exec_registry"') < copy < owner
        # The roster the runner staged beside the tree goes with it, so a
        # suite reading CORVIS_FLEET_REGISTRY as execdocker finds it (MCPs
        # board task e447de72).
        assert lines[copy + 1] == (
            f"sudo -n cp -a '{names.registry_directory(TARGET)}/.' \"$exec_registry/\""
        )
        # The generation is chosen, and its marker written into the copy,
        # as root before the copy and the cache are given to execdocker
        # (MCPs board task 346abc45).
        generation = list(
            tool_caches.sh_build_lines(
                tools='"$exec_cache/tools"',
                marker=f'"$exec_root/{names.GENERATION_MARKER_NAME}"',
                run_as="sudo -n ",
                says_began=True,
            )
        )
        assert lines[copy + 2 : owner] == generation

    def test_every_step_runs_as_execdocker_from_an_empty_environment_on_its_rootless_socket(
        self,
    ) -> None:
        body = _isolated()

        assert f"  sudo -n -u {EXEC_USER} env -i " in body
        assert f"PATH={EXEC_PATH} " in body
        # The check unit's library path crosses the cleared environment, so a
        # wheel's extension finds the C++ runtime on a NixOS node (MCPs board
        # task c2e3f31f).
        assert 'LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}" ' in body
        assert 'DOCKER_HOST="unix:///run/user/$exec_uid/docker.sock" ' in body
        # The submitter crosses the cleared environment, for the deploy
        # suite's fleet hold (MCPs board task 6c4516af A4).
        assert "BOARD_AGENT_LABEL='opus-demo-0929' " in body
        assert 'npm_config_cache="$tool_cache/npm" ' in body
        assert 'POETRY_CACHE_DIR="$tool_cache/pypoetry" ' in body
        assert 'PLAYWRIGHT_BROWSERS_PATH="$tool_cache/ms-playwright" ' in body
        assert "POETRY_VIRTUALENVS_IN_PROJECT='true' " in body
        assert "PYTEST_XDIST_AUTO_NUM_WORKERS='3' " in body
        assert "CORVIS_TEST_MAX_WORKERS='3' " in body
        assert f"CORVIS_FLEET_WORKSPACE={DEMO_PROJECT} " in body
        assert "CORVIS_FLEET_WORKSPACE=. " in _isolated(path="")
        assert 'CORVIS_FLEET_REGISTRY="$exec_registry" ' in body
        # env -i clears the runner's environment, so the keyring switch must
        # cross it too (MCPs board task c837fd8d).
        assert "POETRY_KEYRING_ENABLED='false' " in body
        assert "/var/run/docker.sock" not in body

    def test_install_steps_run_in_order_as_execdocker_and_a_failure_ends_the_build(
        self,
    ) -> None:
        rebuild = InstallStep(phase="workspace-build", argv=("npm", "rebuild"))
        lines = _isolated(install=(NPM_CI, rebuild)).splitlines()
        log = names.log_path(TARGET)
        result = f"{TARGET}/{names.RESULT_NAME}"

        first = lines.index(f"as_exec \"cd '$exec_root' && npm ci\" >> '{log}' 2>&1")
        second = lines.index(f"as_exec \"cd '$exec_root' && npm rebuild\" >> '{log}' 2>&1")
        recipe = lines.index(
            f"as_exec \"cd '$exec_root/{DEMO_PROJECT}' && "
            f"flock --shared '{EXEC_LANE_LOCK}' make check\" >> '{log}' 2>&1"
        )
        assert first < second < recipe
        assert lines[first - 4] == f"printf '$ %s\\n' 'npm ci' >> '{log}'"
        assert "fleet-phase %s started" in lines[first - 2] and "'install'" in lines[first - 2]
        assert "'workspace-build'" in lines[second - 2]
        assert "'check'" in lines[recipe - 2]
        assert lines[first + 1 : first + 12] == [
            "status=$?",
            "set -e",
            'phase_seconds="$(( $(date +%s) - phase_started ))"',
            f"printf 'fleet-phase %s ended %s after %s s, exit %s\\n' 'install' {SH_UTC_STAMP} "
            f'"$phase_seconds" "$status" >> \'{log}\'',
            'phases="${phases:+$phases,}install:${phase_seconds}s"',
            'if [ "$status" -ne 0 ]; then',
            f"  {sh_run_summary_line(log)}",
            '  sudo -n rm -rf "$exec_root"',
            f"  printf '%s\\n' \"$status\" > '{result}'",
            "  exit 0",
            "fi",
        ]

    def test_the_copy_is_removed_before_the_status_is_written_last(self) -> None:
        lines = _isolated().splitlines()
        ci_load = sh_ci_load_end_lines(CI_CPU_STAT, names.log_path(TARGET))
        loaded = len(ci_load)

        assert lines[-7 - loaded] == "set -e"
        assert lines[-5 - loaded].startswith(
            "printf 'fleet-phase %s ended %s after %s s, exit %s\\n' 'check'"
        )
        assert lines[-4 - loaded] == 'phases="${phases:+$phases,}check:${phase_seconds}s"'
        assert lines[-3 - loaded : -3] == ci_load
        assert lines[-3:] == [
            sh_run_summary_line(names.log_path(TARGET)),
            'sudo -n rm -rf "$exec_root"',
            f"printf '%s\\n' \"$status\" > '{TARGET}/{names.RESULT_NAME}'",
        ]

    def test_a_root_project_runs_its_recipe_at_the_copy_s_root(self) -> None:
        lines = _isolated(path="").splitlines()

        assert (
            f"as_exec \"cd '$exec_root' && flock --shared '{EXEC_LANE_LOCK}' make check\""
            f" >> '{names.log_path(TARGET)}' 2>&1"
        ) in lines


def test_a_docker_project_that_declares_loopback_only_is_refused_by_name() -> None:
    # Its build runs as execdocker, which holds no sudo to call
    # fleet-check-unit (MCPs board task 649fcc7a).
    with pytest.raises(ValueError, match="declares both the docker tag and loopback_only"):
        dialect.build_script(
            target=TARGET,
            path="slime",
            workers=2,
            install=(),
            cache_root="/s/cache",
            isolated_docker=True,
            agent="opus-demo-0929",
            loopback_unit="fleet-slime-lavender-1792900000",
        )


@pytest.mark.parametrize(
    ("tags", "isolated"),
    [
        ((NodeTag.LINUX, NodeTag.DOCKER), True),
        ((NodeTag.LINUX,), False),
        ((), False),
    ],
)
def test_the_recipe_is_isolated_exactly_when_the_project_declares_docker(
    tags: tuple[NodeTag, ...], isolated: bool
) -> None:
    built = recipe.recipe_for(_plan(tags), path="slime", install=(NPM_CI,))

    assert built == recipe.Recipe(
        path="slime",
        install=(NPM_CI,),
        isolated_docker=isolated,
        loopback_only=False,
    )
    assert recipe.working_tree_recipe("libs/x", _plan(tags))["isolated_docker"] is isolated


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command from inside this suite and keep what it said.

    Args:
        argv: The command.

    Returns:
        Its exit status and both streams.
    """
    return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=120)


@pytest.mark.host_linux_docker
def test_this_suite_itself_runs_as_execdocker_and_cannot_reach_the_stack() -> None:
    """The isolation, measured from inside the suite's own isolated build.

    tools/fleet-execution-linux requires the docker tag, so the fleet runs
    this very suite through the script :func:`_isolated` pins: copied into
    execdocker's home and run through ``sudo -n -u execdocker env -i``. That
    run IS the real dispatch path, so each fact the isolation claims is read
    here, in the process it produced. Rendering a second copy of the script
    from inside it is not possible: execdocker has no sudo, which is itself
    part of the isolation (MCPs board task a8ee9b21, measured on diphtheria
    2026-09-29: 'sudo: a password is required').
    """
    user = _run(["id", "-un"])
    uid = _run(["id", "-u"])
    environment = _run(["env"]).stdout.splitlines()
    inherited = dict(line.split("=", 1) for line in environment if "=" in line)
    options = _run(["docker", "info", "--format", "{{range .SecurityOptions}}{{println .}}{{end}}"])
    stack = _run(["docker", "-H", "unix:///var/run/docker.sock", "info"])
    escalate = _run(["sudo", "-n", "true"])
    # The recipe holds the lane's lock shared (MCPs board task 86186077), so
    # the lane's prune, which takes it exclusively, cannot start while this
    # check runs, and another check still can.
    lane_lock = EXEC_LANE_LOCK.replace("$exec_uid", uid.stdout.strip())
    prune = _run(["flock", "--exclusive", "--nonblock", lane_lock, "true"])
    beside = _run(["flock", "--shared", "--nonblock", lane_lock, "true"])

    assert user.stdout.strip() == EXEC_USER, user.stderr
    assert "SSH_AUTH_SOCK" not in inherited
    assert inherited["DOCKER_HOST"] == f"unix:///run/user/{uid.stdout.strip()}/docker.sock"
    assert "name=rootless" in options.stdout.splitlines(), options.stderr
    assert stack.returncode != 0, stack.stdout
    assert escalate.returncode != 0, escalate.stdout
    assert (prune.returncode, beside.returncode) == (1, 0), prune.stderr + beside.stderr


@pytest.mark.host_linux_docker
def test_this_suites_venv_runs_the_fleets_pinned_python() -> None:
    """The venv the isolated build made for this suite is on the pinned 3.11.

    Poetry builds a venv on the first ``python`` on PATH and takes no
    ``python3``; with none on execdocker's cleared PATH, as on every Ubuntu
    node until MCPs scripts/host/lib/fleet-exec-docker.sh step_exec_poetry
    linked /usr/local/bin/python to /usr/bin/python3.11 (MCPs board task
    53b4725d), it fell back to /usr/bin/python3.12, which this package's
    ``python = "^3.11"`` accepts. So the interpreter running this case is
    the one Poetry chose in this node's lane.
    """
    running = f"{sys.version_info.major}.{sys.version_info.minor}"

    assert running == REQUIRED_PYTHON, sys.executable


@pytest.mark.host_linux_docker
def test_this_suites_interpreter_carries_its_own_python_h() -> None:
    """The pinned interpreter's development headers are on this node.

    A C extension built in a venv, as Triton builds its cuda_utils helper,
    compiles against the interpreter's include directory, and a node without
    that interpreter's Python.h failed 25 of Model-Trainer's cases (MCPs
    board task bcb7c6ef). The lane's interpreter is /usr/bin/python3.11 on
    an Ubuntu node, the same file the node user's runner lane links, so this
    reads the headers every lane on the node builds against.
    """
    header = pathlib.Path(sysconfig.get_paths()["include"]) / "Python.h"

    assert header.is_file(), str(header)
