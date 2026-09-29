"""A docker project's build runs as execdocker, never as the runner.

MCPs board task a8ee9b21. On diphtheria the runner's user is in the docker
group, so a docker project built as that user would reach the production
stack's daemon. :mod:`fleet.core.linux_isolated_build` runs such a build as
execdocker against its rootless daemon instead. The text cases pin the
properties that make that an isolation; the host case runs inside
tools/fleet-execution-linux's own isolated build, which the docker tag
routes through that script, and proves from there that it runs as
execdocker, sees a rootless daemon, is refused by the stack's socket and
cannot escalate.
"""

from __future__ import annotations

import subprocess

import pytest

from fleet.contracts.project import ProjectConfig
from fleet.contracts.tags import NodeTag
from fleet.core import dispatch, names
from fleet.core.dialect_linux import PROLOGUE, LinuxDialect
from fleet.core.dialect_windows import WindowsDialect
from fleet.core.linux_isolated_build import EXEC_PATH, EXEC_USER
from tests.conftest import DEMO_PROJECT

TARGET = "/home/corvis/fleet/stage/slime-execution-1790600000"


def _isolated(
    *, target: str = TARGET, path: str = DEMO_PROJECT, install: tuple[tuple[str, ...], ...] = ()
) -> str:
    """Render the isolated build for one run.

    Args:
        target: The dispatch directory.
        path: The recipe's directory inside the export.
        install: The install steps.

    Returns:
        The script's text.
    """
    return LinuxDialect().build_script(
        target=target,
        path=path,
        workers=3,
        install=install,
        cache_root="/s/cache",
        isolated_docker=True,
        elevated=False,
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
    )


class TestIsolatedText:
    def test_it_starts_with_the_prologue_and_copies_the_tree_into_execdocker_s_home(
        self,
    ) -> None:
        lines = _isolated().splitlines()

        assert _isolated().startswith(PROLOGUE)
        assert f'exec_uid="$(id -u {EXEC_USER})"' in lines
        assert 'exec_root="$exec_home/fleet/stage/slime-execution-1790600000"' in lines
        copy = lines.index(f"sudo -n cp -a '{TARGET}/.' \"$exec_root/\"")
        owner = lines.index(f'sudo -n chown -R {EXEC_USER}:{EXEC_USER} "$exec_root" "$exec_cache"')
        assert lines.index('sudo -n rm -rf "$exec_root"') < copy < owner

    def test_every_step_runs_as_execdocker_from_an_empty_environment_on_its_rootless_socket(
        self,
    ) -> None:
        body = _isolated()

        assert f"  sudo -n -u {EXEC_USER} env -i " in body
        assert f"PATH={EXEC_PATH} " in body
        assert 'DOCKER_HOST="unix:///run/user/$exec_uid/docker.sock" ' in body
        assert 'npm_config_cache="$exec_cache/npm" ' in body
        assert "PYTEST_XDIST_AUTO_NUM_WORKERS='3' " in body
        assert "/var/run/docker.sock" not in body

    def test_install_steps_run_in_order_as_execdocker_and_a_failure_ends_the_build(
        self,
    ) -> None:
        lines = _isolated(install=(("npm", "ci"), ("npm", "rebuild"))).splitlines()
        log = names.log_path(TARGET)
        result = f"{TARGET}/{names.RESULT_NAME}"

        first = lines.index(f"as_exec \"cd '$exec_root' && npm ci\" >> '{log}' 2>&1")
        second = lines.index(f"as_exec \"cd '$exec_root' && npm rebuild\" >> '{log}' 2>&1")
        recipe = lines.index(
            f"as_exec \"cd '$exec_root/{DEMO_PROJECT}' && make check\" >> '{log}' 2>&1"
        )
        assert first < second < recipe
        assert lines[first - 2] == f"printf '$ %s\\n' 'npm ci' >> '{log}'"
        assert lines[first + 1 : first + 8] == [
            "status=$?",
            "set -e",
            'if [ "$status" -ne 0 ]; then',
            '  sudo -n rm -rf "$exec_root"',
            f"  printf '%s\\n' \"$status\" > '{result}'",
            "  exit 0",
            "fi",
        ]

    def test_the_copy_is_removed_before_the_status_is_written_last(self) -> None:
        lines = _isolated().splitlines()

        assert lines[-3:] == [
            "set -e",
            'sudo -n rm -rf "$exec_root"',
            f"printf '%s\\n' \"$status\" > '{TARGET}/{names.RESULT_NAME}'",
        ]

    def test_a_root_project_runs_its_recipe_at_the_copy_s_root(self) -> None:
        lines = _isolated(path="").splitlines()

        assert (
            f"as_exec \"cd '$exec_root' && make check\" >> '{names.log_path(TARGET)}' 2>&1"
        ) in lines


def test_a_windows_node_refuses_to_render_a_docker_project_s_build() -> None:
    with pytest.raises(ValueError, match="a Windows node has no rootless daemon"):
        WindowsDialect().build_script(
            target="C:/s/run-1",
            path="slime",
            workers=2,
            install=(),
            cache_root="C:/s/cache",
            isolated_docker=True,
            elevated=False,
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
    recipe = dispatch.recipe_for(_plan(tags), path="slime", install=(("npm", "ci"),))

    assert recipe == dispatch.Recipe(
        path="slime", install=(("npm", "ci"),), isolated_docker=isolated, elevated=False
    )
    assert dispatch.working_tree_recipe("libs/x", _plan(tags))["isolated_docker"] is isolated


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

    assert user.stdout.strip() == EXEC_USER, user.stderr
    assert "SSH_AUTH_SOCK" not in inherited
    assert inherited["DOCKER_HOST"] == f"unix:///run/user/{uid.stdout.strip()}/docker.sock"
    assert "name=rootless" in options.stdout.splitlines(), options.stderr
    assert stack.returncode != 0, stack.stdout
    assert escalate.returncode != 0, escalate.stdout
