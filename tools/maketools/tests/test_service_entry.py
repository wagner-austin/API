"""Each service deploys through the real compose path and passes its own healthcheck.

Board task 465689f5. ``make up-<service>`` at the repository root is
:func:`maketools.workspace.compose_up` over the service's docker-compose.yml:
the image builds from the committed Dockerfile, the container starts on its
entry command, and Docker runs the healthcheck the compose file declares.
Every other case drives that function against recorded fakes. These run it,
then :func:`maketools.workspace.compose_down`, on a Linux node's rootless
execution daemon, and assert the container reached ``healthy`` and that
nothing of the project is left.

The first run of this kind found grandma-api's healthcheck calling curl,
which its runtime image never carried, so Docker could only ever have called
the container unhealthy.

Compose itself does the isolation, through the variables it reads:
COMPOSE_PROJECT_NAME names a project minted for the case, and COMPOSE_FILE
lays tests/execution/service-entry.override.yml over the service's own
file. The deploy functions read them through their own ``environ`` hook,
which the case points at that environment, and the case's own docker
commands are given the same environment.

That override takes away only the fixed container name, the host port and
the restart policy, since production holds those names and ports on
diphtheria, and renames the external network to one this case creates and
removes. Each variable the compose file interpolates is set to a
placeholder, because no service's healthz reaches the credentials.

The services here are the ones whose compose file declares everything the
entry reads. The others read an untracked env_file or a Redis worker, and
each joins with that dependency declared in the case.
"""

from __future__ import annotations

import functools
import os
import secrets
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest

from maketools import _test_hooks
from maketools.workspace import compose_down, compose_up
from tests.conftest import restore_defaults

#: The repository root, which every service's build context names.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

#: The override compose lays over each service's own file.
OVERRIDE: Final[Path] = Path(__file__).resolve().parent / "execution" / "service-entry.override.yml"

#: The value every credential the case sets carries.
PLACEHOLDER: Final[str] = "execution-placeholder"

#: Service directory to the variables its compose file interpolates.
SERVICES: Final[Mapping[str, Mapping[str, str]]] = {
    "grandma-api": {"OPENAI_API_KEY": PLACEHOLDER, "API_TOKEN": PLACEHOLDER},
    "github-stats-api": {"GITHUB_TOKEN": PLACEHOLDER},
}

#: The security option a rootless daemon reports, and the stack's never does.
ROOTLESS: Final[str] = "name=rootless"

#: How long a container may take to pass its healthcheck: every service's
#: start period plus its three retries at a 30-second interval, with margin.
HEALTH_WALL_SECONDS: Final[float] = 300.0

#: How often the case reads the container's state while it starts.
POLL_SECONDS: Final[float] = 5.0

#: The whole case, a first build on a node's empty image store included.
CASE_WALL_SECONDS: Final[int] = 1800


def _run(argv: Sequence[str], directory: Path, environment: Mapping[str, str]) -> str:
    """Run one command that must succeed.

    Args:
        argv: The command.
        directory: Its working directory.
        environment: Its whole environment.

    Returns:
        What it printed on stdout.
    """
    result = subprocess.run(
        list(argv),
        cwd=directory,
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{' '.join(argv)}: {result.stderr}"
    return result.stdout


def _run_environment(service: str, project: str, network: str) -> dict[str, str]:
    """The environment every command of one case runs in.

    Args:
        service: The service directory's name, a key of :data:`SERVICES`.
        project: The compose project minted for the case.
        network: The network the case creates for it.

    Returns:
        This process's environment, read through the ``environ`` hook before
        the case rebinds it, plus the compose project, its two files, the
        network and the service's placeholder variables.
    """
    directory = REPO_ROOT / "services" / service
    return {
        **_test_hooks.environ(),
        "COMPOSE_PROJECT_NAME": project,
        "COMPOSE_PATH_SEPARATOR": os.pathsep,
        "COMPOSE_FILE": os.pathsep.join([str(directory / "docker-compose.yml"), str(OVERRIDE)]),
        "EXECUTION_NETWORK": network,
        **SERVICES[service],
    }


def _state(directory: Path, environment: Mapping[str, str]) -> str:
    """Read the service container's run state and health, as ``running/healthy``.

    Args:
        directory: The service directory.
        environment: The case's environment, which names its compose project.

    Returns:
        The state and the health status, joined by a slash.
    """
    listing = ["docker", "compose", "ps", "--all", "--quiet", "api"]
    container = _run(listing, directory, environment).strip()
    return _run(
        ["docker", "inspect", "--format", "{{.State.Status}}/{{.State.Health.Status}}", container],
        directory,
        environment,
    ).strip()


def _settled_state(directory: Path, environment: Mapping[str, str]) -> str:
    """Wait until the container stops starting, or the health wall passes.

    Args:
        directory: The service directory.
        environment: The case's environment.

    Returns:
        The last state read.
    """
    deadline = _test_hooks.now() + HEALTH_WALL_SECONDS
    state = _state(directory, environment)
    while state == "running/starting" and _test_hooks.now() < deadline:
        _test_hooks.sleep(POLL_SECONDS)
        state = _state(directory, environment)
    return state


def _remove(directory: Path, network: str, environment: Mapping[str, str]) -> None:
    """Remove what the case made: its containers, built images and network.

    Args:
        directory: The service directory.
        network: The network the case created.
        environment: The case's environment.
    """
    _run(
        ["docker", "compose", "down", "--rmi", "local", "--volumes", "--remove-orphans"],
        directory,
        environment,
    )
    _run(["docker", "network", "rm", network], directory, environment)


@pytest.mark.host_linux_docker
@pytest.mark.timeout(CASE_WALL_SECONDS)
@pytest.mark.parametrize("service", sorted(SERVICES))
def test_the_service_deploys_and_passes_its_own_healthcheck(
    service: str, request: pytest.FixtureRequest
) -> None:
    project = f"execution-{secrets.token_hex(6)}"
    network = f"{project}-platform"
    environment = _run_environment(service, project, network)
    options = _run(
        ["docker", "info", "--format", "{{range .SecurityOptions}}{{println .}}{{end}}"],
        REPO_ROOT,
        environment,
    )
    assert [option for option in options.splitlines() if option == ROOTLESS] == [ROOTLESS]

    def case_environ() -> dict[str, str]:
        return dict(environment)

    directory = REPO_ROOT / "services" / service
    _test_hooks.environ = case_environ
    request.addfinalizer(restore_defaults)
    _run(["docker", "network", "create", network], REPO_ROOT, environment)
    request.addfinalizer(functools.partial(_remove, directory, network, environment))

    assert compose_up(directory, build_progress="", git_commit=False) == 0
    assert _settled_state(directory, environment) == "running/healthy"
    assert compose_down([directory]) == 0
    label = f"label=com.docker.compose.project={project}"
    left = ["docker", "ps", "--all", "--quiet", "--filter", label]
    assert _run(left, REPO_ROOT, environment) == ""
