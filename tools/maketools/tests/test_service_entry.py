"""Each service deploys through the real compose path and passes its own healthcheck.

Board task 465689f5. ``make up-<service>`` at the repository root is
:func:`maketools.workspace.compose_up` over the service's docker-compose.yml:
the image builds from the committed Dockerfile, the container starts on its
entry command, and Docker runs the healthcheck the compose file declares.
Every other case drives that function against recorded fakes. These run it,
then :func:`maketools.workspace.compose_down`, on a Linux node's rootless
execution daemon, and assert the api reached ``healthy``, that a service's
RQ worker is still running beside it, and that nothing of the project is
left.

The first run of this kind found grandma-api's healthcheck calling curl,
which its runtime image never carried, so Docker could only ever have called
the container unhealthy.

Compose itself does the isolation, through the variables it reads:
COMPOSE_PROJECT_NAME names a project minted for the case, and COMPOSE_FILE
lays tests/execution/service-entry.override.yml over the service's own
file, then service-entry.worker.override.yml for a service with a worker,
which brings the run its own Redis under the stack's name. The deploy
functions read them through their own ``environ`` hook, which the case
points at that environment, and the case's own docker commands are given
the same environment.

The overrides take away only the fixed container names, the host port and
the restart policy, since production holds those names and ports on
diphtheria, and rename the external network to one this case creates and
removes. The service's untracked .env is replaced by a file the case writes
from :data:`SERVICES`: each variable its settings require at boot, measured
by booting it without, set to a placeholder, since no healthcheck reaches a
credential. The case exports the same variables for the ones the compose
file interpolates.

Four services are absent, each for a reason measured on 2026-09-30.
Model-Trainer, Art-Trainer and covenant-radar-api reserve an NVIDIA GPU,
which the rootless daemon has no runtime for. handwriting-ai boots, but its
readyz answers 503 'model not loaded' until a trained model sits in its
artifacts, which neither its image nor the repository carries, so a clean
deploy can never pass its healthcheck.
"""

from __future__ import annotations

import functools
import secrets
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, TypedDict

import pytest

from maketools import _test_hooks
from maketools.workspace import compose_down, compose_up
from tests.conftest import restore_defaults

#: The repository root, which every service's build context names.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

#: The overrides compose lays over each service's own file.
OVERRIDES: Final[Path] = Path(__file__).resolve().parent / "execution"

#: The value every variable the case sets carries.
PLACEHOLDER: Final[str] = "execution-placeholder"


class ServiceEntry(TypedDict):
    """What one service's case needs.

    Attributes:
        worker: Whether the compose file runs an RQ worker beside the api,
            which lays the worker override and its Redis.
        variables: Each variable the service requires at boot, as the
            placeholder the case sets. opportunity-radar-api's GITHUB_REPO
            is a real owner/repo because its settings parse one, and with
            its token it scans through the API instead of looking for a
            libs directory its image does not carry.
    """

    worker: bool
    variables: Mapping[str, str]


#: Service directory to what its case needs.
SERVICES: Final[Mapping[str, ServiceEntry]] = {
    "grandma-api": ServiceEntry(
        worker=False, variables={"OPENAI_API_KEY": PLACEHOLDER, "API_TOKEN": PLACEHOLDER}
    ),
    "github-stats-api": ServiceEntry(worker=False, variables={"GITHUB_TOKEN": PLACEHOLDER}),
    "opportunity-radar-api": ServiceEntry(
        worker=False,
        variables={
            "KAGGLE_API_TOKEN": PLACEHOLDER,
            "GITHUB_TOKEN": PLACEHOLDER,
            "GITHUB_REPO": "wagner-austin/API",
        },
    ),
    "data-bank-api": ServiceEntry(
        worker=True,
        variables={
            "API_UPLOAD_KEYS": PLACEHOLDER,
            "API_READ_KEYS": PLACEHOLDER,
            "API_DELETE_KEYS": PLACEHOLDER,
        },
    ),
    "qr-api": ServiceEntry(worker=True, variables={}),
    "music-wrapped-api": ServiceEntry(worker=True, variables={}),
    "transcript-api": ServiceEntry(worker=True, variables={"OPENAI_API_KEY": PLACEHOLDER}),
    "turkic-api": ServiceEntry(worker=True, variables={"TURKIC_DATA_BANK_API_KEY": PLACEHOLDER}),
}

#: The security option a rootless daemon reports, and the stack's never does.
ROOTLESS: Final[str] = "name=rootless"

#: How long a container may take to pass its healthcheck: every service's
#: start period plus its retries at its interval, with margin.
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


def _run_environment(service: str, project: str, env_file: Path) -> dict[str, str]:
    """The environment every command of one case runs in.

    Args:
        service: The service directory's name, a key of :data:`SERVICES`.
        project: The compose project minted for the case.
        env_file: The file the case wrote the service's variables to.

    Returns:
        This process's environment, read through the ``environ`` hook before
        the case rebinds it, plus the compose project, its files, the
        network, the env file and the service's variables.
    """
    entry = SERVICES[service]
    files = [
        REPO_ROOT / "services" / service / "docker-compose.yml",
        OVERRIDES / "service-entry.override.yml",
    ]
    if entry["worker"]:
        files.append(OVERRIDES / "service-entry.worker.override.yml")
    return {
        **_test_hooks.environ(),
        "COMPOSE_PROJECT_NAME": project,
        "COMPOSE_PATH_SEPARATOR": ":",
        "COMPOSE_FILE": ":".join(str(path) for path in files),
        "EXECUTION_NETWORK": f"{project}-platform",
        "EXECUTION_ENV_FILE": str(env_file),
        **entry["variables"],
    }


def _state(directory: Path, environment: Mapping[str, str], name: str, template: str) -> str:
    """Read one of the project's containers through ``docker inspect``.

    Args:
        directory: The service directory.
        environment: The case's environment, which names its compose project.
        name: The compose service, ``api`` or ``worker``.
        template: The inspect format.

    Returns:
        What the template printed.
    """
    listing = ["docker", "compose", "ps", "--all", "--quiet", name]
    container = _run(listing, directory, environment).strip()
    inspect = ["docker", "inspect", "--format", template, container]
    return _run(inspect, directory, environment).strip()


def _settled_health(directory: Path, environment: Mapping[str, str]) -> str:
    """Wait until the api stops starting, or the health wall passes.

    Args:
        directory: The service directory.
        environment: The case's environment.

    Returns:
        The last state read, as ``running/healthy``.
    """
    template = "{{.State.Status}}/{{.State.Health.Status}}"
    deadline = _test_hooks.now() + HEALTH_WALL_SECONDS
    state = _state(directory, environment, "api", template)
    while state == "running/starting" and _test_hooks.now() < deadline:
        _test_hooks.sleep(POLL_SECONDS)
        state = _state(directory, environment, "api", template)
    return state


def _remove(directory: Path, environment: Mapping[str, str]) -> None:
    """Remove what the case made: its containers, built images and network.

    Args:
        directory: The service directory.
        environment: The case's environment, which names the network.
    """
    down = ["docker", "compose", "down", "--rmi", "local", "--volumes", "--remove-orphans"]
    _run(down, directory, environment)
    _run(["docker", "network", "rm", environment["EXECUTION_NETWORK"]], directory, environment)


@pytest.mark.host_linux_docker
@pytest.mark.timeout(CASE_WALL_SECONDS)
@pytest.mark.parametrize("service", sorted(SERVICES))
def test_the_service_deploys_and_passes_its_own_healthcheck(
    service: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    project = f"execution-{secrets.token_hex(6)}"
    env_file = tmp_path / "service.env"
    env_file.write_text(
        "".join(f"{name}={value}\n" for name, value in SERVICES[service]["variables"].items()),
        encoding="utf-8",
    )
    environment = _run_environment(service, project, env_file)
    info = ["docker", "info", "--format", "{{range .SecurityOptions}}{{println .}}{{end}}"]
    options = _run(info, REPO_ROOT, environment)
    assert [option for option in options.splitlines() if option == ROOTLESS] == [ROOTLESS]

    def case_environ() -> dict[str, str]:
        return dict(environment)

    directory = REPO_ROOT / "services" / service
    _test_hooks.environ = case_environ
    request.addfinalizer(restore_defaults)
    _run(["docker", "network", "create", environment["EXECUTION_NETWORK"]], REPO_ROOT, environment)
    request.addfinalizer(functools.partial(_remove, directory, environment))

    assert compose_up(directory, build_progress="", git_commit=False) == 0
    assert _settled_health(directory, environment) == "running/healthy"
    if SERVICES[service]["worker"]:
        assert _state(directory, environment, "worker", "{{.State.Status}}") == "running"
    assert compose_down([directory]) == 0
    label = f"label=com.docker.compose.project={project}"
    left = ["docker", "ps", "--all", "--quiet", "--filter", label]
    assert _run(left, REPO_ROOT, environment) == ""
