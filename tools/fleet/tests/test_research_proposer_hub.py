"""Actual SSH and rootless daemon twins, run by the existing fleet hub lane."""

from __future__ import annotations

import shlex
from collections.abc import Callable, Sequence

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject

from fleet.core import _test_hooks, remote
from fleet.core.research_proposer import run_proposer
from tests._host import READING_PROPERTY
from tests.test_research_proposer_host import PYTHON_IMAGE

pytestmark = pytest.mark.host_fleet_hub


@pytest.mark.parametrize("failure", [False, True])
def test_actual_remote_proposal_and_timeout_cleanup(
    failure: bool, record_property: Callable[[str, str], None]
) -> None:
    """Reach the declared Colossus lane over real SSH and verify owned cleanup."""
    host = "colossus"
    uid = remote.run_ssh(host, ("id", "-u", "execdocker")).strip()
    socket = f"/run/user/{uid}/docker.sock"
    prefix = ("sudo", "-n", "-u", "execdocker", "docker", "--host", f"unix://{socket}")
    image = remote.run_ssh(
        host, (shlex.join((*prefix, "image", "inspect", "--format", "{{.Id}}", PYTHON_IMAGE)),)
    ).strip()
    labels: list[str] = []

    def observed(
        argv: Sequence[str], *, timeout_seconds: int, stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (), set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        command = shlex.split(argv[-1])
        if "--label" in command:
            labels.append(command[command.index("--label") + 1])
        return _test_hooks._default_run(
            argv, timeout_seconds=timeout_seconds, stdin_bytes=stdin_bytes,
            unset_env=unset_env, set_env=set_env,
        )

    _test_hooks.run = observed
    if failure:
        with pytest.raises(AppError, match="timed out"):
            run_proposer(socket, image, "import time; time.sleep(60)", {}, host=host, timeout=2)
    else:
        context: JSONObject = {"proposal": "literal $HOME `hostname` 'quote'\nnext"}
        result = run_proposer(
            socket, image, "import json,sys; print(json.dumps(json.loads(sys.argv[1])))",
            context, host=host, timeout=15,
        )
        assert result == context
    assert len(labels) == 1
    remaining = remote.run_ssh(host, (shlex.join((
        *prefix, "container", "ls", "--all", "--quiet", "--filter", "label=" + labels[0],
    )),))
    assert remaining.strip() == ""
    record_property(
        READING_PROPERTY,
        f"remote={host} image={image} timeout_case={failure} owned_label={labels[0]} remaining=0",
    )
