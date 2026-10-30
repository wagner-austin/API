"""Real proposal-process denial and timeout twins on the existing rootless lane."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, JSONValue

from fleet.core import _test_hooks
from fleet.core.research_proposer import run_proposer

# Official library/python manifest resolved from registry-1.docker.io on 2026-10-30.
PYTHON_IMAGE = "python@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337"

pytestmark = pytest.mark.host_linux_docker


def command(argv: tuple[str, ...], timeout: int = 30) -> str:
    """Execute real setup/observation commands, refusing any host failure."""
    result = _test_hooks._default_run(argv, timeout_seconds=timeout)
    assert result["returncode"] == 0, result["stderr"]
    return result["stdout"].strip()


@pytest.fixture(scope="module", name="runtime")
def frozen_runtime() -> tuple[str, str]:
    """Use the lane's daemon and a digest-pinned official Python runtime."""
    assert command(("id", "-un")) == "execdocker"
    uid = command(("id", "-u"))
    socket = f"/run/user/{uid}/docker.sock"
    prefix = ("docker", "--host", f"unix://{socket}")
    command((*prefix, "pull", PYTHON_IMAGE), 180)
    image = command((*prefix, "image", "inspect", "--format", "{{.Id}}", PYTHON_IMAGE))
    return socket, image


DENIAL_PROGRAM = """
import json, os, pathlib, socket, sys
context = json.loads(sys.argv[1])
blocked = []
for name in context['private']:
    try:
        pathlib.Path(name).read_bytes()
    except OSError as error:
        blocked.append({'path': name, 'errno': error.errno})
try:
    pathlib.Path(context['policy']).write_text('changed')
except OSError as error:
    policy_errno = error.errno
else:
    policy_errno = 0
try:
    pathlib.Path('/proposer-probe').write_text('changed')
except OSError as error:
    root_write_errno = error.errno
else:
    root_write_errno = 0
try:
    socket.create_connection(('1.1.1.1', 443), timeout=0.2)
except OSError as error:
    network_error = error.errno
else:
    network_error = 0
status = pathlib.Path('/proc/self/status').read_text().splitlines()
fields = dict(line.split(':', 1) for line in status)
print(json.dumps({
    'blocked': blocked, 'policy_errno': policy_errno,
    'root_write_errno': root_write_errno,
    'credential_absent': 'WRDS_PROPOSER_CANARY' not in os.environ,
    'uid': os.getuid(), 'capabilities': fields['CapEff'].strip(),
    'no_new_privileges': fields['NoNewPrivs'].strip(),
    'network_errno': network_error,
    'memory': pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip(),
    'pids': pathlib.Path('/sys/fs/cgroup/pids.max').read_text().strip(),
    'cpu': pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
    'proposal': context['proposal'],
}))
"""


def test_actual_proposer_denies_private_data_credentials_policy_and_network(
    tmp_path: pathlib.Path, runtime: tuple[str, str]
) -> None:
    """The real child exercises each boundary and returns an allowed JSON proposal."""
    policy = tmp_path / "evaluator-policy.json"
    holdout = tmp_path / "holdout.csv"
    credentials = tmp_path / "credentials"
    for path in (policy, holdout, credentials):
        path.write_text("private sentinel")
    private: list[JSONValue] = [
        str(policy),
        str(holdout),
        str(credentials),
        "/proc/1/root" + str(holdout),
    ]
    proposal: JSONObject = {"partition": "validation", "seed": 11}
    context: JSONObject = {"private": private, "policy": str(policy), "proposal": proposal}
    original = config_test_hooks.get_environment
    environment = original()
    config_test_hooks.get_environment = lambda: {**environment, "WRDS_PROPOSER_CANARY": "sentinel"}
    try:
        result = run_proposer(*runtime, DENIAL_PROGRAM, context, host=None, timeout=15)
    finally:
        config_test_hooks.get_environment = original
    assert result == {
        "blocked": [{"path": name, "errno": 2} for name in private],
        "policy_errno": 2,
        "root_write_errno": 30,
        "credential_absent": True,
        "uid": 65534,
        "capabilities": "0000000000000000",
        "no_new_privileges": "1",
        "network_errno": 101,
        "memory": "268435456",
        "pids": "64",
        "cpu": "100000 100000",
        "proposal": proposal,
    }
    assert [path.read_text() for path in (policy, holdout, credentials)] == ["private sentinel"] * 3


@pytest.mark.parametrize(
    ("program", "timeout", "error"),
    [
        ("raise SystemExit(7)", 15, "RESEARCH_PROPOSER_EXIT: program exited 7"),
        ("import time; time.sleep(60)", 1, "RESEARCH_PROPOSER_PROCESS"),
        ("print('not json')", 15, "Invalid JSON payload"),
        ("print('x' * 9000)", 15, "RESEARCH_PROPOSER_OUTPUT"),
    ],
)
def test_actual_failures_and_timeout_remove_the_owned_container(
    runtime: tuple[str, str], program: str, timeout: int, error: str
) -> None:
    """Run the real daemon through failures and verify this invocation's cleanup."""
    labels: list[str] = []

    def recording_run(
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Record the owned label while every operation still reaches the real daemon."""
        if "--label" in argv:
            labels.append(argv[argv.index("--label") + 1])
        return _test_hooks._default_run(
            argv,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=unset_env,
            set_env=set_env,
        )

    _test_hooks.run = recording_run
    socket, _ = runtime
    prefix = ("docker", "--host", f"unix://{socket}")
    with pytest.raises(ValueError, match=error):
        run_proposer(*runtime, program, {}, host=None, timeout=timeout)
    assert len(labels) == 1
    assert (
        command(
            (
                *prefix,
                "container",
                "ls",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                "label=" + labels[0],
            )
        )
        == ""
    )
