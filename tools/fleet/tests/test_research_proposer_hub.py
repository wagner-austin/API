"""Actual SSH and rootless daemon twins, run by the existing fleet hub lane."""

from __future__ import annotations

import shlex
from collections.abc import Callable, Sequence

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, JSONValue, load_json_str

from fleet.core import _test_hooks, remote
from fleet.core.research_proposer import run_proposer
from tests._host import READING_PROPERTY
from tests.test_research_proposer_host import DENIAL_PROGRAM, PYTHON_IMAGE, assert_denied_resources

pytestmark = pytest.mark.host_fleet_hub


@pytest.mark.parametrize("scenario", ["literal", "timeout", "denial"])
def test_actual_remote_proposal_and_timeout_cleanup(
    scenario: str, record_property: Callable[[str, str], None]
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
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        command = shlex.split(argv[-1])
        if "--label" in command:
            labels.append(command[command.index("--label") + 1])
        if scenario == "denial" and command[:4] == ["sudo", "-n", "-u", "execdocker"]:
            argv = (
                *argv[:-1],
                shlex.join(
                    (
                        *command[:4],
                        "env",
                        "WRDS_PROPOSER_CANARY=sentinel",
                        *command[4:],
                    )
                ),
            )
        return _test_hooks._default_run(
            argv,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=unset_env,
            set_env=set_env,
        )

    _test_hooks.run = observed
    if scenario == "timeout":
        with pytest.raises(AppError, match="timed out"):
            run_proposer(socket, image, "import time; time.sleep(60)", {}, host=host, timeout=2)
    elif scenario == "denial":
        setup = (
            "import os,pathlib,tempfile; "
            "assert os.environ['WRDS_PROPOSER_CANARY']=='sentinel'; "
            "root=pathlib.Path(tempfile.mkdtemp(prefix='corvis-proposer-boundary-')); "
            "[(root/name).write_text('private sentinel') "
            "for name in ('policy.json','holdout.csv','credentials')]; print(root)"
        )
        caller = ("sudo", "-n", "-u", "execdocker", "python3", "-c")
        directory = remote.run_ssh(host, (shlex.join((*caller, setup)),)).strip()
        assert directory.startswith("/tmp/corvis-proposer-boundary-")
        read = (
            "import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); "
            "print(json.dumps([(root/name).read_text() "
            "for name in ('policy.json','holdout.csv','credentials')]))"
        )
        observation = (shlex.join((*caller, read, directory)),)
        try:
            before = remote.run_ssh(host, observation)
            assert load_json_str(before) == ["private sentinel"] * 3
            private: list[JSONValue] = [
                directory + "/policy.json",
                directory + "/holdout.csv",
                directory + "/credentials",
                "/proc/1/root" + directory + "/holdout.csv",
            ]
            proposal: JSONObject = {"partition": "validation", "seed": 11}
            denied = run_proposer(
                socket,
                image,
                DENIAL_PROGRAM,
                {"private": private, "policy": private[0], "proposal": proposal},
                host=host,
                timeout=15,
            )
            assert_denied_resources(denied, private, proposal)
            assert remote.run_ssh(host, observation) == before
            record_property(READING_PROPERTY, f"remote={host} denial={denied}")
        finally:
            cleanup = (
                "import pathlib,shutil,sys; root=pathlib.Path(sys.argv[1]); "
                "shutil.rmtree(root); assert not root.exists()"
            )
            remote.run_ssh(host, (shlex.join((*caller, cleanup, directory)),))
    else:
        context: JSONObject = {"proposal": "literal $HOME `hostname` 'quote'\nnext"}
        result = run_proposer(
            socket,
            image,
            "import json,sys; print(json.dumps(json.loads(sys.argv[1])))",
            context,
            host=host,
            timeout=15,
        )
        assert result == context
    assert len(labels) == 1
    remaining = remote.run_ssh(
        host,
        (
            shlex.join(
                (
                    *prefix,
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--filter",
                    "label=" + labels[0],
                )
            ),
        ),
    )
    assert remaining.strip() == ""
    record_property(
        READING_PROPERTY,
        f"remote={host} image={image} scenario={scenario} owned_label={labels[0]} remaining=0",
    )
