"""Run untrusted proposal programs without evaluator files or transport credentials.

The trusted caller selects an already-present immutable Python image and passes
only approved context. The existing execdocker lane owns the rootless daemon;
this module neither provisions another account nor creates another job queue.
Returned objects still require the maintained research admission controller.
An administrative caller or a compromised kernel is outside this boundary.
"""

from __future__ import annotations

import re
from uuid import uuid4

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_list,
    require_str,
)

from fleet.contracts.research import require_digest
from fleet.core import _test_hooks
from fleet.core.linux_isolated_build import EXEC_PATH

MAX_BYTES = 8192


def _checked(prefix: tuple[str, ...], args: tuple[str, ...], timeout: int = 30) -> str:
    """Run a bounded daemon operation and propagate its actual failure."""
    result = _test_hooks.run((*prefix, *args), timeout_seconds=timeout)
    if result["returncode"] != 0:
        raise ValueError(
            f"RESEARCH_PROPOSER_PROCESS: {args[0]} exit {result['returncode']}; {result['stderr']}"
        )
    return result["stdout"]


def _runtime(socket: str, image: str) -> tuple[str, ...]:
    """Require the local rootless daemon and enforceable cgroup resource limits."""
    if re.fullmatch(r"/run/user/[1-9][0-9]*/docker.sock", socket) is None:
        raise ValueError("RESEARCH_PROPOSER_SOCKET: require a local user daemon socket")
    require_digest(image.removeprefix("sha256:"))
    if not image.startswith("sha256:"):
        raise ValueError("RESEARCH_PROPOSER_IMAGE: require an immutable local image ID")
    prefix = (
        "/usr/bin/env",
        "-i",
        f"PATH={EXEC_PATH}",
        "HOME=/nonexistent",
        "docker",
        "--host",
        f"unix://{socket}",
    )
    info = narrow_json_to_dict(load_json_str(_checked(prefix, ("info", "--format", "{{json .}}"))))
    options = [
        require_str({"item": item}, "item") for item in require_list(info, "SecurityOptions")
    ]
    if (
        "name=rootless" not in options
        or any(info.get(key) is not True for key in ("MemoryLimit", "CpuCfsQuota", "PidsLimit"))
        or info.get("CgroupVersion") != "2"
    ):
        raise ValueError(
            "RESEARCH_PROPOSER_RUNTIME: require rootless cgroup-v2 resource enforcement"
        )
    volumes = load_json_str(
        _checked(prefix, ("image", "inspect", "--format", "{{json .Config.Volumes}}", image))
    )
    if volumes not in (None, {}):
        raise ValueError("RESEARCH_PROPOSER_IMAGE: implicit image volumes are forbidden")
    return prefix


def _owned(prefix: tuple[str, ...], label: str) -> tuple[str, ...]:
    """Read the full IDs of containers bearing this invocation's private nonce."""
    output = _checked(
        prefix, ("container", "ls", "--all", "--quiet", "--no-trunc", "--filter", f"label={label}")
    )
    return tuple(require_digest(item) for item in output.split())


def _remove_owned(prefix: tuple[str, ...], label: str) -> None:
    """Remove only this invocation's containers and verify none remain."""
    owned = _owned(prefix, label)
    if owned:
        _checked(prefix, ("container", "rm", "--force", *owned))
    if _owned(prefix, label):
        raise ValueError("RESEARCH_PROPOSER_CLEANUP: owned container remains after removal")


def run_proposer(
    socket: str, image: str, program: str, context: JSONObject, *, timeout: int
) -> JSONObject:
    """Execute a bounded offline proposal program under the existing rootless lane.

    Args:
        socket: Provisioned execdocker daemon socket, never a network endpoint.
        image: Evaluator-approved local sha256 image ID containing Python and env.
        program: Python source; receives approved JSON context in sys.argv[1].
        context: Approved public/validation context, never credentials or holdout.
        timeout: Wall-clock seconds allowed for the program, between 1 and 120.

    Returns:
        Untrusted proposal object for the existing admission controller to check.

    Raises:
        ValueError: RESEARCH_PROPOSER_* for unsafe runtime, bounds, process exit,
            timeout, output or cleanup failure. JSON decoding errors propagate.
        OSError: The maintained process boundary cannot start a daemon operation.
    """
    payload = dump_json_str(context)
    if (
        not 1 <= timeout <= 120
        or not program
        or any(len(value.encode()) > MAX_BYTES for value in (program, payload))
    ):
        raise ValueError("RESEARCH_PROPOSER_BOUNDS: bounded program, context and deadline required")
    prefix = _runtime(socket, image)
    label = f"corvis.research.proposer={uuid4().hex}"
    try:
        identifier = _checked(
            prefix,
            (
                "container",
                "create",
                "--pull=never",
                "--label",
                label,
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--user=65534:65534",
                "--pids-limit=64",
                "--memory=256m",
                "--memory-swap=256m",
                "--cpus=1",
                "--log-driver=local",
                "--log-opt=max-size=16k",
                "--log-opt=max-file=1",
                "--log-opt=compress=false",
                "--entrypoint=/usr/bin/env",
                image,
                "-i",
                "PATH=/usr/local/bin:/usr/bin:/bin",
                "/usr/local/bin/python",
                "-I",
                "-c",
                program,
                payload,
            ),
        ).strip()
        require_digest(identifier)
        _checked(prefix, ("container", "start", identifier))
        status = _checked(prefix, ("container", "wait", identifier), timeout).strip()
        if status != "0":
            raise ValueError(f"RESEARCH_PROPOSER_EXIT: program exited {status}")
        output = _checked(prefix, ("container", "logs", identifier))
        if len(output.encode()) > MAX_BYTES:
            raise ValueError("RESEARCH_PROPOSER_OUTPUT: proposal exceeds the output limit")
        return narrow_json_to_dict(load_json_str(output))
    finally:
        _remove_owned(prefix, label)


__all__ = ["MAX_BYTES", "run_proposer"]
