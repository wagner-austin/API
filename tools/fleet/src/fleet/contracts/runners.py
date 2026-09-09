"""The CI runner roster: which runner installs each machine carries, and why.

This document exists because on 2026-09-08 every fact in it was folklore. A
Windows scheduled task (``wsl-keepalive``) was the only thing holding six
runners' WSL VM open, and nothing recorded it -- a restart that missed it took
both repos' CI pools offline silently. The licensed game tree, the llama.cpp
checkout and ``/data`` had each been placed on lavender by hand, findable only
by reading a workflow's assert step or a board post. ``fleet-runners audit``
scores a host against this roster; ``fleet-runners render`` emits the converge
script for a new one.

THIS IS THE THIRD MACHINE DOCUMENT, NOT A COPY OF THE OTHER TWO. The fleet is
already written down twice -- ``fleet-mcp/fleet-nodes.json`` (MCPs) owns which
machines exist and whether one should answer; ``fleet.json`` here owns what a
machine can run for DISPATCH. This roster owns what a machine serves to
GITHUB ACTIONS: runner installs, their labels, and the host assets their job
classes require. The columns barely overlap, and merging them was already
considered and rejected once (see :mod:`fleet.core.registry`).

ABSENCE IS NOT A STATE, same rule as :mod:`fleet.contracts.node`: optional
concepts are spelled as explicit ``null`` rather than omitted keys, so a host
nobody finished describing cannot be told apart from one that genuinely has
no keepalive task.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_list,
    require_str,
    require_str_list,
)
from typing_extensions import TypedDict


class RunnerInstall(TypedDict):
    """One GitHub Actions runner install on a host.

    Attributes:
        repo: The repository it is registered to, ``owner/name`` form --
            registrations are per-repo, which is why one box carries many.
        runner_name: The name shown by the repo's runner registration, e.g.
            ``lavender-wsl-2``. Names are per-repo: two installs on one box
            may share a name across repos without being the same install.
        side: Which execution environment the install lives in: ``"wsl"``
            (inside the host's WSL distro, run by systemd) or ``"windows"``
            (on the host OS, run by a Windows service). Added 2026-09-09,
            when three Windows-side installs (MCPs, portable-claude,
            tree-bot) were live on lavender and invisible to the audit and
            the load sampler because the roster could not describe them.
        service: The service that runs it. For ``wsl``, the systemd unit
            (``actions.runner.<owner>-<repo>.<name>.service``); for
            ``windows``, the Windows service name (same, minus
            ``.service``). Recorded exactly rather than derived, because
            "is the service running" is the audit question.
        workdir: Absolute path of the install's ``_work`` tree inside its
            execution environment -- a ``/home/...`` path for ``wsl``, a
            drive-letter path for ``windows``; the decoder enforces the
            match. This is the path a poetry in-project venv bakes into its
            shebangs -- the fact behind the 2026-09-08 cache poisoning --
            and the audit asserts it exists.
        labels: The custom labels the install answers to, e.g.
            ``["lavender-wsl"]``. Labels are the pool: a workflow targets a
            label, never a machine, which is what makes a new box a five
            minute join instead of a workflow edit.
    """

    repo: str
    runner_name: str
    side: Literal["wsl", "windows"]
    service: str
    workdir: str
    labels: list[str]


class FileAsset(TypedDict):
    """One path a host must hold for its runner jobs to be honest.

    Attributes:
        path: Absolute path inside the execution environment.
        sha256: Content pin for a file whose exact bytes are load-bearing
            (the RustedWarfareBot provenance jar), or ``None`` for a path
            whose presence is the whole requirement. Explicitly null rather
            than absent -- an unpinned asset is a decision, not an oversight.
        writable: Whether runner jobs write under it (``/data``). Presence
            without writability satisfied the naive check and then failed
            every Model-Trainer checkpoint test, which is why this is its
            own field rather than implied.
        reason: One sentence naming the job class that needs it. The audit
            prints it beside a drift line so the reader learns what breaks
            without opening this file.
        manual: True when the asset can never be fetched by a script -- the
            licensed game tree, which must not enter the repo, its caches,
            or any hosted VM. ``render`` emits these as a loud REQUIRED
            MANUAL STEP instead of a command.
        provision_command: The exact shell command that creates the asset on
            a new host, for an asset that is neither manual nor a writable
            directory -- e.g. the ``git clone`` for the llama.cpp checkout.
            Null for a manual asset (nothing may fetch it), null for a
            writable one (the mkdir+chown is implied by ``writable``), and
            REQUIRED otherwise: a fetchable asset with no fetch command
            renders a provision that stops at "missing" with no way forward,
            which is a dead end this contract refuses to describe.
    """

    path: str
    sha256: str | None
    writable: bool
    reason: str
    manual: bool
    provision_command: str | None


class HostRunnerSpec(TypedDict):
    """One machine serving GitHub Actions, and everything it must hold.

    Attributes:
        name: The host's fleet name, matching ``fleet.json`` and
            ``fleet-nodes.json`` so the three documents can be joined.
        host: SSH destination -- an alias from ``~/.ssh/config``, the same
            rule as :class:`fleet.contracts.node.NodeConfig`.
        wsl_distro: The WSL distribution the runners live inside. Required
            rather than nullable: the fleet's remote layer drives Windows
            hosts (see :mod:`fleet.core.remote`), so a bare-Linux CI host is
            a machine this tooling cannot audit yet -- and a roster must not
            describe what the audit cannot score, or the entry sits
            permanently unverified while reading as covered. When a bare
            Linux box joins the fleet, this field and the audit widen
            together.
        keepalive_task: The Windows scheduled task that holds the WSL VM
            open, or ``None`` on a host that needs none. THIS FIELD IS THE
            INCIDENT THIS MODULE EXISTS FOR: systemd services do not keep a
            WSL VM alive, and the task's name lived nowhere until it cost
            six runners.
        wslconfig_min_memory_gb: The least memory ``.wslconfig`` must grant
            the VM, or ``None`` when the host default is acceptable.
            Lavender's pool needs 26: two CI jobs peak near 10GB each and
            the Windows default grants half the machine.
        scratch_dir: Absolute Windows directory on the host for transient
            audit scripts, the same role ``stage_root`` plays for dispatch.
            Declared here rather than joined from ``fleet.json`` at runtime:
            an audit that needed a second document to locate a writable
            directory would fail on exactly the machine the second document
            forgot.
        gpu_required: Whether runner jobs on this host digest a real GPU
            (Model-Trainer's default hooks do). Audited via ``nvidia-smi``
            inside the execution environment.
        systemd_timers: Timers that must be enabled inside the execution
            environment, e.g. ``ci-clean.timer`` -- hygiene is part of the
            provision, not an afterthought on one box's disk.
        installs: Every runner install this host carries.
        assets: Every path its job classes require.
    """

    name: str
    host: str
    wsl_distro: str
    keepalive_task: str | None
    wslconfig_min_memory_gb: int | None
    scratch_dir: str
    gpu_required: bool
    systemd_timers: list[str]
    installs: list[RunnerInstall]
    assets: list[FileAsset]


class RunnerSpec(TypedDict):
    """The whole roster.

    Attributes:
        hosts: Every machine serving GitHub Actions for these repositories.
    """

    hosts: list[HostRunnerSpec]


def encode_runner_install(install: RunnerInstall) -> JSONObject:
    """Encode one runner install.

    Args:
        install: The install to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "repo": install["repo"],
        "runner_name": install["runner_name"],
        "side": install["side"],
        "service": install["service"],
        "workdir": install["workdir"],
        "labels": list(install["labels"]),
    }


def decode_runner_install(value: JSONValue) -> RunnerInstall:
    """Decode and validate one runner install.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated install.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, ``repo`` is not ``owner/name``, ``labels`` is empty,
            ``side`` is not ``wsl`` or ``windows``, or ``workdir``'s shape
            does not match the side -- a Windows path on a wsl install (or
            the reverse) describes an install that cannot exist, and the
            audit pointed at it would report drift no command can fix. A
            runner with no custom label is unreachable by every workflow in
            these repositories, so declaring one would record an install
            nothing can use.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"runner install must be a JSON object, got {type(value).__name__}")
    repo = require_str(value, "repo")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise JSONTypeError(f"repo must be 'owner/name', got {repo!r}")
    labels = require_str_list(value, "labels")
    if not labels:
        raise JSONTypeError(
            "labels must be non-empty; a runner with no custom label cannot be targeted "
            "by any workflow here and would sit registered but unreachable"
        )
    raw_side = require_str(value, "side")
    if raw_side not in ("wsl", "windows"):
        raise JSONTypeError(f"side must be 'wsl' or 'windows', got {raw_side!r}")
    side: Literal["wsl", "windows"] = "wsl" if raw_side == "wsl" else "windows"
    workdir = require_str(value, "workdir")
    if side == "wsl" and not workdir.startswith("/"):
        raise JSONTypeError(
            f"a wsl install's workdir must be an absolute POSIX path, got {workdir!r}"
        )
    if side == "windows" and not (
        len(workdir) > 2 and workdir[0].isalpha() and workdir[1] == ":" and workdir[2] in "/\\"
    ):
        raise JSONTypeError(
            f"a windows install's workdir must be a drive-letter path, got {workdir!r}"
        )
    return RunnerInstall(
        repo=repo,
        runner_name=require_str(value, "runner_name"),
        side=side,
        service=require_str(value, "service"),
        workdir=workdir,
        labels=labels,
    )


def encode_file_asset(asset: FileAsset) -> JSONObject:
    """Encode one host asset.

    Args:
        asset: The asset to encode.

    Returns:
        JSON-serialisable mapping with ``sha256`` present and null when
        unpinned rather than omitted.
    """
    return {
        "path": asset["path"],
        "sha256": asset["sha256"],
        "writable": asset["writable"],
        "reason": asset["reason"],
        "manual": asset["manual"],
        "provision_command": asset["provision_command"],
    }


def decode_file_asset(value: JSONValue) -> FileAsset:
    """Decode and validate one host asset.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated asset.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, ``sha256`` is absent rather than explicitly null, a
            declared pin is not 64 hex characters, or ``provision_command``
            contradicts the asset's kind -- present on a manual or writable
            asset, or absent on one that is neither. A truncated pin matches
            nothing and would report every audit as drift -- rejecting it at
            decode names the actual fault.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"file asset must be a JSON object, got {type(value).__name__}")
    if "sha256" not in value:
        raise JSONTypeError(
            "file asset must declare 'sha256', using null for an unpinned path. An absent "
            "key cannot be told apart from a pin nobody finished writing down."
        )
    raw_pin = value["sha256"]
    pin: str | None
    if raw_pin is None:
        pin = None
    elif isinstance(raw_pin, str):
        if len(raw_pin) != 64 or any(c not in "0123456789abcdef" for c in raw_pin):
            raise JSONTypeError(
                f"sha256 must be 64 lowercase hex characters, got {raw_pin!r}; a malformed "
                "pin matches nothing and would score every audit as drift"
            )
        pin = raw_pin
    else:
        raise JSONTypeError(f"sha256 must be a string or null, got {type(raw_pin).__name__}")
    writable = require_bool(value, "writable")
    manual = require_bool(value, "manual")
    provision_command = _optional_null_str(value, "provision_command")
    if provision_command is not None and (manual or writable):
        raise JSONTypeError(
            "provision_command must be null on a manual or writable asset: a manual "
            "asset may not be fetched by any script, and a writable one's mkdir+chown "
            "is implied by 'writable'"
        )
    if provision_command is None and not manual and not writable:
        raise JSONTypeError(
            "a fetchable asset must carry provision_command: without it the rendered "
            "provision stops at 'missing' with no way forward"
        )
    return FileAsset(
        path=require_str(value, "path"),
        sha256=pin,
        writable=writable,
        reason=require_str(value, "reason"),
        manual=manual,
        provision_command=provision_command,
    )


def encode_host_runner_spec(spec: HostRunnerSpec) -> JSONObject:
    """Encode one host's declaration.

    Args:
        spec: The host to encode.

    Returns:
        JSON-serialisable mapping with every optional concept present and
        null rather than omitted.
    """
    return {
        "name": spec["name"],
        "host": spec["host"],
        "wsl_distro": spec["wsl_distro"],
        "keepalive_task": spec["keepalive_task"],
        "wslconfig_min_memory_gb": spec["wslconfig_min_memory_gb"],
        "scratch_dir": spec["scratch_dir"],
        "gpu_required": spec["gpu_required"],
        "systemd_timers": list(spec["systemd_timers"]),
        "installs": [encode_runner_install(install) for install in spec["installs"]],
        "assets": [encode_file_asset(asset) for asset in spec["assets"]],
    }


def _optional_null_str(obj: JSONObject, key: str) -> str | None:
    """Read a key that must be present and either a string or null.

    Args:
        obj: The object holding the key.
        key: The key to read.

    Returns:
        The string, or None when the value is null.

    Raises:
        JSONTypeError: If the key is absent or holds any other type. Absence
            is rejected for the module-docstring reason: an entry nobody
            finished describing must not read as one with nothing to hold.
    """
    if key not in obj:
        raise JSONTypeError(
            f"entry must declare {key!r}, using null when the concept does not apply"
        )
    value = obj[key]
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise JSONTypeError(f"{key} must be a string or null, got {type(value).__name__}")


def decode_host_runner_spec(value: JSONValue) -> HostRunnerSpec:
    """Decode and validate one host's declaration.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated host.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, an optional concept is absent rather than null, or the
            memory floor is not positive.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"host must be a JSON object, got {type(value).__name__}")
    keepalive_task = _optional_null_str(value, "keepalive_task")
    if "wslconfig_min_memory_gb" not in value:
        raise JSONTypeError(
            "host must declare 'wslconfig_min_memory_gb', using null when the host "
            "default is acceptable"
        )
    raw_memory = value["wslconfig_min_memory_gb"]
    memory: int | None
    if raw_memory is None:
        memory = None
    elif isinstance(raw_memory, bool) or not isinstance(raw_memory, int):
        raise JSONTypeError(
            f"wslconfig_min_memory_gb must be an integer or null, got {type(raw_memory).__name__}"
        )
    else:
        if raw_memory <= 0:
            raise JSONTypeError(
                f"wslconfig_min_memory_gb must be positive, got {raw_memory}; a floor of "
                "nothing is the host default spelled confusingly"
            )
        memory = raw_memory
    installs = [decode_runner_install(entry) for entry in require_list(value, "installs")]
    if not installs:
        raise JSONTypeError(
            "installs must be non-empty; a host with no runner installs is not a CI host "
            "and does not belong in this roster"
        )
    return HostRunnerSpec(
        name=require_str(value, "name"),
        host=require_str(value, "host"),
        wsl_distro=require_str(value, "wsl_distro"),
        keepalive_task=keepalive_task,
        wslconfig_min_memory_gb=memory,
        scratch_dir=require_str(value, "scratch_dir"),
        gpu_required=require_bool(value, "gpu_required"),
        systemd_timers=require_str_list(value, "systemd_timers"),
        installs=installs,
        assets=[decode_file_asset(entry) for entry in require_list(value, "assets")],
    )


def encode_runner_spec(spec: RunnerSpec) -> JSONObject:
    """Encode the whole roster.

    Args:
        spec: The roster to encode.

    Returns:
        JSON-serialisable mapping carrying every host.
    """
    return {"hosts": [encode_host_runner_spec(host) for host in spec["hosts"]]}


def decode_runner_spec(value: JSONValue) -> RunnerSpec:
    """Decode and validate the whole roster.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated roster.

    Raises:
        JSONTypeError: If the value is not an object, ``hosts`` is missing
            or holds a non-object, two hosts share a name, or the roster is
            empty. An empty roster audits nothing and exits 0, which reads
            exactly like a healthy fleet -- the same trap ``fleet-nodes
            --probe never`` refuses one module over.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"runner spec must be a JSON object, got {type(value).__name__}")
    hosts = [decode_host_runner_spec(entry) for entry in require_list(value, "hosts")]
    if not hosts:
        raise JSONTypeError(
            "hosts must be non-empty; an empty roster audits nothing and exits 0, which "
            "reads exactly like a healthy fleet"
        )
    names = [host["name"] for host in hosts]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise JSONTypeError(
            f"host names must be unique, got duplicates: {', '.join(duplicates)}; two "
            "declarations for one machine would audit it twice and disagree"
        )
    return RunnerSpec(hosts=hosts)


__all__ = [
    "FileAsset",
    "HostRunnerSpec",
    "RunnerInstall",
    "RunnerSpec",
    "decode_file_asset",
    "decode_host_runner_spec",
    "decode_runner_install",
    "decode_runner_spec",
    "encode_file_asset",
    "encode_host_runner_spec",
    "encode_runner_install",
    "encode_runner_spec",
]
