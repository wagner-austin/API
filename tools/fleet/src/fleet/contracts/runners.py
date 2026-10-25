"""The CI runner roster: which runner installs each machine carries, and why.

This document exists because on 2026-09-08 every fact in it was folklore:
the licensed game tree, the llama.cpp checkout and ``/data`` had each been
placed on lavender by hand, findable only by reading a workflow's assert step
or a board post. ``fleet-runners audit`` scores a host against this roster;
``fleet-runners render`` emits the converge script for a new one.

A RUNNER HOST IS A LINUX MACHINE. Until 2026-10-19 lavender was a Windows
machine whose WSL distro held the Linux runners beside four Windows ones;
it was reimaged to Ubuntu that day and every Windows install was deleted
(MCPs board task 1e9ab738), so each runner is a systemd service on the host
itself, reached over ssh like every other fleet node. loki's four Windows
runners, the last ones, left the roster with loki on 2026-10-25 (board task
e705f08c).

THIS IS A SECOND MACHINE DOCUMENT, NOT A COPY OF THE FIRST. The fleet's
roster, ``registry/machines`` (:mod:`fleet.contracts.machine`), owns which
machines exist, whether one should answer and what it can run for DISPATCH.
This roster owns what a machine serves to GITHUB ACTIONS: runner installs,
their labels, and the host assets their job classes require, none of which
the fleet's roster declares.

ABSENCE IS NOT A STATE, same rule as :mod:`fleet.contracts.node`: optional
concepts are spelled as explicit ``null`` rather than omitted keys, so an
asset nobody finished describing cannot be told apart from one that
genuinely has no pin.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_bool,
    require_dict,
    require_int,
    require_list,
    require_str,
    require_str_list,
)
from typing_extensions import TypedDict

from fleet.contracts.runner_base import HostBase, decode_host_base, encode_host_base
from fleet.contracts.runner_registration import (
    RunnerRegistration,
    decode_runner_registration,
    encode_runner_registration,
)
from fleet.contracts.runner_slice import CiSlice, decode_ci_slice, encode_ci_slice


class RunnerInstall(RunnerRegistration):
    """One GitHub Actions runner install on a Linux host.

    Its registration (:class:`fleet.contracts.runner_registration.RunnerRegistration`)
    names the systemd unit that runs it, ``.service`` suffix included.

    Attributes:
        workdir: Absolute POSIX path of the install's ``_work`` tree. This
            is the path a poetry in-project venv bakes into its shebangs --
            the fact behind the 2026-09-08 cache poisoning -- and the audit
            asserts it exists.
    """

    workdir: str


class FileAsset(TypedDict):
    """One path a host must hold for its runner jobs to be honest.

    Attributes:
        path: Absolute path on the host.
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
        name: The host's fleet name, a machine of ``registry/machines``, so
            the two documents can be joined.
        host: SSH destination -- an alias from ``~/.ssh/config``, the same
            rule as :class:`fleet.contracts.node.NodeConfig`. Its account
            runs root payloads through ``sudo -n``.
        memory_gb: The memory the host's kernel reports, in whole GB. The
            CI slice's ceiling is held below it, so the runners can never be
            granted the memory sshd and the fleet's own checks need.
        payload_dir: Absolute POSIX directory on the host the root payloads
            and the audit script are sent to, the same role ``stage_root``
            plays for dispatch. Declared here rather than joined from the
            registry at runtime: an audit that needed a second document to
            locate a writable directory would fail on exactly the machine
            the second document forgot.
        gpu_required: Whether runner jobs on this host digest a real GPU
            (Model-Trainer's default hooks do). Audited via ``nvidia-smi``
            on each runner's own PATH.
        systemd_timers: Timers that must be enabled on the host, e.g.
            ``ci-clean.timer`` -- hygiene is part of the provision, not an
            afterthought on one box's disk.
        job_timeout_minutes: The longest ``timeout-minutes`` any job these
            runners take may declare. A runner holding a process older than
            this with no ``Runner.Worker`` alive is holding what a finished
            job left (MCPs board task 53528106), which the audit fails.
        installs: Every runner install this host carries.
        assets: Every path its job classes require.
        base: What the host must hold before any runner can register, and
            its root's disk ceiling (:mod:`fleet.contracts.runner_base`).
        ci_slice: The memory and CPU the host's runners may take together,
            beside the fleet's own lane on the same machine
            (:mod:`fleet.contracts.runner_slice`).
    """

    name: str
    host: str
    memory_gb: int
    payload_dir: str
    gpu_required: bool
    systemd_timers: list[str]
    job_timeout_minutes: int
    installs: list[RunnerInstall]
    assets: list[FileAsset]
    base: HostBase
    ci_slice: CiSlice


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
    return {**encode_runner_registration(install), "workdir": install["workdir"]}


def decode_runner_install(value: JSONValue) -> RunnerInstall:
    """Decode and validate one runner install.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated install.

    Raises:
        JSONTypeError: If the value is not an object, its registration is
            refused (:func:`fleet.contracts.runner_registration.decode_runner_registration`),
            or ``workdir`` is not an absolute POSIX path.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"runner install must be a JSON object, got {type(value).__name__}")
    registration = decode_runner_registration(value)
    workdir = require_str(value, "workdir")
    if not workdir.startswith("/"):
        raise JSONTypeError(f"an install's workdir must be an absolute POSIX path, got {workdir!r}")
    return RunnerInstall(
        repo=registration["repo"],
        runner_name=registration["runner_name"],
        service=registration["service"],
        labels=registration["labels"],
        workdir=workdir,
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
        "memory_gb": spec["memory_gb"],
        "payload_dir": spec["payload_dir"],
        "gpu_required": spec["gpu_required"],
        "systemd_timers": list(spec["systemd_timers"]),
        "job_timeout_minutes": spec["job_timeout_minutes"],
        "installs": [encode_runner_install(install) for install in spec["installs"]],
        "assets": [encode_file_asset(asset) for asset in spec["assets"]],
        "base": encode_host_base(spec["base"]),
        "ci_slice": encode_ci_slice(spec["ci_slice"]),
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
            mistyped, the memory or the job timeout is not positive, or the
            payload directory is not an absolute POSIX path.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"host must be a JSON object, got {type(value).__name__}")
    memory = require_int(value, "memory_gb")
    if memory <= 0:
        raise JSONTypeError(f"memory_gb must be positive, got {memory}")
    payload_dir = require_str(value, "payload_dir")
    if not payload_dir.startswith("/"):
        raise JSONTypeError(f"payload_dir must be an absolute POSIX path, got {payload_dir!r}")
    job_timeout = require_int(value, "job_timeout_minutes")
    if job_timeout <= 0:
        raise JSONTypeError(
            f"job_timeout_minutes must be positive, got {job_timeout}; every job a runner "
            "takes is bounded, and the audit measures a finished job's leftovers against it"
        )
    installs = [decode_runner_install(entry) for entry in require_list(value, "installs")]
    if not installs:
        raise JSONTypeError(
            "installs must be non-empty; a host with no runner installs is not a CI host "
            "and does not belong in this roster"
        )
    return HostRunnerSpec(
        name=require_str(value, "name"),
        host=require_str(value, "host"),
        memory_gb=memory,
        payload_dir=payload_dir,
        gpu_required=require_bool(value, "gpu_required"),
        systemd_timers=require_str_list(value, "systemd_timers"),
        job_timeout_minutes=job_timeout,
        installs=installs,
        assets=[decode_file_asset(entry) for entry in require_list(value, "assets")],
        base=decode_host_base(require_dict(value, "base")),
        ci_slice=decode_ci_slice(require_dict(value, "ci_slice"), host_memory_gb=memory),
    )


def encode_runner_spec(spec: RunnerSpec) -> JSONObject:
    """Encode the whole roster.

    Args:
        spec: The roster to encode.

    Returns:
        JSON-serialisable mapping carrying every host.
    """
    return {
        "hosts": [encode_host_runner_spec(host) for host in spec["hosts"]],
    }


def decode_runner_spec(value: JSONValue) -> RunnerSpec:
    """Decode and validate the whole roster.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated roster.

    Raises:
        JSONTypeError: If the value is not an object, ``hosts`` is missing
            or holds a non-object, two hosts share a name, or ``hosts`` is
            empty. An empty roster audits nothing and exits 0, which reads
            exactly like a healthy fleet.
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


def parse_runner_spec(raw: str, *, source: str) -> RunnerSpec:
    """Read a roster's text, the one reader of the checkout's file and the rolled one.

    Args:
        raw: The text of ``runners.json``.
        source: Where it was read, for the refusal.

    Returns:
        The validated roster.

    Raises:
        AppError: ``RUNNER_SPEC_UNREADABLE`` when the text does not decode as
            a roster.
    """
    try:
        return decode_runner_spec(load_json_str(raw))
    except JSONTypeError as fault:
        raise AppError(
            FleetErrorCode.RUNNER_SPEC_UNREADABLE,
            f"the runner roster at {source} cannot be read: {fault}",
        ) from fault


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
    "parse_runner_spec",
]
