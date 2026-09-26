"""What a runner host needs BELOW its runners: the base a rebuild lays first.

The roster (:mod:`fleet.contracts.runners`) always said which runners a host
carries and which assets their jobs need. What it never said is how a stock
Windows install becomes a machine those runners can live on: the Windows
features WSL needs, the WSL release, the distro image, the packages inside
it, the execution policy the Windows runner's PowerShell steps need, and the
PATH its jobs see. On 2026-09-26 all of that was done by hand on the
reinstalled lavender (board task 1aa6a021), and one missed step, the
execution policy Windows leaves at Restricted, failed every PowerShell step
on the Windows runner until it was found. This block is those steps written
down once, so ``fleet-runners --rebuild`` can lay them and the audit can hold
the host to them.

PINNED, NOT FLOATING. The WSL release and the distro image are each a URL
plus the SHA-256 of the bytes measured on the rebuild. A rebuild a month
later lays the same bits or stops at the digest, never a newer image that
nobody tested.

THE DISK CEILING LIVES HERE TOO. A runner host is meant to sit essentially
empty; the ceiling is the most its distro may hold before the audit fails
the host, written beside the baseline measured on an idle rebuilt host, so a
failing row says both what is allowed and what normal looked like.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_dict,
    require_int,
    require_list,
    require_str,
    require_str_list,
)
from typing_extensions import TypedDict

#: Execution policies a runner host may be held to. Restricted, the Windows
#: default, refuses every script a PowerShell step writes to ``_temp``;
#: RemoteSigned runs local scripts and is what the rebuilt lavender carries.
EXECUTION_POLICIES: tuple[str, ...] = ("RemoteSigned", "AllSigned")


class PinnedDownload(TypedDict):
    """One file the rebuild downloads and refuses unless its bytes match.

    Attributes:
        version: The human name of what the bytes are, e.g. ``2.7.14`` for
            the WSL release; printed by the rebuild and checked where the
            installed thing can report its own version.
        url: Where the bytes come from, ``https`` only.
        sha256: The 64-hex digest measured when the pin was taken.
    """

    version: str
    url: str
    sha256: str


class DiskCeiling(TypedDict):
    """How much a runner host's distro may hold before the audit fails it.

    Attributes:
        ceiling_gb: The most the distro's root may use, in GB.
        baseline_gb: What an idle rebuilt host's root used, in GB.
        baseline_measured: The date the baseline was measured,
            ``YYYY-MM-DD``.
    """

    ceiling_gb: int
    baseline_gb: int
    baseline_measured: str


class MachineVariable(TypedDict):
    """One machine-scope environment variable every runner service inherits.

    Attributes:
        name: The variable's name, a plain identifier. ``Path`` is refused:
            the machine PATH is ``machine_path_entries``, appended to rather
            than replaced.
        value: The exact value the variable must hold.
        reason: Why the host needs it, printed beside a drifted audit row.
    """

    name: str
    value: str
    reason: str


class HostBase(TypedDict):
    """Everything a stock Windows install needs before a runner can register.

    Attributes:
        windows_features: Optional features to enable, e.g.
            ``VirtualMachinePlatform``; enabling one asks for a reboot, which
            the rebuild performs and waits out.
        wsl_msi: The WSL release installed from its MSI.
        rootfs: The distro image imported as the host's ``wsl_distro``.
        distro_dir: Windows directory the distro's disk is created in.
        apt_packages: Packages installed inside the distro before any runner.
        machine_path_entries: Directories appended to the machine PATH, which
            every Windows runner service inherits.
        machine_environment: Machine-scope variables every Windows runner
            service inherits, distinct by name.
        execution_policy: The LocalMachine execution policy, one of
            :data:`EXECUTION_POLICIES`.
        disk: The distro's disk ceiling and its idle baseline.
    """

    windows_features: list[str]
    wsl_msi: PinnedDownload
    rootfs: PinnedDownload
    distro_dir: str
    apt_packages: list[str]
    machine_path_entries: list[str]
    machine_environment: list[MachineVariable]
    execution_policy: str
    disk: DiskCeiling


def encode_pinned_download(pin: PinnedDownload) -> JSONObject:
    """Encode one pinned download.

    Args:
        pin: The download.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {"version": pin["version"], "url": pin["url"], "sha256": pin["sha256"]}


def decode_pinned_download(value: JSONValue) -> PinnedDownload:
    """Decode and validate one pinned download.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated download.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the URL is not ``https``, or the digest is not 64
            lowercase hex characters. A malformed digest matches no bytes and
            would stop every rebuild at a check nobody could satisfy.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"pinned download must be a JSON object, got {type(value).__name__}")
    url = require_str(value, "url")
    if not url.startswith("https://"):
        raise JSONTypeError(f"a pinned download's url must be https, got {url!r}")
    digest = require_str(value, "sha256")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise JSONTypeError(f"sha256 must be 64 lowercase hex characters, got {digest!r}")
    return PinnedDownload(version=require_str(value, "version"), url=url, sha256=digest)


def encode_disk_ceiling(disk: DiskCeiling) -> JSONObject:
    """Encode a disk ceiling.

    Args:
        disk: The ceiling.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "ceiling_gb": disk["ceiling_gb"],
        "baseline_gb": disk["baseline_gb"],
        "baseline_measured": disk["baseline_measured"],
    }


def decode_disk_ceiling(value: JSONValue) -> DiskCeiling:
    """Decode and validate a disk ceiling.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated ceiling.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, a size is not positive, the baseline is not below the
            ceiling, or the date is not ``YYYY-MM-DD``. A ceiling at or under
            the idle baseline fails a freshly rebuilt host, which is a roster
            that contradicts itself.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"disk ceiling must be a JSON object, got {type(value).__name__}")
    ceiling = require_int(value, "ceiling_gb")
    baseline = require_int(value, "baseline_gb")
    if baseline <= 0 or ceiling <= 0:
        raise JSONTypeError(
            f"ceiling_gb and baseline_gb must be positive, got {ceiling} and {baseline}"
        )
    if baseline >= ceiling:
        raise JSONTypeError(
            f"baseline_gb {baseline} must be below ceiling_gb {ceiling}; a ceiling at or "
            "under the idle baseline fails a freshly rebuilt host"
        )
    measured = require_str(value, "baseline_measured")
    parts = measured.split("-")
    if [len(part) for part in parts] != [4, 2, 2] or not all(part.isdigit() for part in parts):
        raise JSONTypeError(f"baseline_measured must be YYYY-MM-DD, got {measured!r}")
    return DiskCeiling(ceiling_gb=ceiling, baseline_gb=baseline, baseline_measured=measured)


def encode_machine_variable(variable: MachineVariable) -> JSONObject:
    """Encode one machine variable.

    Args:
        variable: The variable.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {"name": variable["name"], "value": variable["value"], "reason": variable["reason"]}


def decode_machine_variable(value: JSONValue) -> MachineVariable:
    """Decode and validate one machine variable.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated variable.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the name is not a plain identifier or is ``Path``, or
            the value or reason is empty. An empty value would delete the
            variable the row exists to hold.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"machine variable must be a JSON object, got {type(value).__name__}")
    name = require_str(value, "name")
    if not (name[:1].isalpha() or name[:1] == "_") or not all(
        c.isalnum() or c == "_" for c in name
    ):
        raise JSONTypeError(f"a machine variable's name must be a plain identifier, got {name!r}")
    if name.lower() == "path":
        raise JSONTypeError("the machine PATH is machine_path_entries, not a machine variable")
    held = require_str(value, "value")
    reason = require_str(value, "reason")
    if not held or not reason:
        raise JSONTypeError(f"machine variable {name} needs a non-empty value and reason")
    return MachineVariable(name=name, value=held, reason=reason)


def _decode_machine_environment(value: JSONObject) -> list[MachineVariable]:
    """Decode the base's machine variables, refusing a name declared twice.

    Args:
        value: The base's JSON object.

    Returns:
        The variables, in declaration order.

    Raises:
        JSONTypeError: If the field is missing or not a list, an entry is invalid, or
            two entries share a name, compared case-insensitively as Windows
            compares them. Two values for one variable make the base rewrite
            it on every run and the audit fail one of the rows forever.
    """
    variables = [
        decode_machine_variable(entry) for entry in require_list(value, "machine_environment")
    ]
    names = [variable["name"].lower() for variable in variables]
    if len(set(names)) != len(names):
        raise JSONTypeError(f"machine_environment declares a name twice: {names}")
    return variables


def encode_host_base(base: HostBase) -> JSONObject:
    """Encode a host's base.

    Args:
        base: The base.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "windows_features": list(base["windows_features"]),
        "wsl_msi": encode_pinned_download(base["wsl_msi"]),
        "rootfs": encode_pinned_download(base["rootfs"]),
        "distro_dir": base["distro_dir"],
        "apt_packages": list(base["apt_packages"]),
        "machine_path_entries": list(base["machine_path_entries"]),
        "machine_environment": [
            encode_machine_variable(variable) for variable in base["machine_environment"]
        ],
        "execution_policy": base["execution_policy"],
        "disk": encode_disk_ceiling(base["disk"]),
    }


def decode_host_base(value: JSONValue) -> HostBase:
    """Decode and validate a host's base.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated base.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, ``distro_dir`` is not a drive-letter path, no package
            is declared, or the execution policy is not one of
            :data:`EXECUTION_POLICIES`. A distro with no packages cannot run
            docker, and every Linux runner job here needs it.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"host base must be a JSON object, got {type(value).__name__}")
    distro_dir = require_str(value, "distro_dir")
    if not (len(distro_dir) > 2 and distro_dir[0].isalpha() and distro_dir[1:3] == ":/"):
        raise JSONTypeError(
            f"distro_dir must be a forward-slashed drive-letter path, got {distro_dir!r}"
        )
    packages = require_str_list(value, "apt_packages")
    if not packages:
        raise JSONTypeError("apt_packages must be non-empty; a bare distro cannot run docker")
    policy = require_str(value, "execution_policy")
    if policy not in EXECUTION_POLICIES:
        raise JSONTypeError(
            f"execution_policy must be one of {', '.join(EXECUTION_POLICIES)}, got {policy!r}"
        )
    return HostBase(
        windows_features=require_str_list(value, "windows_features"),
        wsl_msi=decode_pinned_download(require_dict(value, "wsl_msi")),
        rootfs=decode_pinned_download(require_dict(value, "rootfs")),
        distro_dir=distro_dir,
        apt_packages=packages,
        machine_path_entries=require_str_list(value, "machine_path_entries"),
        machine_environment=_decode_machine_environment(value),
        execution_policy=policy,
        disk=decode_disk_ceiling(require_dict(value, "disk")),
    )


__all__ = [
    "EXECUTION_POLICIES",
    "DiskCeiling",
    "HostBase",
    "MachineVariable",
    "PinnedDownload",
    "decode_disk_ceiling",
    "decode_host_base",
    "decode_machine_variable",
    "decode_pinned_download",
    "encode_disk_ceiling",
    "encode_host_base",
    "encode_machine_variable",
    "encode_pinned_download",
]
