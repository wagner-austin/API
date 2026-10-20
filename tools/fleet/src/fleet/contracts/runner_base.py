"""What a runner host needs BELOW its runners: the base a rebuild lays first.

The roster (:mod:`fleet.contracts.runners`) always said which runners a host
carries and which assets their jobs need. This block says what the machine
must hold beyond a fleet worker before any runner can register on it: the
packages only its CI jobs call (ffmpeg, go, poppler, ...) and the disk its
root may use. A runner host is a fleet machine installed from a CORVIS stick
or a netboot and provisioned by MCPs scripts/host/fleet-node/provision.sh,
which is the one source of its docker, GPU driver and toolchain, so none of
those is listed here (MCPs board task 1e9ab738, which retired the Windows
host and its WSL guest).

THE DISK CEILING LIVES HERE TOO. A runner host is meant to sit essentially
empty; the ceiling is the most its root may hold before the audit fails the
host, written beside the baseline measured on an idle rebuilt host, so a
failing row says both what is allowed and what normal looked like.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_dict,
    require_int,
    require_str,
    require_str_list,
)
from typing_extensions import TypedDict


class DiskCeiling(TypedDict):
    """How much the host's root may hold, and what idle looked like.

    Attributes:
        ceiling_gb: The most the host's root may use, in GB.
        baseline_gb: What an idle rebuilt host's root used, in GB.
        baseline_measured: The date the baseline was measured,
            ``YYYY-MM-DD``.
        cache_path: The runner account's cache directory, absolute: what
            pip, npm, poetry and playwright fill across jobs.
        cache_ceiling_gb: The most that directory may hold, in GB.
        work_ceiling_gb: The most each install's ``_work`` tree may hold,
            in GB.
    """

    ceiling_gb: int
    baseline_gb: int
    baseline_measured: str
    cache_path: str
    cache_ceiling_gb: int
    work_ceiling_gb: int


class HostBase(TypedDict):
    """Everything a runner host holds before its first runner registers.

    Attributes:
        apt_packages: Packages installed before any runner that a fleet
            worker does not already carry.
        disk: The root's disk ceiling and its idle baseline.
    """

    apt_packages: list[str]
    disk: DiskCeiling


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
        "cache_path": disk["cache_path"],
        "cache_ceiling_gb": disk["cache_ceiling_gb"],
        "work_ceiling_gb": disk["work_ceiling_gb"],
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
            ceiling, the date is not ``YYYY-MM-DD``, or the cache path is not
            absolute. A ceiling at or under the idle baseline fails a freshly
            rebuilt host, which is a roster that contradicts itself.
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
    cache_path = require_str(value, "cache_path")
    if not cache_path.startswith("/"):
        raise JSONTypeError(f"cache_path must be absolute, got {cache_path!r}")
    cache_ceiling = require_int(value, "cache_ceiling_gb")
    work_ceiling = require_int(value, "work_ceiling_gb")
    if cache_ceiling <= 0 or work_ceiling <= 0:
        raise JSONTypeError(
            f"cache_ceiling_gb and work_ceiling_gb must be positive, got {cache_ceiling} and "
            f"{work_ceiling}"
        )
    return DiskCeiling(
        ceiling_gb=ceiling,
        baseline_gb=baseline,
        baseline_measured=measured,
        cache_path=cache_path,
        cache_ceiling_gb=cache_ceiling,
        work_ceiling_gb=work_ceiling,
    )


def encode_host_base(base: HostBase) -> JSONObject:
    """Encode a host's base.

    Args:
        base: The base.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "apt_packages": list(base["apt_packages"]),
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
            mistyped, or no package is declared. The capability tags the
            runner jobs need (ffmpeg, go, pdftotext) are packages a fleet
            worker does not carry, so a base with none describes a host
            those jobs would fail on.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"host base must be a JSON object, got {type(value).__name__}")
    packages = require_str_list(value, "apt_packages")
    if not packages:
        raise JSONTypeError(
            "apt_packages must be non-empty; a fleet worker lacks the tagged tools runner jobs call"
        )
    return HostBase(
        apt_packages=packages,
        disk=decode_disk_ceiling(require_dict(value, "disk")),
    )


__all__ = [
    "DiskCeiling",
    "HostBase",
    "decode_disk_ceiling",
    "decode_host_base",
    "encode_disk_ceiling",
    "encode_host_base",
]
