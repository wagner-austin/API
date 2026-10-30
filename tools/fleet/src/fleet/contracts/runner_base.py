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

THE CEILING ADMITS WHAT THE HOST'S OWN BOUNDS ALLOW (board task 621902c3).
lavender's 150 GB was written on 2026-09-26 for a Windows host whose
runners alone used its disk, and it was carried over to the Linux fleet
worker of 2026-10-19 while the bounds beside it allowed 18 GB of baseline,
60 GiB of build cache, a 60 GB runner cache and 15 GB for each of its
installs' work trees: the audit failed the host at 190 GB with every one of
those within its bound. So a host's ceiling is at least
:func:`disk_bounds_gb`, and a roster whose ceiling sits under it is
refused where the host decodes (:func:`fleet.contracts.runners.decode_host_runner_spec`).
"""

from __future__ import annotations

from typing import Final

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

#: The build cache a runner host's ci-clean leaves on its docker daemon, in
#: docker's ``gb``, which counts 1024**3 bytes (:mod:`fleet.core.ci_clean`).
#: It was 60, the bound diphtheria's stack prunes to; on 2026-10-30 lavender
#: held 39 GB of build cache in 1171 records, none in use and 36.44 GB of it
#: reclaimable, inside that bound and over a fifth of its root's ceiling. At
#: 20 the root still read 155 GB at 2026-10-30T20:02Z with 21.31 GB of build
#: cache, none in use, against the 150 GB the operator asked for ('Get it
#: under 150', MCPs board task 621902c3).
BUILD_CACHE_BOUND_GIB: Final = 10


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


def disk_bounds_gb(disk: DiskCeiling, installs: int) -> int:
    """What a runner host's root may hold with every bounded area full.

    Args:
        disk: The host's ceiling and the bounds beside it.
        installs: How many runner installs the host carries, each with a
            work tree held to ``work_ceiling_gb``.

    Returns:
        The idle baseline, plus :data:`BUILD_CACHE_BOUND_GIB`, the runner
        cache's ceiling and one work ceiling per install, in GB.
    """
    return (
        disk["baseline_gb"]
        + BUILD_CACHE_BOUND_GIB
        + disk["cache_ceiling_gb"]
        + disk["work_ceiling_gb"] * installs
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
    "BUILD_CACHE_BOUND_GIB",
    "DiskCeiling",
    "HostBase",
    "decode_disk_ceiling",
    "decode_host_base",
    "disk_bounds_gb",
    "encode_disk_ceiling",
    "encode_host_base",
]
