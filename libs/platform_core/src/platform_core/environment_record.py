"""Which machine a run used, and which library versions it resolved to.

The axes :mod:`platform_core.comparability` was missing. That module fixes the
software axis with an image digest and the hardware axis with a card and a
driver -- and for research that pulls no torch, both of those are
:const:`~platform_core.comparability.NO_VALUE`. A gradient-boosting run out of
a directory environment therefore recorded NOTHING that identified the machine
or the libraries, so two runs on two different boxes produced byte-identical
fingerprints and :func:`~platform_core.comparability.find_differences`
reported no difference between them.

That is not a hypothetical. :mod:`platform_core.determinism_cpu` states in its
own docstring that different microarchitectures select different BLAS kernels
and that an AVX-512 wheel does not compute the same partial sums as an AVX2
one -- and then the fingerprint had no field in which to say which one ran.
The pinned thread COUNT was recorded; the machine it was pinned on was not.

TWO AXES RATHER THAN ONE, and the reason is calibration. A
:class:`~platform_core.comparability.Calibration` spans one axis between two
rendered values. Folding the libraries into the host would mean a numpy bump
on an unchanged box read as a host change, and the calibration measured for it
would name a value that includes the CPU -- so it would not apply to the same
bump on any other machine. Separating them keeps each measurement reusable.

WHAT THE STDLIB PROBE CAN AND CANNOT DISTINGUISH, stated because an axis whose
resolution is unstated will be trusted past it. :class:`HostRecord` carries the
operating system and build, the instruction-set architecture, and the logical
core count. It separates Windows from Linux, x86_64 from arm64, and an 8-core
box from a 24-core one. **It does not separate two x86_64 Linux machines of
different CPU model**, because no stdlib call reports the model portably and
this module will not grow a per-OS chain of readers to guess at one. A caller
that knows its machine -- a scheduler that knows the node type, a launcher
that was told -- injects its own :class:`HostProbe` and gets a sharper axis;
:func:`stdlib_host_probe` is the floor, not the ceiling.
"""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Mapping
from typing import Protocol

from typing_extensions import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_str,
)


class HostRecord(TypedDict):
    """The machine a run's numbers were computed on.

    Attributes:
        platform: The operating system, release and build, as
            ``platform.platform()`` reports it, e.g.
            ``"Windows-10-10.0.26200-SP0"``. Non-empty by construction.
        machine: The instruction-set architecture, as ``platform.machine()``
            reports it, e.g. ``"AMD64"`` or ``"x86_64"``. Recorded separately
            from the platform string because a numeric wheel is built per
            architecture, and that is the axis that decides which kernels the
            run had available. Non-empty by construction.
        logical_cores: How many logical processors the machine has. Recorded
            even when a run pins its thread count to one, because the pin is
            a SETTING and this is the machine: a run that inherited a default
            on a 24-core box and one that inherited it on an 8-core box
            differ here and nowhere else.
    """

    platform: str
    machine: str
    logical_cores: int


class PackageVersion(TypedDict):
    """One installed distribution and the version that resolved.

    Attributes:
        name: The distribution name as installed, e.g. ``"numpy"``.
        version: The resolved version, e.g. ``"2.3.5"``. What was INSTALLED,
            never what was requested: a lock file records intent, and the two
            have disagreed on this project's own published arms.
    """

    name: str
    version: str


class HostProbe(Protocol):
    """The host facts a fingerprint records.

    A protocol rather than direct stdlib calls so a test can state a machine
    without owning one, and so a caller who can identify its CPU model more
    sharply than the stdlib does can say so.
    """

    def platform(self) -> str:
        """Return the operating system, release and build.

        Returns:
            The platform string.
        """
        ...

    def machine(self) -> str:
        """Return the instruction-set architecture.

        Returns:
            The architecture string.
        """
        ...

    def logical_cores(self) -> int:
        """Return the machine's logical processor count.

        Returns:
            The count, one or greater.
        """
        ...


class VersionReader(Protocol):
    """Reads the resolved version of one installed distribution."""

    def __call__(self, distribution: str) -> str:
        """Return the installed version of a distribution.

        Args:
            distribution: The distribution name.

        Returns:
            Its resolved version.
        """
        ...


class CoreCounter(Protocol):
    """Reports how many logical processors the machine has, if it knows.

    The signature is ``os.cpu_count``'s, including its ``None``: a protocol
    that promised an ``int`` would oblige every implementation to invent one
    on the platforms that do not report a count.
    """

    def __call__(self) -> int | None:
        """Return the logical processor count, or None when unknown.

        Returns:
            The count, or None.
        """
        ...


class UnknownCoreCountError(RuntimeError):
    """The machine did not report how many logical processors it has.

    Raised rather than recorded as a zero or a guess. A core count is an input
    to every threaded reduction the run performs; a fingerprint that invented
    one would compare equal to a machine that really had it.
    """


class _StdlibHostProbe:
    """Reads host facts from the standard library.

    The real implementation, wired by production callers through
    :func:`stdlib_host_probe`. Its resolution is the module docstring's:
    operating system, architecture and core count, and not CPU model.

    ``platform.platform`` and ``platform.machine`` are called directly because
    they are pure reads that cannot fail. The core count is injected because
    it CAN come back unknown, and the arm that rejects that is one this
    library must be able to exercise without owning a machine that has it.
    """

    def __init__(self, count_cores: CoreCounter) -> None:
        """Store the core-count reader.

        Args:
            count_cores: Reader for the logical processor count.
        """
        self._count_cores = count_cores

    def platform(self) -> str:
        """Return the operating system, release and build.

        Returns:
            ``platform.platform()``.
        """
        return platform.platform()

    def machine(self) -> str:
        """Return the instruction-set architecture.

        Returns:
            ``platform.machine()``.
        """
        return platform.machine()

    def logical_cores(self) -> int:
        """Return the machine's logical processor count.

        Returns:
            The count the reader gave.

        Raises:
            UnknownCoreCountError: When the reader does not know the count.
        """
        count = self._count_cores()
        if count is None:
            raise UnknownCoreCountError("this platform does not report a logical processor count")
        return count


def stdlib_host_probe(count_cores: CoreCounter) -> HostProbe:
    """Build the probe that reads host facts from the standard library.

    Args:
        count_cores: Reader for the logical processor count. Production passes
            ``os.cpu_count``.

    Returns:
        The probe. Annotated as :class:`HostProbe` at the assignment so the
        protocol, not the concrete class, is what callers depend on.
    """
    probe: HostProbe = _StdlibHostProbe(count_cores)
    return probe


def installed_version(distribution: str) -> str:
    """Read one distribution's resolved version from the installed metadata.

    The real :class:`VersionReader`, wired by production callers.

    Args:
        distribution: The distribution name.

    Returns:
        Its installed version.

    Raises:
        importlib.metadata.PackageNotFoundError: When the distribution is not
            installed. Propagated: a caller asking to fingerprint a library
            the run does not have is naming the wrong library, and recording
            an absence as an empty version would make that unreadable later.
    """
    return importlib.metadata.version(distribution)


def host_record(*, platform: str, machine: str, logical_cores: int) -> HostRecord:
    """Build a host record, rejecting values that cannot identify a machine.

    Args:
        platform: The operating system, release and build.
        machine: The instruction-set architecture.
        logical_cores: The logical processor count.

    Returns:
        The record.

    Raises:
        ValueError: When ``platform`` or ``machine`` is empty, or
            ``logical_cores`` is below one. Every run has a machine, so
            unlike an image digest there is no honest "unknown" here -- an
            empty axis would be a capture failure recorded as a fact.
    """
    if platform == "":
        raise ValueError("platform must name the operating system this run used")
    if machine == "":
        raise ValueError("machine must name the instruction-set architecture this run used")
    if logical_cores < 1:
        raise ValueError(f"logical_cores must be at least one, got {logical_cores}")
    return HostRecord(platform=platform, machine=machine, logical_cores=logical_cores)


def capture_host_record(probe: HostProbe) -> HostRecord:
    """Read the machine this process is running on.

    Args:
        probe: Reader for the host facts.

    Returns:
        The record.

    Raises:
        ValueError: When the probe reports a value that cannot identify a
            machine, per :func:`host_record`.
        UnknownCoreCountError: Propagated from :meth:`HostProbe.logical_cores`.
    """
    return host_record(
        platform=probe.platform(),
        machine=probe.machine(),
        logical_cores=probe.logical_cores(),
    )


def _package_name(package: PackageVersion) -> str:
    """Sort key putting packages in name order.

    A named function rather than a lambda, because a lambda's parameter infers
    as ``Any`` under this repo's mypy settings and the sort key would then be
    unchecked.

    Args:
        package: The package to key.

    Returns:
        Its distribution name.
    """
    return package["name"]


def package_versions(versions: Mapping[str, str]) -> tuple[PackageVersion, ...]:
    """Build the package axis, putting the entries in canonical order.

    Args:
        versions: Resolved version by distribution name.

    Returns:
        The entries sorted by name, so two runs of the same environment render
        byte-identically whatever order their producer emitted them in.

    Raises:
        ValueError: When a name or a version is empty. An entry that cannot
            say which library, or which version of it, adds an axis value that
            differs from every real one without saying why.
    """
    entries: list[PackageVersion] = []
    for name, version in versions.items():
        if name == "":
            raise ValueError("a package entry must name a distribution")
        if version == "":
            raise ValueError(f"package {name!r} must carry the version that resolved")
        entries.append(PackageVersion(name=name, version=version))
    return tuple(sorted(entries, key=_package_name))


def capture_package_versions(
    distributions: tuple[str, ...], read_version: VersionReader
) -> tuple[PackageVersion, ...]:
    """Read the resolved versions of the libraries a run's numbers depend on.

    WHICH DISTRIBUTIONS IS THE CALLER'S, deliberately. A fingerprint that
    recorded every installed package would differ between two runs over a
    dev-dependency bump that cannot reach the arithmetic, and every such
    difference makes a genuine one harder to see. The caller names the
    libraries whose version decides its numbers.

    Args:
        distributions: The distribution names to record.
        read_version: Reader for one distribution's installed version.

    Returns:
        The entries, sorted by name.

    Raises:
        ValueError: When ``distributions`` is empty, when it names one twice,
            or when a name or resolved version is empty. An empty set is a
            caller mistake rather than a run with no libraries: it would
            record "nothing decides these numbers", which is never true of a
            numeric run.
        importlib.metadata.PackageNotFoundError: Propagated from the reader
            when a named distribution is not installed.
    """
    if distributions == ():
        raise ValueError("name the distributions whose versions decide this run's numbers")
    duplicated = sorted({n for n in distributions if distributions.count(n) > 1})
    if duplicated:
        raise ValueError(f"distributions must be distinct; repeated: {duplicated}")
    return package_versions({name: read_version(name) for name in distributions})


def encode_host_record(record: HostRecord) -> JSONObject:
    """Encode a host record for a run record.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying the platform, the architecture and the core
        count.
    """
    return {
        "platform": record["platform"],
        "machine": record["machine"],
        "logical_cores": record["logical_cores"],
    }


def decode_host_record(value: JSONValue) -> HostRecord:
    """Validate a JSON value as a host record.

    Args:
        value: The value to validate, typically from a stored run record.

    Returns:
        The validated record.

    Raises:
        JSONTypeError: When ``value`` is not an object, or a field is absent
            or mistyped.
        ValueError: When a decoded field cannot identify a machine, per
            :func:`host_record`.
    """
    obj = narrow_json_to_dict(value)
    return host_record(
        platform=require_str(obj, "platform"),
        machine=require_str(obj, "machine"),
        logical_cores=require_int(obj, "logical_cores"),
    )


def encode_package_versions(packages: tuple[PackageVersion, ...]) -> list[JSONValue]:
    """Encode the package axis for a run record.

    Args:
        packages: The entries to encode, in canonical order.

    Returns:
        A JSON list of ``{"name": ..., "version": ...}`` objects. A list of
        pairs rather than a name-to-version object so the encoded order is the
        canonical one and two equal axes serialise to identical bytes.
    """
    return [{"name": p["name"], "version": p["version"]} for p in packages]


def decode_package_versions(value: JSONValue) -> tuple[PackageVersion, ...]:
    """Validate a JSON value as the package axis.

    Args:
        value: The value to validate, typically from a stored run record.

    Returns:
        The validated entries, in canonical order.

    Raises:
        JSONTypeError: When ``value`` is not a list, or an entry is not an
            object with a string name and a string version.
        ValueError: When an entry carries an empty name or version, per
            :func:`package_versions`.
    """
    if not isinstance(value, list):
        raise JSONTypeError(f"Package versions must be a list, got {type(value).__name__}")
    versions: dict[str, str] = {}
    for entry in value:
        item = narrow_json_to_dict(entry)
        name = require_str(item, "name")
        if name in versions:
            raise JSONTypeError(f"Package {name!r} appears twice")
        versions[name] = require_str(item, "version")
    return package_versions(versions)


def render_host_record(record: HostRecord) -> str:
    """Render a host record as one stable comparison key.

    Args:
        record: The record to render.

    Returns:
        The platform, architecture and core count in a fixed order, so two
        runs on the same machine render byte-identically and a difference is
        legible without reading two nested objects side by side.
    """
    return f"{record['platform']}/{record['machine']}/{record['logical_cores']}"


def render_package_versions(packages: tuple[PackageVersion, ...]) -> str:
    """Render the package axis as one stable comparison key.

    Args:
        packages: The entries, in canonical order.

    Returns:
        The name-version pairs joined in canonical order.
    """
    return ",".join(f"{p['name']}={p['version']}" for p in packages)


__all__ = [
    "CoreCounter",
    "HostProbe",
    "HostRecord",
    "PackageVersion",
    "UnknownCoreCountError",
    "VersionReader",
    "capture_host_record",
    "capture_package_versions",
    "decode_host_record",
    "decode_package_versions",
    "encode_host_record",
    "encode_package_versions",
    "host_record",
    "installed_version",
    "package_versions",
    "render_host_record",
    "render_package_versions",
    "stdlib_host_probe",
]
