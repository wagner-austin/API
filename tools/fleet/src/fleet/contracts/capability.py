"""Toolchains a node declares as versions and its probe re-measures every tick.

WHY A NODE DECLARES THEM AT ALL. Some projects build native code during
their install, and a node without the compiler fails there, after staging
the whole tree and before any test runs:

- ``rust`` (MCPs board task 1e2da299). API ``services/covenant-radar-api``
  depends on the maturin crate ``libs/cleargbm_rs`` as a path dependency
  without ``develop = true``, so ``poetry sync`` builds it through ``cargo``.
  On 2026-09-26 no fleet node had cargo.
- ``cxx`` (MCPs board task 3f19c136). Every MCPs TypeScript project installs
  with a root ``npm ci``, which rebuilds ``hnswlib-node`` under node-gyp, and
  node-gyp needs a C++ toolchain: the Visual Studio VC tools on Windows, g++
  on Linux. Measured 2026-09-27, no Windows node had the VC tools; dispatch
  job 31fd1548 on lavender and 69f696cb on serendipity died in
  ``find-visualstudio.js``.

A project that needs one requires the tag of the same name
(:mod:`fleet.contracts.tags`), and a node carries it when its declaration
names a version.

WHY THE DECLARATION IS A VERSION AND IS RE-MEASURED. A boolean would be the
hand-written tag the tags module exists to avoid, true until somebody
remembered to change it. The declaration is instead the exact version the
node's probe reported, and the runner's toolchain probe asks again every
tick; a node whose answer differs from its declaration claims nothing
(:func:`fleet.core.toolchain.readiness_gap`), so the tag is exactly as true
as the last measurement. A node that has the toolchain and declares none only
withholds the tag, which misroutes nothing, so that direction is reported by
the ready summary rather than refused.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from platform_core.errors import FleetErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.toolchain import ToolReport


class Capability(StrEnum):
    """A toolchain a node declares by version. Each value is also its tag."""

    RUST = "rust"
    CXX = "cxx"


#: The probe line each capability is read from. ``cargo`` is the tool's own
#: name because maturin builds with it; ``cxx`` is a synthetic line, since the
#: toolchain node-gyp finds is not one executable on both platforms (vswhere's
#: VC tools component on Windows, ``g++ -dumpfullversion`` on Linux).
PROBE_NAME: Final[dict[Capability, str]] = {Capability.RUST: "cargo", Capability.CXX: "cxx"}

#: What refuses a node whose declaration disagrees with its probe.
MISMATCH_CODE: Final[dict[Capability, FleetErrorCode]] = {
    Capability.RUST: FleetErrorCode.NODE_RUST_MISMATCH,
    Capability.CXX: FleetErrorCode.NODE_CXX_MISMATCH,
}

#: Where a declared version is read, for the decode refusal.
_SOURCE: Final[dict[Capability, str]] = {
    Capability.RUST: "cargo --version prints, e.g. '1.98.1'",
    Capability.CXX: (
        "vswhere reports for the VC tools on Windows or g++ -dumpfullversion prints on Linux, "
        "e.g. '13.3.0'"
    ),
}

#: What goes wrong when a node carries a tag it cannot honour.
_CONSEQUENCE: Final[dict[Capability, str]] = {
    Capability.RUST: "a crate build claimed on it would fail in poetry sync",
    Capability.CXX: "an npm ci claimed on it would fail rebuilding a native module under node-gyp",
}

#: A declared version: two to four dotted numbers, the shapes cargo (1.98.1),
#: g++ (13.3.0) and vswhere (17.14.36310.24) print.
VERSION: Final = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}")


def _version_in(capability: Capability, answer: str) -> str | None:
    """Pull the version out of one probe answer.

    Args:
        capability: Which toolchain answered.
        answer: What the probe line carried.

    Returns:
        The version, or None when the answer is not of the expected shape.
        cargo's is its SECOND word, because the last is its build date;
        the ``cxx`` line carries the bare version.
    """
    words = answer.split()
    if capability is Capability.RUST:
        candidate = words[1] if len(words) >= 2 and words[0] == "cargo" else ""
    else:
        candidate = answer.strip()
    return candidate if VERSION.fullmatch(candidate) else None


def measured(capability: Capability, reports: tuple[ToolReport, ...]) -> str | None:
    """Read the version of one toolchain a node's probe reported.

    Args:
        capability: The toolchain.
        reports: What the node answered.

    Returns:
        The version; the whole answer when a present toolchain answered in
        another shape (a shim with no toolchain to run prints an error), so
        it can equal no declaration and the refusal quotes it; or None when
        the probe reported the toolchain absent or not at all.
    """
    for report in reports:
        if report["name"] != PROBE_NAME[capability] or not report["present"]:
            continue
        version = _version_in(capability, report["version"])
        return report["version"] if version is None else version
    return None


def capability_gap(
    capability: Capability, declared: str | None, reports: tuple[ToolReport, ...]
) -> str | None:
    """Say how a node's declared toolchain disagrees with its probe.

    Args:
        capability: The toolchain.
        declared: The node's declaration of it.
        reports: What the node answered.

    Returns:
        None when the node declares none, or exactly the version its probe
        reported. Otherwise the disagreement, naming both and the
        declaration that would match.
    """
    if declared is None:
        return None
    found = measured(capability, reports)
    if found == declared:
        return None
    probe = PROBE_NAME[capability]
    reported = f"no {probe}" if found is None else f"{probe} {found!r}"
    fix = "null" if found is None or not VERSION.fullmatch(found) else repr(found)
    return (
        f"declares {capability} {declared!r} but its probe reports {reported}; the declaration "
        f"is what gives a node the {capability} tag, so {_CONSEQUENCE[capability]}. Set "
        f"{capability} to {fix}, or install that toolchain"
    )


def decode_capability(capability: Capability, value: JSONValue) -> str | None:
    """Read a node's declaration of one toolchain.

    Args:
        capability: The toolchain, which is also the key.
        value: The declared value.

    Returns:
        The version, or None for a node without that toolchain.

    Raises:
        JSONTypeError: If it is neither null nor a string of
            :data:`VERSION`'s shape.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise JSONTypeError(
            f"{capability} must be null or the version {_SOURCE[capability]}, got {value!r}; it "
            "is compared with the node's probe every tick"
        )
    return value


__all__ = [
    "MISMATCH_CODE",
    "PROBE_NAME",
    "VERSION",
    "Capability",
    "capability_gap",
    "decode_capability",
    "measured",
]
