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
- ``docker`` (MCPs board task 6c4516af). MCPs/execution-deploy runs the
  real make deploy against a throwaway compose project, and on the
  operator's ruling of 2026-09-27 it may only use a ROOTLESS daemon under a
  user outside the docker group (``execdocker``, provisioned by MCPs
  ``scripts/host/lib/fleet-exec-docker.sh`` on every docker node), never
  the daemon the node's own account uses. The version is that daemon's own
  ``ServerVersion``, reported only when the daemon also says it is rootless
  and that user has the compose and buildx CLI plugins the lane runs.
- ``stack`` (MCPs board task 554bffc1). doc-extract-api, transcriber-api
  and pg-backup-sidecar each run, inside their own ``make check``, a test
  that starts one of the corvis compose stack's images on its network
  (:data:`STACK_IMAGES`, :data:`STACK_NETWORK`). They passed only while
  diphtheria, which builds and runs that stack, was the one testdb node; on
  2026-09-29 lavender-wsl became the second, ran doc-extract-api at MCPs
  b55aa8ce8 and failed two such tests where ``docker run`` exited 125 for
  want of the image and the network. The version is the ServerVersion of
  the daemon the node's own account reaches, reported only when that daemon
  has the network and every one of the images.

A project that needs one requires the tag of the same name
(:mod:`fleet.contracts.tags`), and a node carries it when its declaration
names a version.

THE RUNNER CLAIMS WITH WHAT ITS PROBE FOUND, NOT WITH THE DECLARATION (MCPs
board task 939ec5c7). The runner's toolchain probe asks every tick, and a
toolchain that answers with a version gives the node its tag on that tick
(:func:`detected`, :mod:`fleet.contracts.detection`), so installing one makes
the node eligible on its next tick with no file edited, and removing one takes
the tag away just as fast. Until then a node whose answer differed from its
declaration claimed nothing at all, which idled it for every job, not only the
ones needing that toolchain. The declaration stays, as the version the hub's
own planning (:func:`fleet.contracts.tags.node_tags`) reads without asking the
node, and every tick compares it with the answer and logs the difference
(:func:`version_drift`) so fleet.json can be corrected.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.toolchain import ToolReport


class Capability(StrEnum):
    """A toolchain a node declares by version. Each value is also its tag."""

    RUST = "rust"
    CXX = "cxx"
    DOCKER = "docker"
    STACK = "stack"


#: The network the stack's containers share, which the three suites'
#: ``docker run --network`` names.
STACK_NETWORK: Final = "mcp-network"

#: The stack images a suite starts in its own ``make check``: doc-extract-api's
#: ``tests/test_docker_integration/_helpers.py``, transcriber-api's
#: ``tests/test_docker_integration.py`` and pg-backup-sidecar's
#: ``tests/entry.docker.integration.test.ts``, in that order.
STACK_IMAGES: Final[tuple[str, ...]] = (
    "mcps-doc-extract-worker:latest",
    "mcps-transcriber-worker:latest",
    "mcps-pg-backup-sidecar:latest",
)

#: The probe line each capability is read from. ``cargo`` is the tool's own
#: name because maturin builds with it; ``cxx`` is a synthetic line, since the
#: toolchain node-gyp finds is not one executable on both platforms (vswhere's
#: VC tools component on Windows, ``g++ -dumpfullversion`` on Linux); so is
#: ``docker``, whose answer is the rootless execdocker daemon's, not whatever
#: ``docker`` on the runner's PATH would reach; and so is ``stack``, whose
#: answer is that PATH daemon's, and only when it holds the stack.
PROBE_NAME: Final[dict[Capability, str]] = {
    Capability.RUST: "cargo",
    Capability.CXX: "cxx",
    Capability.DOCKER: "docker",
    Capability.STACK: "stack",
}

#: Where a declared version is read, for the decode refusal.
_SOURCE: Final[dict[Capability, str]] = {
    Capability.RUST: "cargo --version prints, e.g. '1.98.1'",
    Capability.CXX: (
        "vswhere reports for the VC tools on Windows or g++ -dumpfullversion prints on Linux, "
        "e.g. '13.3.0'"
    ),
    Capability.DOCKER: (
        "the execdocker user's rootless daemon reports as its ServerVersion, e.g. '29.8.1'"
    ),
    Capability.STACK: (
        f"the node account's docker daemon reports as its ServerVersion while it holds "
        f"{STACK_NETWORK} and {', '.join(STACK_IMAGES)}, e.g. '29.8.1'"
    ),
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
        the ``cxx``, ``docker`` and ``stack`` lines carry the bare version.
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
        it can equal no declaration and the drift line quotes it; or None when
        the probe reported the toolchain absent or not at all.
    """
    for report in reports:
        if report["name"] != PROBE_NAME[capability] or not report["present"]:
            continue
        version = _version_in(capability, report["version"])
        return report["version"] if version is None else version
    return None


def detected(capability: Capability, reports: tuple[ToolReport, ...]) -> bool:
    """Whether a node's probe found a working toolchain this tick.

    Args:
        capability: The toolchain.
        reports: What the node answered.

    Returns:
        True only when the toolchain answered with a version of
        :data:`VERSION`'s shape; a present shim that prints an error has no
        toolchain behind it, so it carries no tag.
    """
    found = measured(capability, reports)
    return found is not None and VERSION.fullmatch(found) is not None


def version_drift(
    capability: Capability, declared: str | None, reports: tuple[ToolReport, ...]
) -> str | None:
    """Say how a node's declared toolchain differs from what its probe found.

    Args:
        capability: The toolchain.
        declared: The node's declaration of it.
        reports: What the node answered.

    Returns:
        None when the declaration is exactly the version the probe reported,
        or both say there is none. Otherwise the difference, naming both, the
        tag the runner claims with this tick (:func:`detected`) and the
        declaration that would match.
    """
    found = measured(capability, reports)
    if found == declared:
        return None
    probe = PROBE_NAME[capability]
    reported = f"no {probe}" if found is None else f"{probe} {found!r}"
    claims = "with" if detected(capability, reports) else "without"
    fix = repr(found) if detected(capability, reports) else "null"
    said = "none" if declared is None else repr(declared)
    return (
        f"declares {capability} {said} but its probe reports {reported}, so it claims {claims} "
        f"the {capability} tag; set {capability} to {fix} in fleet.json"
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
    "PROBE_NAME",
    "STACK_IMAGES",
    "STACK_NETWORK",
    "VERSION",
    "Capability",
    "decode_capability",
    "detected",
    "measured",
    "version_drift",
]
