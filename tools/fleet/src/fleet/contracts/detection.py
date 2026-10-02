"""The tags a runner claims with, read from its node every tick.

MCPs board task 939ec5c7, the operator on 2026-10-02: "capable pcs are tagged
and take jobs tagged for their abikities". Until then a node's tags were the
fields of its fleet.json entry, so a tool installed on a node did nothing
until somebody edited the file in the API repo, and a declaration that no
longer matched idled the node for every job. Now the toolchain probe the
runner already pays for each tick answers every capability a tag stands for,
and the runner claims with what it answered:

* ``gpu``: a CUDA device ``nvidia-smi`` reports, the line
  :data:`GPU_PROBE`, answered as ``<name>, <compute capability>``, which is
  what the tag has always meant (never an integrated adapter).
* ``testdb``: the loopback postgres container :data:`TESTDB_CONTAINER`
  exists under the node account's docker daemon, the line
  :data:`TESTDB_PROBE`, answered with its image. MCPs ``scripts/testdb-setup.sh``
  restarts it empty before each run, so existing is what a suite needs.
* ``rust``, ``cxx``, ``docker``, ``stack``: the toolchain answered with a
  version (:func:`fleet.contracts.capability.detected`).
* ``ffmpeg`` and any other tagged tool: the executable was found
  (:func:`fleet.contracts.tags.tool_tags`).

The platform tag is the declared one, because it chose the dialect the probe
was written in: a node of the other platform cannot answer it at all.
``elevated`` is not here: it names a second runner the node declares, and
that runner re-reads its token every tick (:mod:`fleet.contracts.elevation`).

The declaration is still compared every tick (:func:`tag_drift`), because the
hub's own planning reads it without asking the node
(:func:`fleet.contracts.tags.node_tags`), and a difference means fleet.json
should change.
"""

from __future__ import annotations

from typing import Final

from fleet.contracts.capability import Capability, detected, version_drift
from fleet.contracts.node import NodeConfig, declared_capability
from fleet.contracts.tags import CAPABILITY_TAG, PLATFORM_TAG, NodeTag, tool_tags
from fleet.contracts.toolchain import ToolReport

#: The toolchain probe's line for the node's CUDA device.
GPU_PROBE: Final = "gpu"

#: The toolchain probe's line for the fleet test database's container.
TESTDB_PROBE: Final = "testdb"

#: The loopback postgres container a ``testdb`` node runs.
TESTDB_CONTAINER: Final = "corvis-fleet-testdb"


def _answer(name: str, reports: tuple[ToolReport, ...]) -> str | None:
    """What one probe line answered, when it reported the thing present.

    Args:
        name: The line's name.
        reports: What the node answered.

    Returns:
        The line's answer, or None when it reported absent or not at all.
    """
    for report in reports:
        if report["name"] == name and report["present"]:
            return report["version"]
    return None


def detected_tags(node: NodeConfig, reports: tuple[ToolReport, ...]) -> frozenset[NodeTag]:
    """The tags a node's probe found this tick.

    Args:
        node: The node's declaration, for its platform.
        reports: What its toolchain probe answered.

    Returns:
        Its platform's tag, plus ``gpu`` and ``testdb`` when their lines
        answered present, the tag of every capability that answered with a
        version, and the tag of every tagged tool found. Never ``elevated``.
    """
    tags: set[NodeTag] = {PLATFORM_TAG[node["platform"]]}
    if _answer(GPU_PROBE, reports) is not None:
        tags.add(NodeTag.GPU)
    if _answer(TESTDB_PROBE, reports) is not None:
        tags.add(NodeTag.TESTDB)
    tags.update(tag for capability, tag in CAPABILITY_TAG.items() if detected(capability, reports))
    return frozenset(tags | tool_tags(reports))


def tag_drift(node: NodeConfig, reports: tuple[ToolReport, ...]) -> tuple[str, ...]:
    """Every way a node's declaration differs from what its probe found.

    Args:
        node: The node's declaration.
        reports: What its toolchain probe answered this tick.

    Returns:
        One sentence per difference, in tag order: a declared CUDA device the
        probe does not report or a device it reports that differs from the
        declaration or is undeclared, the test database the same way, and each
        capability whose version differs
        (:func:`fleet.contracts.capability.version_drift`). Empty when the
        declaration matches.
    """
    lines: list[str] = []
    gpu = node["gpu"]
    declared_gpu = None if gpu is None else f"{gpu['model']}, {gpu['compute_capability']}"
    found_gpu = _answer(GPU_PROBE, reports)
    if found_gpu != declared_gpu:
        said = "none" if declared_gpu is None else repr(declared_gpu)
        reported = "none" if found_gpu is None else repr(found_gpu)
        claims = "without" if found_gpu is None else "with"
        lines.append(
            f"declares gpu {said} but nvidia-smi reports {reported}, so it claims {claims} the "
            "gpu tag; correct gpu in fleet.json"
        )
    found_testdb = _answer(TESTDB_PROBE, reports)
    if node["test_database"] != (found_testdb is not None):
        if found_testdb is None:
            lines.append(
                f"declares test_database true but no {TESTDB_CONTAINER} container exists, so it "
                "claims without the testdb tag; set test_database to false in fleet.json, or "
                "provision the container"
            )
        else:
            lines.append(
                f"declares test_database false but {TESTDB_CONTAINER} exists ({found_testdb}), "
                "so it claims with the testdb tag; set test_database to true in fleet.json"
            )
    lines.extend(
        drift
        for capability in Capability
        if (drift := version_drift(capability, declared_capability(node, capability), reports))
        is not None
    )
    return tuple(lines)


__all__ = [
    "GPU_PROBE",
    "TESTDB_CONTAINER",
    "TESTDB_PROBE",
    "detected_tags",
    "tag_drift",
]
