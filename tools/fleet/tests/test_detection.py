"""The tags a runner claims with, read from its node's probe (MCPs 939ec5c7).

diphtheria's answer is its own, read with the sh probe on 2026-10-02, when
it matched fleet.json, so the drift cases change one side at a time.
"""

from __future__ import annotations

from fleet.contracts.detection import (
    GPU_PROBE,
    TESTDB_CONTAINER,
    TESTDB_PROBE,
    detected_tags,
    tag_drift,
)
from fleet.contracts.node import NodeConfig, NodeGpu
from fleet.contracts.tags import NodeTag, node_tags
from fleet.core import toolchain
from tests._toolchain_fixtures import node

#: diphtheria's capability lines, 2026-10-02, verbatim.
DIPHTHERIA_2026_10_02 = (
    "ffmpeg=yes=ffmpeg version 6.1.1-3ubuntu5\n"
    "cargo=yes=cargo 1.98.1 (797e8a9bc 2026-08-05)\n"
    "cxx=yes=13.3.0\n"
    "docker=yes=29.8.1\n"
    "stack=yes=29.8.1\n"
    "gpu=yes=NVIDIA RTX A2000 12GB, 8.6\n"
    "testdb=yes=pgvector/pgvector:pg16-bookworm\n"
)

#: DERIVED from :data:`DIPHTHERIA_2026_10_02`: a node with ffmpeg and no
#: compiler, card, test database or daemon, the shape pendragon answered
#: that day.
BARE = "ffmpeg=yes=ffmpeg version 6.1.1-3ubuntu5\ncxx=no=\ngpu=no=\ntestdb=no=\ndocker=no=\n"

#: diphtheria's card as the registry declares it.
A2000 = NodeGpu(
    model="NVIDIA RTX A2000 12GB",
    vram_mib=12282,
    compute_capability="8.6",
    driver_version="580.95.05",
)


def _diphtheria() -> NodeConfig:
    """diphtheria as the registry declares it.

    Returns:
        The node.
    """
    declared = node("diphtheria", rust="1.98.1", cxx="13.3.0", docker="29.8.1", stack="29.8.1")
    declared["gpu"] = A2000
    declared["test_database"] = True
    return declared


class TestTheProbeLines:
    def test_the_names_the_probes_print(self) -> None:
        assert (GPU_PROBE, TESTDB_PROBE, TESTDB_CONTAINER) == (
            "gpu",
            "testdb",
            "corvis-fleet-testdb",
        )

    def test_both_lines_are_read_from_the_probe(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_10_02)
        assert [report["name"] for report in reports][-2:] == ["gpu", "testdb"]


class TestDetectedTags:
    def test_diphtheria_answers_every_tag_it_declares_and_ffmpeg(self) -> None:
        detected = detected_tags(toolchain.read_reports(DIPHTHERIA_2026_10_02))

        assert detected == node_tags(_diphtheria()) | {NodeTag.FFMPEG}

    def test_a_bare_node_answers_linux_and_ffmpeg_only(self) -> None:
        detected = detected_tags(toolchain.read_reports(BARE))

        assert detected == {NodeTag.LINUX, NodeTag.FFMPEG}

    def test_installing_a_compiler_adds_its_tag_with_no_declaration(self) -> None:
        installed = BARE.replace("cxx=no=", "cxx=yes=13.3.0")

        assert NodeTag.CXX in detected_tags(toolchain.read_reports(installed))


class TestTagDrift:
    def test_a_node_matching_its_declaration_has_none(self) -> None:
        assert tag_drift(_diphtheria(), toolchain.read_reports(DIPHTHERIA_2026_10_02)) == ()

    def test_a_declared_card_and_test_database_the_probe_lacks(self) -> None:
        answer = DIPHTHERIA_2026_10_02.replace(
            "gpu=yes=NVIDIA RTX A2000 12GB, 8.6", "gpu=no="
        ).replace("testdb=yes=pgvector/pgvector:pg16-bookworm", "testdb=no=")

        assert tag_drift(_diphtheria(), toolchain.read_reports(answer)) == (
            "declares gpu 'NVIDIA RTX A2000 12GB, 8.6' but nvidia-smi reports none, so it claims "
            "without the gpu tag; correct gpu in its registry/machines file",
            "declares test_database true but no corvis-fleet-testdb container exists, so it "
            "claims without the testdb tag; set test_database to false in its registry/machines "
            "file, or provision the container",
        )

    def test_an_undeclared_card_test_database_and_compiler(self) -> None:
        answer = (
            "gpu=yes=NVIDIA GeForce GTX 1630, 7.5\n"
            "testdb=yes=pgvector/pgvector:pg16-bookworm\n"
            "cxx=yes=13.3.0\n"
        )

        assert tag_drift(node("pendragon"), toolchain.read_reports(answer)) == (
            "declares gpu none but nvidia-smi reports 'NVIDIA GeForce GTX 1630, 7.5', so it "
            "claims with the gpu tag; correct gpu in its registry/machines file",
            "declares test_database false but corvis-fleet-testdb exists "
            "(pgvector/pgvector:pg16-bookworm), so it claims with the testdb tag; set "
            "test_database to true in its registry/machines file",
            "declares cxx none but its probe reports cxx '13.3.0', so it claims with the cxx "
            "tag; set cxx to '13.3.0' in its registry/machines file",
        )
