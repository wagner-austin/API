"""Toolchains a node declares by version and its probe re-measures every tick.

MCPs board tasks 1e2da299 (``rust``) and 3f19c136 (``cxx``). The runner
claims a toolchain's tag when its probe answers a version (MCPs board task
939ec5c7) and compares the node's declaration with that answer every tick,
so these cases pin the halves together: the contract that reads the
declaration, the reader that pulls the version out of a probe line, what
counts as detected, and the drift line a disagreement logs instead of the
refusal it used to be. The answers are diphtheria's, measured the evenings
rustup was installed and the ``cxx`` line was added.
"""

from __future__ import annotations

import re

import pytest
from platform_core.json_utils import JSONTypeError

from fleet.contracts.capability import (
    PROBE_NAME,
    STACK_IMAGES,
    STACK_NETWORK,
    Capability,
    decode_capability,
    detected,
    measured,
    version_drift,
)
from fleet.contracts.node import declared_capability, decode_node_config, encode_node_config
from fleet.contracts.toolchain import ToolReport
from fleet.core import toolchain
from tests._toolchain_fixtures import (
    DIPHTHERIA_2026_09_23,
    DIPHTHERIA_2026_09_27,
    DIPHTHERIA_2026_09_27_CXX,
    node,
)

#: A present toolchain whose answer carries no version: invented, the shape
#: a shim prints when it has no toolchain to run.
SHIM_ERROR = "error: no default toolchain is configured"

RUST = Capability.RUST
CXX = Capability.CXX
DOCKER = Capability.DOCKER
STACK = Capability.STACK


def _answered(name: str, present: bool, version: str) -> tuple[ToolReport, ...]:
    """One probe line, beside a python one the reader must skip.

    Args:
        name: The probe line's name.
        present: Whether the toolchain was found.
        version: What the line carried.

    Returns:
        The reports.
    """
    return (
        ToolReport(name="python", present=True, version="Python 3.11.9"),
        ToolReport(name=name, present=present, version=version),
    )


class TestTheCapabilityTables:
    def test_each_capability_has_its_probe_line(self) -> None:
        assert [capability.value for capability in Capability] == [
            "rust",
            "cxx",
            "docker",
            "stack",
        ]
        assert PROBE_NAME == {RUST: "cargo", CXX: "cxx", DOCKER: "docker", STACK: "stack"}

    def test_the_stack_is_the_network_and_the_three_suites_images(self) -> None:
        """What doc-extract-api, transcriber-api and pg-backup-sidecar start
        in their own make check (MCPs board task 554bffc1)."""
        assert STACK_NETWORK == "mcp-network"
        assert STACK_IMAGES == (
            "mcps-doc-extract-worker:latest",
            "mcps-transcriber-worker:latest",
            "mcps-pg-backup-sidecar:latest",
        )


class TestMeasured:
    def test_diphtheria_s_answers_read_as_their_versions(self) -> None:
        """cargo's version is its SECOND word, its last being a build date;
        the cxx line carries g++'s bare -dumpfullversion."""
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        assert measured(RUST, reports) == "1.98.1"
        assert measured(CXX, reports) == "13.3.0"

    def test_a_windows_vc_tools_version_has_four_parts(self) -> None:
        assert measured(CXX, _answered("cxx", True, "17.14.36310.24")) == "17.14.36310.24"

    def test_the_docker_line_carries_the_rootless_daemon_s_bare_server_version(self) -> None:
        """MCPs board task 6c4516af: the probe prints only the ServerVersion,
        and only when the daemon says it is rootless."""
        assert measured(DOCKER, _answered("docker", True, "29.8.1")) == "29.8.1"
        assert measured(DOCKER, _answered("docker", False, "")) is None

    def test_the_stack_line_carries_the_stack_daemon_s_bare_server_version(self) -> None:
        """MCPs board task 554bffc1: diphtheria's answer of 2026-09-29, and
        lavender-wsl's, whose daemon has no mcp-network."""
        assert measured(STACK, _answered("stack", True, "29.8.1")) == "29.8.1"
        assert measured(STACK, _answered("stack", False, "")) is None

    def test_an_absent_or_missing_line_measures_none(self) -> None:
        older = toolchain.read_reports(DIPHTHERIA_2026_09_23)
        assert measured(RUST, older) is None
        assert measured(CXX, toolchain.read_reports(DIPHTHERIA_2026_09_27)) is None
        assert measured(RUST, _answered("cargo", False, "")) is None
        assert measured(CXX, _answered("cxx", False, "")) is None

    def test_an_answer_of_another_shape_comes_back_verbatim(self) -> None:
        assert measured(RUST, _answered("cargo", True, SHIM_ERROR)) == SHIM_ERROR
        assert measured(RUST, _answered("cargo", True, "cargo")) == "cargo"
        assert measured(RUST, _answered("cargo", True, "cargo nightly")) == "cargo nightly"
        assert measured(CXX, _answered("cxx", True, SHIM_ERROR)) == SHIM_ERROR


class TestDetected:
    def test_a_version_is_a_detected_toolchain(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        assert detected(RUST, reports)
        assert detected(CXX, reports)

    def test_an_absent_line_or_a_shim_s_error_is_not(self) -> None:
        assert not detected(RUST, toolchain.read_reports(DIPHTHERIA_2026_09_23))
        assert not detected(DOCKER, _answered("docker", False, ""))
        assert not detected(RUST, _answered("cargo", True, SHIM_ERROR))


class TestVersionDrift:
    def test_a_matching_declaration_or_none_on_both_sides_is_no_drift(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        assert version_drift(RUST, "1.98.1", reports) is None
        assert version_drift(CXX, "13.3.0", reports) is None
        assert version_drift(DOCKER, None, reports) is None

    def test_an_undeclared_toolchain_is_claimed_and_named(self) -> None:
        """Installing one makes the node eligible with no file edited."""
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        assert version_drift(RUST, None, reports) == (
            "declares rust none but its probe reports cargo '1.98.1', so it claims with the "
            "rust tag; set rust to '1.98.1' in fleet.json"
        )

    def test_a_declared_toolchain_the_probe_does_not_find_is_claimed_without(self) -> None:
        assert version_drift(RUST, "1.98.1", toolchain.read_reports(DIPHTHERIA_2026_09_23)) == (
            "declares rust '1.98.1' but its probe reports no cargo, so it claims without the "
            "rust tag; set rust to null in fleet.json"
        )
        assert version_drift(STACK, "29.8.1", _answered("stack", False, "")) == (
            "declares stack '29.8.1' but its probe reports no stack, so it claims without the "
            "stack tag; set stack to null in fleet.json"
        )

    def test_another_version_keeps_the_tag_and_names_the_one_that_would_match(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        assert version_drift(CXX, "13.2.0", reports) == (
            "declares cxx '13.2.0' but its probe reports cxx '13.3.0', so it claims with the "
            "cxx tag; set cxx to '13.3.0' in fleet.json"
        )

    def test_an_unreadable_answer_is_quoted_and_claims_without(self) -> None:
        assert version_drift(RUST, "1.98.1", _answered("cargo", True, SHIM_ERROR)) == (
            f"declares rust '1.98.1' but its probe reports cargo {SHIM_ERROR!r}, so it claims "
            "without the rust tag; set rust to null in fleet.json"
        )


class TestDecodeCapability:
    def test_null_and_versions_of_two_to_four_parts_decode(self) -> None:
        assert decode_capability(RUST, None) is None
        assert decode_capability(RUST, "1.98.1") == "1.98.1"
        assert decode_capability(CXX, "13.3") == "13.3"
        assert decode_capability(CXX, "17.14.36310.24") == "17.14.36310.24"

    @pytest.mark.parametrize(
        "value", ["1", "cargo 1.98.1", "1.98.1-nightly", "1.2.3.4.5", 1.98, True]
    )
    def test_anything_else_is_refused_naming_where_the_version_comes_from(
        self, value: str | float | bool
    ) -> None:
        with pytest.raises(
            JSONTypeError,
            match=rf"^rust must be null or the version cargo --version prints, e\.g\. '1\.98\.1', "
            rf"got {re.escape(repr(value))}; it is compared with the node's probe every tick$",
        ):
            decode_capability(RUST, value)

    def test_the_cxx_refusal_names_both_platforms_sources(self) -> None:
        with pytest.raises(
            JSONTypeError,
            match=r"^cxx must be null or the version vswhere reports for the VC tools on "
            r"Windows or g\+\+ -dumpfullversion prints on Linux, e\.g\. '13\.3\.0', got 'msvc'",
        ):
            decode_capability(CXX, "msvc")

    def test_the_stack_refusal_names_the_network_and_every_image(self) -> None:
        with pytest.raises(
            JSONTypeError,
            match=r"^stack must be null or the version the node account's docker daemon reports "
            r"as its ServerVersion while it holds mcp-network and "
            r"mcps-doc-extract-worker:latest, mcps-transcriber-worker:latest, "
            r"mcps-pg-backup-sidecar:latest, e\.g\. '29\.8\.1', got True",
        ):
            decode_capability(STACK, True)


class TestTheNodeContractCarriesBoth:
    def test_declared_toolchains_survive_encoding_and_read_back_by_capability(self) -> None:
        declared = node("diphtheria", rust="1.98.1", cxx="13.3.0", docker="29.8.1", stack="29.8.2")
        assert decode_node_config(encode_node_config(declared)) == declared
        assert declared_capability(declared, RUST) == "1.98.1"
        assert declared_capability(declared, CXX) == "13.3.0"
        assert declared_capability(declared, DOCKER) == "29.8.1"
        assert declared_capability(declared, STACK) == "29.8.2"
        assert encode_node_config(node())["cxx"] is None
        assert encode_node_config(node())["docker"] is None
        assert encode_node_config(node())["stack"] is None

    @pytest.mark.parametrize("key", ["rust", "cxx", "docker", "stack"])
    def test_an_absent_key_is_refused(self, key: str) -> None:
        encoded = encode_node_config(node())
        del encoded[key]
        with pytest.raises(
            JSONTypeError,
            match=rf"^node must declare '{key}': the version of that toolchain its probe reports",
        ):
            decode_node_config(encoded)

    def test_a_malformed_declaration_is_refused_through_the_node(self) -> None:
        with pytest.raises(JSONTypeError, match=r"^cxx must be null or the version"):
            decode_node_config({**encode_node_config(node()), "cxx": "stable"})


class TestTheReadinessGate:
    def test_a_node_whose_declarations_match_is_ready_and_says_both(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        declared = node("diphtheria", rust="1.98.1", cxx="13.3.0")
        assert toolchain.readiness_gap("diphtheria", declared, reports) is None
        assert toolchain.ready_summary(reports) == (
            "python 3.11.15; node v24.21.0; poetry, git, make, tar present; cargo 1.98.1; "
            "cxx 13.3.0; ffmpeg absent; hooks absent; go absent; chrome absent"
        )

    def test_a_node_without_either_says_nothing_of_them(self) -> None:
        assert toolchain.ready_summary(toolchain.read_reports(DIPHTHERIA_2026_09_23)) == (
            "python 3.11.15; node v24.21.0; poetry, git, make, tar present; ffmpeg absent; "
            "hooks absent; go absent; chrome absent"
        )

    def test_a_disagreeing_declaration_closes_no_gate(self) -> None:
        """It used to refuse the node for every job (MCPs board task 939ec5c7)."""
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        assert (
            toolchain.readiness_gap(
                "diphtheria", node("diphtheria", rust="1.97.0", cxx="13.3.0"), reports
            )
            is None
        )
        assert (
            toolchain.readiness_gap(
                "lavender",
                node("lavender", cxx="17.14.36310.24"),
                toolchain.read_reports(DIPHTHERIA_2026_09_27),
            )
            is None
        )
