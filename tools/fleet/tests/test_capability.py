"""Toolchains a node declares by version and its probe re-measures every tick.

MCPs board tasks 1e2da299 (``rust``) and 3f19c136 (``cxx``). The tag a
project requires is derived from the node's declaration, and the runner's
readiness gate compares that declaration with what the probe answered, so
these cases pin the three halves together: the contract that reads the
declaration, the reader that pulls the version out of a probe line, and the
gate that refuses a node whose two disagree. The answers are diphtheria's,
measured the evenings rustup was installed and the ``cxx`` line was added.
"""

from __future__ import annotations

import re

import pytest
from platform_core.errors import FleetErrorCode
from platform_core.json_utils import JSONTypeError

from fleet.contracts.capability import (
    MISMATCH_CODE,
    PROBE_NAME,
    Capability,
    capability_gap,
    decode_capability,
    measured,
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
    def test_each_capability_has_its_probe_line_and_refusal_code(self) -> None:
        assert [capability.value for capability in Capability] == ["rust", "cxx"]
        assert PROBE_NAME == {RUST: "cargo", CXX: "cxx"}
        assert MISMATCH_CODE == {
            RUST: FleetErrorCode.NODE_RUST_MISMATCH,
            CXX: FleetErrorCode.NODE_CXX_MISMATCH,
        }


class TestMeasured:
    def test_diphtheria_s_answers_read_as_their_versions(self) -> None:
        """cargo's version is its SECOND word, its last being a build date;
        the cxx line carries g++'s bare -dumpfullversion."""
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        assert measured(RUST, reports) == "1.98.1"
        assert measured(CXX, reports) == "13.3.0"

    def test_a_windows_vc_tools_version_has_four_parts(self) -> None:
        assert measured(CXX, _answered("cxx", True, "17.14.36310.24")) == "17.14.36310.24"

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


class TestCapabilityGap:
    def test_a_node_declaring_none_is_never_refused_even_with_the_toolchain(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        assert capability_gap(RUST, None, reports) is None
        assert capability_gap(CXX, None, reports) is None

    def test_the_measured_version_satisfies_its_declaration(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        assert capability_gap(RUST, "1.98.1", reports) is None
        assert capability_gap(CXX, "13.3.0", reports) is None

    def test_a_declared_toolchain_the_probe_does_not_find_says_set_null(self) -> None:
        assert capability_gap(RUST, "1.98.1", toolchain.read_reports(DIPHTHERIA_2026_09_23)) == (
            "declares rust '1.98.1' but its probe reports no cargo; the declaration is what "
            "gives a node the rust tag, so a crate build claimed on it would fail in poetry "
            "sync. Set rust to null, or install that toolchain"
        )
        assert capability_gap(CXX, "17.14.36310.24", _answered("cxx", False, "")) == (
            "declares cxx '17.14.36310.24' but its probe reports no cxx; the declaration is "
            "what gives a node the cxx tag, so an npm ci claimed on it would fail rebuilding a "
            "native module under node-gyp. Set cxx to null, or install that toolchain"
        )

    def test_another_version_names_the_one_that_would_match(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        assert capability_gap(RUST, "1.97.0", reports) == (
            "declares rust '1.97.0' but its probe reports cargo '1.98.1'; the declaration is "
            "what gives a node the rust tag, so a crate build claimed on it would fail in "
            "poetry sync. Set rust to '1.98.1', or install that toolchain"
        )
        assert capability_gap(CXX, "13.2.0", reports) == (
            "declares cxx '13.2.0' but its probe reports cxx '13.3.0'; the declaration is what "
            "gives a node the cxx tag, so an npm ci claimed on it would fail rebuilding a "
            "native module under node-gyp. Set cxx to '13.3.0', or install that toolchain"
        )

    def test_an_unreadable_answer_is_quoted_and_offers_null(self) -> None:
        assert capability_gap(RUST, "1.98.1", _answered("cargo", True, SHIM_ERROR)) == (
            f"declares rust '1.98.1' but its probe reports cargo {SHIM_ERROR!r}; the "
            "declaration is what gives a node the rust tag, so a crate build claimed on it "
            "would fail in poetry sync. Set rust to null, or install that toolchain"
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


class TestTheNodeContractCarriesBoth:
    def test_declared_toolchains_survive_encoding_and_read_back_by_capability(self) -> None:
        declared = node("diphtheria", rust="1.98.1", cxx="13.3.0")
        assert decode_node_config(encode_node_config(declared)) == declared
        assert declared_capability(declared, RUST) == "1.98.1"
        assert declared_capability(declared, CXX) == "13.3.0"
        assert encode_node_config(node())["cxx"] is None

    @pytest.mark.parametrize("key", ["rust", "cxx"])
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
            "cxx 13.3.0"
        )

    def test_a_node_without_either_says_nothing_of_them(self) -> None:
        assert toolchain.ready_summary(toolchain.read_reports(DIPHTHERIA_2026_09_23)) == (
            "python 3.11.15; node v24.21.0; poetry, git, make, tar present"
        )

    def test_each_disagreeing_declaration_is_refused_with_its_own_code(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27_CXX)
        rust = toolchain.readiness_gap(
            "diphtheria", node("diphtheria", rust="1.97.0", cxx="13.3.0"), reports
        )
        cxx = toolchain.readiness_gap(
            "lavender",
            node("lavender", cxx="17.14.36310.24"),
            toolchain.read_reports(DIPHTHERIA_2026_09_27),
        )
        stated = [("", "") if gap is None else (gap.code.value, gap.message) for gap in (rust, cxx)]
        assert stated == [
            (
                FleetErrorCode.NODE_RUST_MISMATCH.value,
                "diphtheria (diphtheria) declares rust '1.97.0' but its probe reports cargo "
                "'1.98.1'; the declaration is what gives a node the rust tag, so a crate build "
                "claimed on it would fail in poetry sync. Set rust to '1.98.1', or install that "
                "toolchain",
            ),
            (
                FleetErrorCode.NODE_CXX_MISMATCH.value,
                "lavender (lavender) declares cxx '17.14.36310.24' but its probe reports no cxx; "
                "the declaration is what gives a node the cxx tag, so an npm ci claimed on it "
                "would fail rebuilding a native module under node-gyp. Set cxx to null, or "
                "install that toolchain",
            ),
        ]
