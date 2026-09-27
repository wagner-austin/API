"""A node's Rust toolchain: declared as a version, re-measured every tick.

MCPs board task 1e2da299. The ``rust`` tag a crate-building project requires
is derived from the node's ``rust`` declaration, and the runner's readiness
gate compares that declaration with what ``cargo --version`` answered, so
these cases pin the three halves together: the contract that reads the
declaration, the reader that pulls the version out of cargo's answer, and
the gate that refuses a node whose two disagree. The cargo answer is
diphtheria's, measured the evening rustup was installed there.
"""

from __future__ import annotations

import re

import pytest
from platform_core.errors import FleetErrorCode
from platform_core.json_utils import JSONTypeError

from fleet.contracts.node import decode_node_config, encode_node_config
from fleet.contracts.rust import decode_rust, measured_rust, rust_gap
from fleet.contracts.toolchain import ToolReport
from fleet.core import toolchain
from tests._toolchain_fixtures import DIPHTHERIA_2026_09_23, DIPHTHERIA_2026_09_27, node

#: A present cargo whose answer is not ``cargo <number> ...``: invented, the
#: shape a shim prints when it has no toolchain to run.
SHIM_ERROR = "error: no default toolchain is configured"


def _cargo(present: bool, version: str) -> tuple[ToolReport, ...]:
    """One cargo report, beside a python one the reader must skip.

    Args:
        present: Whether cargo was found.
        version: What it answered.

    Returns:
        The reports.
    """
    return (
        ToolReport(name="python", present=True, version="Python 3.11.9"),
        ToolReport(name="cargo", present=present, version=version),
    )


class TestMeasuredRust:
    def test_diphtheria_s_answer_reads_as_its_version_not_its_build_date(self) -> None:
        assert measured_rust(toolchain.read_reports(DIPHTHERIA_2026_09_27)) == "1.98.1"

    def test_a_probe_without_a_cargo_line_or_with_an_absent_one_measures_none(self) -> None:
        assert measured_rust(toolchain.read_reports(DIPHTHERIA_2026_09_23)) is None
        assert measured_rust(_cargo(False, "")) is None

    def test_an_answer_of_another_shape_comes_back_verbatim(self) -> None:
        assert measured_rust(_cargo(True, SHIM_ERROR)) == SHIM_ERROR
        assert measured_rust(_cargo(True, "cargo")) == "cargo"
        assert measured_rust(_cargo(True, "cargo nightly")) == "cargo nightly"


class TestRustGap:
    def test_a_node_declaring_none_is_never_refused_even_with_cargo(self) -> None:
        assert rust_gap(None, toolchain.read_reports(DIPHTHERIA_2026_09_27)) is None
        assert rust_gap(None, _cargo(False, "")) is None

    def test_the_measured_version_satisfies_its_declaration(self) -> None:
        assert rust_gap("1.98.1", toolchain.read_reports(DIPHTHERIA_2026_09_27)) is None

    def test_a_declared_toolchain_with_no_cargo_says_set_null(self) -> None:
        assert rust_gap("1.98.1", toolchain.read_reports(DIPHTHERIA_2026_09_23)) == (
            "declares rust '1.98.1' but its probe reports no cargo; the declaration is what "
            "gives a node the rust tag, so a crate build claimed on it would fail in poetry "
            "sync. Set rust to null, or install that toolchain"
        )

    def test_another_version_names_the_one_that_would_match(self) -> None:
        assert rust_gap("1.97.0", toolchain.read_reports(DIPHTHERIA_2026_09_27)) == (
            "declares rust '1.97.0' but its probe reports cargo '1.98.1'; the declaration is "
            "what gives a node the rust tag, so a crate build claimed on it would fail in "
            "poetry sync. Set rust to '1.98.1', or install that toolchain"
        )

    def test_an_unreadable_cargo_quotes_it_and_offers_null(self) -> None:
        assert rust_gap("1.98.1", _cargo(True, SHIM_ERROR)) == (
            f"declares rust '1.98.1' but its probe reports cargo {SHIM_ERROR!r}; the "
            "declaration is what gives a node the rust tag, so a crate build claimed on it "
            "would fail in poetry sync. Set rust to null, or install that toolchain"
        )


class TestDecodeRust:
    def test_null_and_a_version_decode(self) -> None:
        assert decode_rust(None) is None
        assert decode_rust("1.98.1") == "1.98.1"

    @pytest.mark.parametrize("value", ["1.98", "cargo 1.98.1", "1.98.1-nightly", 1.98, True])
    def test_anything_but_a_three_part_version_is_refused(self, value: str | float | bool) -> None:
        with pytest.raises(
            JSONTypeError,
            match=rf"^rust must be null or the version cargo --version prints, e.g. '1\.98\.1', "
            rf"got {re.escape(repr(value))}; it is compared with the node's probe",
        ):
            decode_rust(value)


class TestTheNodeContractCarriesRust:
    def test_a_declared_toolchain_survives_encoding(self) -> None:
        declared = node("diphtheria", rust="1.98.1")
        assert decode_node_config(encode_node_config(declared)) == declared
        assert encode_node_config(node())["rust"] is None

    def test_an_absent_rust_key_is_refused(self) -> None:
        encoded = encode_node_config(node())
        del encoded["rust"]
        with pytest.raises(JSONTypeError, match=r"^node must declare 'rust': the version cargo"):
            decode_node_config(encoded)

    def test_a_malformed_declaration_is_refused_through_the_node(self) -> None:
        with pytest.raises(JSONTypeError, match=r"^rust must be null or the version"):
            decode_node_config({**encode_node_config(node()), "rust": "stable"})


class TestTheReadinessGate:
    def test_a_node_whose_declaration_matches_is_ready_and_says_its_cargo(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_27)
        declared = node("diphtheria", rust="1.98.1")
        assert toolchain.readiness_gap("diphtheria", declared, reports) is None
        assert toolchain.ready_summary(reports) == (
            "python 3.11.15; node v24.21.0; poetry, git, make, tar present; cargo 1.98.1"
        )

    def test_a_node_without_cargo_says_nothing_of_it(self) -> None:
        assert toolchain.ready_summary(toolchain.read_reports(DIPHTHERIA_2026_09_23)) == (
            "python 3.11.15; node v24.21.0; poetry, git, make, tar present"
        )

    def test_a_disagreeing_declaration_is_refused_with_its_own_code(self) -> None:
        gap = toolchain.readiness_gap(
            "diphtheria",
            node("diphtheria", rust="1.97.0"),
            toolchain.read_reports(DIPHTHERIA_2026_09_27),
        )
        stated = ("", "") if gap is None else (gap.code.value, gap.message)
        assert stated == (
            FleetErrorCode.NODE_RUST_MISMATCH.value,
            "diphtheria (diphtheria) declares rust '1.97.0' but its probe reports cargo "
            "'1.98.1'; the declaration is what gives a node the rust tag, so a crate build "
            "claimed on it would fail in poetry sync. Set rust to '1.98.1', or install that "
            "toolchain",
        )
