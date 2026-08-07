"""Tests for the per-kind physics-claim checkers."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.physics_claims import (
    CLAIM_TARGETS,
    _public_symbol_addresses,
    run_physics_claim_rules,
)

from tests.scripts._physics_claim_fixtures import (
    _ONE_OF,
    FIXTURE_MODULE,
    FIXTURE_PACKAGE,
    _claims_page,
    _run,
    _write_page,
)


def test_module_target_binds_only_its_own_all(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare module is a legal target, enumerating just its __all__.

    Module targets are what let a large package be onboarded a module
    at a time instead of in one big-bang commit — the reverse-coverage
    rule is all-or-nothing per target, so the target has to be
    scopeable below package granularity.
    """
    _write_page(tmp_path, "pages.md", "# Empty\n")
    count = run_physics_claim_rules(tmp_path, package_name=FIXTURE_MODULE)
    # facts.__all__ is exactly ANSWER + double, and neither is claimed.
    assert count == 2
    out = capsys.readouterr().out
    assert f"{FIXTURE_MODULE}:ANSWER: public bound symbol has no wiki claim" in out
    assert f"{FIXTURE_MODULE}:double: public bound symbol has no wiki claim" in out


def test_module_target_goes_green_when_claimed(tmp_path: Path) -> None:
    """The module arm satisfies reverse coverage like the package arm."""
    claims = (
        f'{{"claims": ['
        f'{{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},'
        f'{{"id": "double", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2 * value", "probes": [{"args": [4], "expect": 8}]}'
        "]}"
    )
    _write_page(tmp_path, "mod.md", _claims_page(claims))
    assert run_physics_claim_rules(tmp_path, package_name=FIXTURE_MODULE) == 0


def test_module_without_all_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A physics module without __all__ cannot be reverse-checked."""
    _write_page(tmp_path, "pages.md", "# Empty\n")
    count = run_physics_claim_rules(tmp_path, package_name="tests.scripts.physics_fixture_noall")
    assert count == 1
    out = capsys.readouterr().out
    assert "tests.scripts.physics_fixture_noall.bare: bound module lacks __all__" in out


def test_real_repo_binding_is_green() -> None:
    """THE test: the real wiki claims match the real code, all targets.

    Runs with no ``package_name`` override, so it exercises the same
    multi-target path the guard uses — including reverse coverage
    pooled across every entry of :const:`CLAIM_TARGETS`.

    This replaced a single-target form pinned to ``PHYSICS_PACKAGE``.
    Once a second target existed, that form was actively wrong: the
    override restricts which targets claims may bind INTO, so the 61
    ``protocol.commands`` claims all reported "is outside
    tankpit_bot.physics". A single-target assertion cannot describe a
    repo with more than one bound target.
    """
    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "wiki" / "pages").is_dir()
    assert run_physics_claim_rules(repo_root) == 0


def test_claim_targets_are_all_importable() -> None:
    """Every declared target resolves; a typo here silently binds nothing."""
    for target in CLAIM_TARGETS:
        addresses, violations = _public_symbol_addresses(target)
        assert violations == []
        assert addresses, f"{target} contributed no public symbols"


def test_bytes_claim_matching_constant_is_green(tmp_path: Path) -> None:
    """A bytes claim whose latin-1 text equals the constant passes."""
    claims = (
        f'{{"claims": ['
        f'{{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},'
        f'{{"id": "double", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2 * value", "probes": [{"args": [1], "expect": 2}]},'
        f'{{"id": "greeting", "code": "{FIXTURE_MODULE}:GREETING", "bytes": "A1"}}'
        "]}"
    )
    _write_page(tmp_path, "bytes.md", _claims_page(claims))
    assert _run(tmp_path) == 0


def test_bytes_claim_mismatch_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bytes claim that disagrees with the constant is a violation."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:GREETING", "bytes": "A0"}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: claim says b'A0', code has b'A1'" in out


def test_bytes_claim_on_non_bytes_symbol_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Binding a bytes claim to an int constant is rejected."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER", "bytes": "42"}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: symbol is not a bytes constant" in out


def test_non_string_bytes_field_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bytes field must be a string; JSON has no bytes literal."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:GREETING", "bytes": 1}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: 'bytes' must be a latin-1 string" in out


def test_members_claim_binds_an_enum(tmp_path: Path) -> None:
    """An IntEnum class is claimed as its full name -> value mapping."""
    claims = (
        f'{{"claims": ['
        f'{{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},'
        f'{{"id": "double", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2 * value", "probes": [{"args": [2], "expect": 4}]},'
        f'{{"id": "colour", "code": "{FIXTURE_MODULE}:Colour",'
        ' "members": {"RED": 0, "BLUE": 1}}'
        "]}"
    )
    _write_page(tmp_path, "m.md", _claims_page(claims))
    assert _run(tmp_path) == 0


def test_members_claim_on_mapping_sequence_and_set(tmp_path: Path) -> None:
    """Mappings, ordered sequences and unordered sets all verify."""
    claims = (
        f'{{"claims": ['
        f'{{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},'
        f'{{"id": "double", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2 * value", "probes": [{"args": [2], "expect": 4}]},'
        f'{{"id": "fuel", "code": "{FIXTURE_MODULE}:COLOUR_FUEL",'
        ' "members": {"RED": 10, "BLUE": 20}},'
        f'{{"id": "names", "code": "{FIXTURE_MODULE}:COLOUR_NAMES",'
        ' "members": ["red", "blue"]},'
        f'{{"id": "codes", "code": "{FIXTURE_MODULE}:ODD_CODES",'
        ' "members": [2, 3, 1]}'
        "]}"
    )
    _write_page(tmp_path, "m.md", _claims_page(claims))
    assert _run(tmp_path) == 0


def test_members_unwraps_enum_valued_containers(tmp_path: Path) -> None:
    """A container whose VALUES are enum members compares by int value.

    Claimed as ``{"red": 0}``, not ``{"red": "Colour.RED"}`` — the wiki
    states the wire number, so the projection must reach it. No
    currently-bound module has this shape, which is exactly why the
    fixture carries one: otherwise the unwrapping arm is never run and
    the first enum-valued table added to a bound module would compare
    against a repr.
    """
    claims = (
        f'{{"claims": ['
        f'{{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},'
        f'{{"id": "double", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2 * value", "probes": [{"args": [2], "expect": 4}]},'
        f'{{"id": "by-name", "code": "{FIXTURE_MODULE}:COLOUR_BY_NAME",'
        ' "members": {"red": 0, "blue": 1}}'
        "]}"
    )
    _write_page(tmp_path, "m.md", _claims_page(claims))
    assert _run(tmp_path) == 0


def test_keys_claim_verifies_a_record_field_set(tmp_path: Path) -> None:
    """A record type's field names are claimed and compared sorted."""
    claims = (
        f'{{"claims": ['
        f'{{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},'
        f'{{"id": "double", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2 * value", "probes": [{"args": [2], "expect": 4}]},'
        f'{{"id": "rec", "code": "{FIXTURE_MODULE}:SampleRecord",'
        ' "keys": ["right", "left"]}'
        "]}"
    )
    _write_page(tmp_path, "k.md", _claims_page(claims))
    assert _run(tmp_path) == 0


def test_keys_claim_detects_a_renamed_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Renaming, adding or dropping a field breaks the claim.

    This is the whole point of the kind: a ``law`` claim on a record
    type keeps passing while its fields change underneath.
    """
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:SampleRecord",'
        ' "keys": ["left", "middle", "right"]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: claim keys" in out


def test_keys_claim_on_symbol_without_fields_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A keys claim on a plain constant is the wrong kind."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER", "keys": ["a"]}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "bad.md#x: symbol has no annotated fields to claim" in out


def test_keys_field_must_be_an_array(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A non-array keys field is rejected before comparison."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:SampleRecord", "keys": 3}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "bad.md#x: 'keys' must be a JSON array of field names" in out


def test_members_sequence_order_is_load_bearing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tuple's order IS the fact — RANK_NAMES is indexed by rank."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:COLOUR_NAMES",'
        ' "members": ["blue", "red"]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: claim members" in out


def test_members_omitting_an_entry_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Equality is total: an omitted entry fails like an invented one.

    A partially stated table reads as complete to the next reader,
    which is the failure the whole rule exists to prevent.
    """
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:COLOUR_FUEL",'
        ' "members": {"RED": 10}}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: claim members" in out


def test_members_on_scalar_symbol_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A members claim on an int is the wrong kind."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER", "members": {{"a": 1}}}}]}}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "bad.md#x: symbol is not an enum, mapping, sequence or set" in out


def test_members_must_be_object_or_array(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A scalar members field is rejected before comparison."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:COLOUR_FUEL", "members": 7}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: 'members' must be a JSON object or array" in out


def test_module_target_matches_exactly_not_by_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A claim outside the bound module is rejected by name.

    Guards the exact-match arm of ``_binds_into_target``: without it a
    module target could never be satisfied, since a module never
    startswith itself + '.'.
    """
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_PACKAGE}.other:Z", "value": 1}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert run_physics_claim_rules(tmp_path, package_name=FIXTURE_MODULE) >= 1
    out = capsys.readouterr().out
    assert f"'{FIXTURE_PACKAGE}.other' is outside {FIXTURE_MODULE}" in out


def test_law_claim_binds_a_symbol_green(tmp_path: Path) -> None:
    """A prose-law claim satisfies both directions for its symbol.

    ``law`` claims exist for physics symbols that cannot be verified
    on an int probe grid (predicates over protocol objects, raster
    geometry) — schema extension 2026-07-30 for
    ``state.line_of_sight`` ([[flag-triage-20260729]] F3).
    """
    claims = (
        f'{{"claims": ['
        f'{{"id": "answer-law", "code": "{FIXTURE_MODULE}:ANSWER",'
        ' "law": "The answer is pinned by prose."},'
        f'{{"id": "double", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2 * value", "probes": [{"args": [3], "expect": 6}]}'
        "]}"
    )
    _write_page(tmp_path, "law.md", _claims_page(claims))
    assert _run(tmp_path) == 0


def test_empty_law_string_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A blank law is not a claim."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER", "law": "  "}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: 'law' must be a non-empty prose string" in out


def test_non_string_law_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A non-string law field is rejected."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER", "law": 7}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: 'law' must be a non-empty prose string" in out


def test_law_with_value_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """law is mutually exclusive with value/probes like the other kinds."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER",'
        ' "value": 42, "law": "also prose"}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert f"physics_claim_violation bad.md#x: {_ONE_OF}" in out
