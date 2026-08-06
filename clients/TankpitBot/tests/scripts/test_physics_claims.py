"""Tests for the wiki-claim binding guard rule.

Each test builds a fake project tree whose ``wiki/pages`` claims bind
into the fixture package ``tests.scripts.physics_fixture``; the final
test runs the rule against the REAL repository — the actual
wiki-to-physics binding must be green.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.physics_claims import (
    CLAIM_FENCE_OPEN,
    CLAIM_TARGETS,
    _public_symbol_addresses,
    run_physics_claim_rules,
)

FIXTURE_PACKAGE = "tests.scripts.physics_fixture"
FIXTURE_MODULE = f"{FIXTURE_PACKAGE}.facts"
#: The exactly-one-of-kinds message, kept in one place so adding a
#: claim kind updates every assertion that quotes it.
_ONE_OF = "claim needs exactly one of value/bytes/members/probes/law"

GREEN_PAGE = f"""# Fixture Economy

{CLAIM_FENCE_OPEN}
{{
  "claims": [
    {{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},
    {{
      "id": "double",
      "code": "{FIXTURE_MODULE}:double",
      "formula": "2 * value",
      "probes": [
        {{"args": [2], "expect": 4}},
        {{"args": [0], "expect": 0}}
      ]
    }}
  ]
}}
```
"""


def _write_page(project_root: Path, name: str, text: str) -> None:
    """Create one wiki page inside a fake project tree.

    Args:
        project_root: Fake project root.
        name: Page filename.
        text: Page markdown text.
    """
    pages_dir = project_root / "wiki" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / name).write_text(text, encoding="utf-8")


def _claims_page(claims_json: str) -> str:
    """Wrap a claims JSON body in a minimal wiki page.

    Args:
        claims_json: The JSON text of the claim block.

    Returns:
        Markdown page text with one fenced claim block.
    """
    return f"# Page\n\n{CLAIM_FENCE_OPEN}\n{claims_json}\n```\n"


def _run(project_root: Path) -> int:
    """Run the rule against a fake tree bound to the fixture package.

    Args:
        project_root: Fake project root.

    Returns:
        Violation count.
    """
    return run_physics_claim_rules(project_root, package_name=FIXTURE_PACKAGE)


def test_missing_wiki_dir_passes(tmp_path: Path) -> None:
    """A tree without wiki/pages is out of scope for the rule."""
    assert _run(tmp_path) == 0


def test_green_binding_passes(tmp_path: Path) -> None:
    """A wiki that claims every public symbol correctly is green."""
    _write_page(tmp_path, "economy.md", GREEN_PAGE)
    assert _run(tmp_path) == 0


def test_unclosed_fence_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A claim fence with no closing marker is a violation."""
    _write_page(tmp_path, "bad.md", f"# Page\n\n{CLAIM_FENCE_OPEN}\n{{}}\n")
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert f"physics_claim_violation bad.md: unclosed '{CLAIM_FENCE_OPEN}' fence" in out


def test_malformed_json_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A claim block that is not valid JSON is a violation."""
    _write_page(tmp_path, "bad.md", _claims_page("{not json"))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md: claim block is not valid JSON" in out


def test_non_object_claim_block_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A claim block whose top level is not an object is a violation."""
    _write_page(tmp_path, "bad.md", _claims_page("[1, 2]"))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md: claim block must be a JSON object" in out


def test_missing_claims_list_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A claim block without a claims list is a violation."""
    _write_page(tmp_path, "bad.md", _claims_page('{"claims": 7}'))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md: claim block lacks a 'claims' list" in out


def test_non_object_claim_entry_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A claims list holding a non-object entry is a violation."""
    _write_page(tmp_path, "bad.md", _claims_page('{"claims": [7]}'))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md: claim entries must be JSON objects" in out


def test_claim_missing_id_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A claim without id/code cannot be checked."""
    _write_page(tmp_path, "bad.md", _claims_page('{"claims": [{"value": 1}]}'))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md: claim needs string 'id' and 'code' fields" in out


def test_claim_with_value_and_probes_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """value and probes are mutually exclusive."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER",'
        ' "value": 42, "probes": []}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert f"physics_claim_violation bad.md#x: {_ONE_OF}" in out


def test_claim_with_neither_value_nor_probes_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A claim must carry a check."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER"}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert f"physics_claim_violation bad.md#x: {_ONE_OF}" in out


def test_code_without_colon_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The code address must be module:symbol."""
    claims = '{"claims": [{"id": "x", "code": "nope", "value": 1}]}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: code 'nope' is not 'module:symbol'" in out


def test_code_outside_package_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Claims may only bind into the physics package."""
    claims = '{"claims": [{"id": "x", "code": "os.path:sep", "value": 1}]}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert f"physics_claim_violation bad.md#x: 'os.path' is outside {FIXTURE_PACKAGE}" in out


def test_unimportable_module_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A claim naming a module that does not import is a violation."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_PACKAGE}.nope:X", "value": 1}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert f"bad.md#x: module '{FIXTURE_PACKAGE}.nope' does not import" in out


def test_missing_symbol_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A claim naming an absent symbol is a violation."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:GONE", "value": 1}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert f"physics_claim_violation bad.md#x: 'GONE' not found in {FIXTURE_MODULE}" in out


def test_value_mismatch_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A drifted constant is the core red-gate case."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER", "value": 41}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: claim says 41, code has 42" in out


def test_value_on_bool_symbol_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """bool is not an int constant even though it subclasses int."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:TRUTHY", "value": 1}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: symbol is not an int constant" in out


def test_value_on_str_symbol_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-numeric symbol cannot satisfy a value claim."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:NAME", "value": 1}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: symbol is not an int constant" in out


def test_non_int_value_field_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The value field itself must be an int."""
    claims = f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER", "value": "42"}}]}}'
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: 'value' must be an int" in out


def test_probe_claim_without_formula_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Probe claims must state the human-readable formula."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:double",'
        ' "probes": [{"args": [1], "expect": 2}]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: 'formula' must be a string" in out


def test_non_list_probes_field_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The probes field must be a list."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2v", "probes": 7}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: 'probes' must be a list" in out


def test_empty_probe_grid_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A probe claim with no probes checks nothing."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2v", "probes": []}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: probe claim has an empty probe grid" in out


def test_probes_on_constant_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A probe claim must bind a callable."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:ANSWER",'
        ' "formula": "n/a", "probes": [{"args": [1], "expect": 2}]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "bad.md#x: symbol is not callable but claim has probes" in out


def test_non_object_probe_entry_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A probe entry that is not an object is a violation."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2v", "probes": [7]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: probe entries must be JSON objects" in out


def test_probe_without_args_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A probe without an args list is a violation."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2v", "probes": [{"expect": 2}]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: probe 'args' must be a list of ints" in out


def test_probe_with_non_int_args_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Probe args holding a non-int are a violation."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2v", "probes": [{"args": ["1"], "expect": 2}]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: probe 'args' must be a list of ints" in out


def test_probe_with_non_int_expect_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A probe expect that is not an int is a violation."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2v", "probes": [{"args": [1], "expect": "2"}]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: probe 'expect' must be an int" in out


def test_probe_arity_mismatch_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Probe args that do not fit the signature are a violation."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2v", "probes": [{"args": [1, 2], "expect": 2}]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: probe args [1, 2] do not fit the signature" in out


def test_non_int_probe_result_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A formula must return ints at every probe point."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:label",'
        ' "formula": "vN", "probes": [{"args": [1], "expect": 2}]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: probe [1] returned a non-int" in out


def test_probe_result_mismatch_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A drifted formula is the core red-gate case for callables."""
    claims = (
        f'{{"claims": [{{"id": "x", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2v", "probes": [{"args": [3], "expect": 7}]}]}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#x: probe [3] expected 7, got 6" in out


def test_duplicate_claim_id_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Claim ids are globally unique across the wiki."""
    claims = (
        f'{{"claims": [{{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},'
        f' {{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}}]}}'
    )
    _write_page(tmp_path, "bad.md", _claims_page(claims))
    assert _run(tmp_path) >= 1
    out = capsys.readouterr().out
    assert "physics_claim_violation bad.md#answer: duplicate claim id" in out


def test_unclaimed_public_symbol_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every __all__ symbol of the package must be claimed (reverse rule)."""
    claims = f'{{"claims": [{{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}}]}}'
    _write_page(tmp_path, "pages.md", _claims_page(claims))
    assert _run(tmp_path) == 1
    out = capsys.readouterr().out
    assert f"{FIXTURE_MODULE}:double: public bound symbol has no wiki claim" in out


def test_doubly_claimed_symbol_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A symbol bound by two claims has two masters (reverse rule)."""
    claims = (
        f'{{"claims": [{{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},'
        f' {{"id": "answer-again", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},'
        f' {{"id": "double", "code": "{FIXTURE_MODULE}:double",'
        ' "formula": "2v", "probes": [{"args": [1], "expect": 2}]}]}'
    )
    _write_page(tmp_path, "pages.md", _claims_page(claims))
    assert _run(tmp_path) == 1
    out = capsys.readouterr().out
    assert f"{FIXTURE_MODULE}:ANSWER: bound by 2 claims, expected exactly 1" in out


def test_unresolvable_target_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A target that names nothing importable fails the reverse sweep."""
    _write_page(tmp_path, "pages.md", "# Empty\n")
    count = run_physics_claim_rules(tmp_path, package_name="tests.scripts.no_such_target")
    assert count == 1
    out = capsys.readouterr().out
    assert "target 'tests.scripts.no_such_target' does not resolve" in out


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
    ``physics.line_of_sight`` ([[flag-triage-20260729]] F3).
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
