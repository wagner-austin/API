"""Tests for the encoded-run-fingerprint literal rule.

The case that motivated it is ``_flags_the_literal_that_actually_went_stale``:
those are the exact keys the two ``Model-Trainer`` fixtures carried, and they
are what a fingerprint literal looks like the moment before the type grows an
axis underneath it.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.run_fingerprint_rules import RunFingerprintLiteralRule


def _write(path: Path, text: str) -> None:
    """Write a source file for the rule to scan.

    Args:
        path: Where to write it.
        text: The source.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


_STALE_LITERAL = """
FINGERPRINT = {
    "image_digest": "sha256:abc",
    "gpu_model": "NVIDIA GeForce RTX 3090 Ti",
    "driver_version": "591.86",
    "determinism": {"stack": "torch", "settings": {}},
}
"""


def test_it_flags_the_literal_that_actually_went_stale(tmp_path: Path) -> None:
    target = tmp_path / "tests" / "test_cloze_api.py"
    _write(target, _STALE_LITERAL)

    violations = RunFingerprintLiteralRule().run([target])

    assert [v.kind for v in violations] == ["run-fingerprint-json-literal"]


def test_the_message_names_the_fix_rather_than_only_the_fault(tmp_path: Path) -> None:
    target = tmp_path / "tests" / "test_cloze_api.py"
    _write(target, _STALE_LITERAL)

    violations = RunFingerprintLiteralRule().run([target])

    assert "sample_run_fingerprint" in violations[0].line
    assert "encode_run_fingerprint" in violations[0].line


def test_it_reports_the_line_the_literal_starts_on(tmp_path: Path) -> None:
    target = tmp_path / "tests" / "test_cloze_api.py"
    _write(target, _STALE_LITERAL)

    violations = RunFingerprintLiteralRule().run([target])

    assert violations[0].line_no == 2


def test_a_literal_missing_one_axis_is_still_flagged(tmp_path: Path) -> None:
    # Missing an axis is the DEFECT, not an excuse: a four-key literal beside
    # a six-key type is precisely the state this rule exists to catch.
    target = tmp_path / "tests" / "test_x.py"
    _write(
        target,
        'F = {"image_digest": "", "gpu_model": "", "driver_version": "",'
        ' "determinism": {}, "host": {}}\n',
    )

    violations = RunFingerprintLiteralRule().run([target])

    assert [v.kind for v in violations] == ["run-fingerprint-json-literal"]


def test_a_mapping_that_merely_shares_one_key_is_left_alone(tmp_path: Path) -> None:
    # The rule requires every axis key together. A config or a header table
    # carrying "host" is not a fingerprint, and sweeping it up would make the
    # rule something people learn to ignore.
    target = tmp_path / "src" / "config.py"
    _write(target, 'SETTINGS = {"host": "localhost", "port": 5432, "gpu_model": "A100"}\n')

    violations = RunFingerprintLiteralRule().run([target])

    assert violations == []


def test_the_defining_module_may_spell_the_axes_out(tmp_path: Path) -> None:
    # `encode_run_fingerprint` IS the literal; a rule that forbade it there
    # would forbid the encoder from existing.
    target = tmp_path / "platform_core" / "src" / "platform_core" / "comparability.py"
    _write(target, _STALE_LITERAL)

    violations = RunFingerprintLiteralRule().run([target])

    assert violations == []


def test_a_file_with_no_dict_literals_is_clean(tmp_path: Path) -> None:
    target = tmp_path / "src" / "plain.py"
    _write(target, "VALUE = 1\n")

    violations = RunFingerprintLiteralRule().run([target])

    assert violations == []


def test_a_nested_literal_is_found_too(tmp_path: Path) -> None:
    # The one in `test_training_worker_manifest_and_family_branches` sat
    # inside a helper function, not at module scope.
    target = tmp_path / "tests" / "test_y.py"
    _write(
        target,
        "def build():\n"
        "    return {\n"
        '        "image_digest": "",\n'
        '        "gpu_model": "",\n'
        '        "driver_version": "",\n'
        '        "determinism": {},\n'
        "    }\n",
    )

    violations = RunFingerprintLiteralRule().run([target])

    assert [v.kind for v in violations] == ["run-fingerprint-json-literal"]


def test_a_dict_with_computed_keys_is_not_mistaken_for_one(tmp_path: Path) -> None:
    # `{**other}` has a None key in the AST. Reading it as a constant would
    # crash the visitor on ordinary code.
    target = tmp_path / "src" / "merge.py"
    _write(target, "def merge(a, b):\n    return {**a, **b}\n")

    violations = RunFingerprintLiteralRule().run([target])

    assert violations == []


def test_the_rule_names_itself_for_the_report(tmp_path: Path) -> None:
    assert RunFingerprintLiteralRule().name == "run-fingerprint-literal"
