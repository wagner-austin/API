"""Tests for the diagnostic_kind producer-consumer wiring guard rule.

Fixture kinds are interpolated via f-strings on purpose: this file
lives in ``tests/`` and is itself scanned by the real-tree enforcement
run, so a literal ``diagnostic_kind="dead..."`` here would be a
violation of the very rule under test.
"""

from __future__ import annotations

from pathlib import Path

from scripts.diagnostic_kind_rules import FAKE_KIND_PREFIX, run_diagnostic_kind_rules

_DEAD = "dead_kind"
_LIVE = "live_kind"


def _write(path: Path, text: str) -> None:
    """Write one fixture module, creating parents.

    Args:
        path: Target file.
        text: Module source.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _emitter(kind: str) -> str:
    """Build an emitter module emitting one kind.

    Args:
        kind: The kind to emit.

    Returns:
        Module source.
    """
    return f'def go(emit_diagnostic):\n    emit_diagnostic(diagnostic_kind="{kind}", x=1)\n'


def test_missing_directories_pass(tmp_path: Path) -> None:
    """A tree with no src or tests has nothing to violate."""
    assert run_diagnostic_kind_rules(tmp_path) == 0


def test_live_wiring_passes(tmp_path: Path) -> None:
    """A consumer of an emitted kind and a faithful test fixture pass."""
    _write(tmp_path / "src" / "app" / "emit.py", _emitter(_LIVE))
    _write(
        tmp_path / "src" / "tankpit_bot" / "diagnostics" / "reader.py",
        f'def read(record):\n    kind = record["fields"].get("diagnostic_kind")\n'
        f'    return kind == "{_LIVE}"\n',
    )
    _write(
        tmp_path / "tests" / "test_reader.py",
        f'LINE = \'{{"diagnostic_kind": "{_LIVE}"}}\'\n',
    )

    assert run_diagnostic_kind_rules(tmp_path) == 0


def test_consumer_of_a_dead_kind_is_reported(tmp_path: Path) -> None:
    """A comparison against a never-emitted kind is dead wiring.

    The 2026-08-28 audit found three scorecard counters in exactly
    this state (emitters deleted in reworks, consumers rendering
    zero forever).
    """
    _write(tmp_path / "src" / "app" / "emit.py", _emitter(_LIVE))
    _write(
        tmp_path / "src" / "tankpit_bot" / "diagnostics" / "reader.py",
        f'def read(kind):\n    return kind == "{_DEAD}"\n',
    )

    assert run_diagnostic_kind_rules(tmp_path) == 1


def test_membership_and_match_consumption_are_scanned(tmp_path: Path) -> None:
    """``in``-tuples and ``match`` arms both count as consumption."""
    _write(tmp_path / "src" / "app" / "emit.py", _emitter(_LIVE))
    _write(
        tmp_path / "src" / "tankpit_bot" / "diagnostics" / "reader.py",
        f"def read(record, kind_field, other):\n"
        f'    if kind_field in ("{_LIVE}", "{_DEAD}", other):\n'
        f"        return True\n"
        f'    match record["diagnostic_kind"]:\n'
        f'        case "{_DEAD}":\n'
        f"            return True\n"
        f"        case _:\n"
        f"            return False\n",
    )

    assert run_diagnostic_kind_rules(tmp_path) == 2


def test_field_name_filtering_is_not_consumption(tmp_path: Path) -> None:
    """Comparing a FIELD NAME against "diagnostic_kind" collects nothing."""
    _write(
        tmp_path / "src" / "tankpit_bot" / "diagnostics" / "filter.py",
        'def keep(field_name):\n    return field_name != "diagnostic_kind"\n',
    )

    assert run_diagnostic_kind_rules(tmp_path) == 0


def test_fabricating_test_fixture_is_reported(tmp_path: Path) -> None:
    """A test feeding consumers a never-emitted kind certifies a corpse."""
    _write(tmp_path / "src" / "app" / "emit.py", _emitter(_LIVE))
    _write(
        tmp_path / "tests" / "test_fixture.py",
        f'def make():\n    return dict(diagnostic_kind="{_DEAD}")\n',
    )

    assert run_diagnostic_kind_rules(tmp_path) == 1


def test_raw_jsonl_fixture_strings_are_scanned(tmp_path: Path) -> None:
    """Kinds hidden inside raw JSONL string fixtures are still checked."""
    _write(tmp_path / "src" / "app" / "emit.py", _emitter(_LIVE))
    _write(
        tmp_path / "tests" / "test_jsonl.py",
        f'LINE = \'{{"diagnostic_kind": "{_DEAD}"}}\'\n',
    )

    assert run_diagnostic_kind_rules(tmp_path) == 1


def test_fake_prefixed_kinds_are_deliberate_and_pass(tmp_path: Path) -> None:
    """The self-describing fake_ prefix marks unknown-kind tests legal."""
    _write(
        tmp_path / "tests" / "test_fallthrough.py",
        f'KIND = dict(diagnostic_kind="{FAKE_KIND_PREFIX}noise")\n',
    )

    assert run_diagnostic_kind_rules(tmp_path) == 0


def test_non_constant_emission_is_skipped(tmp_path: Path) -> None:
    """The emitter helper's pass-through kwarg contributes no kind."""
    _write(
        tmp_path / "src" / "app" / "helper.py",
        "def helper(emit_diagnostic, diagnostic_kind):\n"
        "    emit_diagnostic(diagnostic_kind=diagnostic_kind)\n",
    )
    _write(
        tmp_path / "src" / "tankpit_bot" / "diagnostics" / "reader.py",
        f'def read(kind):\n    return kind == "{_DEAD}"\n',
    )

    assert run_diagnostic_kind_rules(tmp_path) == 1


def test_the_real_tree_has_live_wiring_only() -> None:
    """ENFORCEMENT: the repository's own wiring must be fully live.

    This is the standing gate the 2026-08-28 dead-diagnostic audit
    would have made unnecessary: every consumed kind has a producer,
    and every test-fabricated kind mirrors a real emitter (or wears
    the fake_ prefix).
    """
    project_root = Path(__file__).resolve().parents[2]

    assert run_diagnostic_kind_rules(project_root) == 0
