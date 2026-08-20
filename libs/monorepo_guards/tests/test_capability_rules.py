from __future__ import annotations

from pathlib import Path

from monorepo_guards.capability_rules import CapabilityDerivationRule

_KIND = "capability-sizes-hardcoded"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_flags_hand_written_size_literal(tmp_path: Path) -> None:
    """A literal tuple is a hand-maintained copy and must be flagged."""
    path = tmp_path / "pkg" / "src" / "factory.py"
    _write(
        path,
        'CAPS = {\n    "supported_sizes": ("tiny", "small"),\n}\n',
    )

    violations = CapabilityDerivationRule().run([path])

    assert [v.kind for v in violations] == [_KIND]
    assert violations[0].line_no == 2


def test_allows_derived_declaration(tmp_path: Path) -> None:
    """tuple(MODEL_SIZES) cannot disagree with MODEL_SIZES, so it is allowed."""
    path = tmp_path / "pkg" / "src" / "factory.py"
    _write(
        path,
        'CAPS = {\n    "supported_sizes": tuple(MODEL_SIZES),\n}\n',
    )

    assert CapabilityDerivationRule().run([path]) == []


def test_allows_empty_tuple(tmp_path: Path) -> None:
    """() is the honest claim of a backend with no size table, not a copy."""
    path = tmp_path / "pkg" / "src" / "factory.py"
    _write(path, 'CAPS = {\n    "supported_sizes": (),\n}\n')

    assert CapabilityDerivationRule().run([path]) == []


def test_allows_empty_tuple_with_trailing_comment(tmp_path: Path) -> None:
    """The real declaration carries an explanatory comment; it must still pass."""
    path = tmp_path / "pkg" / "src" / "factory.py"
    _write(
        path,
        'CAPS = {\n    "supported_sizes": (),  # Size determined by hub_model_id\n}\n',
    )

    assert CapabilityDerivationRule().run([path]) == []


def test_flags_literal_with_trailing_comment(tmp_path: Path) -> None:
    """A comment must not let a hand-written literal through."""
    path = tmp_path / "pkg" / "src" / "factory.py"
    _write(
        path,
        'CAPS = {\n    "supported_sizes": ("small",),  # only one for now\n}\n',
    )

    assert [v.kind for v in CapabilityDerivationRule().run([path])] == [_KIND]


def test_flags_multiline_literal(tmp_path: Path) -> None:
    """A literal spread over several lines is still a literal."""
    path = tmp_path / "pkg" / "src" / "factory.py"
    _write(
        path,
        'CAPS = {\n    "supported_sizes": (\n        "tiny",\n    ),\n}\n',
    )

    assert [v.kind for v in CapabilityDerivationRule().run([path])] == [_KIND]


def test_ignores_files_under_tests(tmp_path: Path) -> None:
    """Test fixtures may build capability dicts by hand."""
    path = tmp_path / "pkg" / "tests" / "test_thing.py"
    _write(path, 'CAPS = {\n    "supported_sizes": ("tiny", "small"),\n}\n')

    assert CapabilityDerivationRule().run([path]) == []


def test_ignores_unrelated_lines(tmp_path: Path) -> None:
    """A file with no capability declaration produces nothing."""
    path = tmp_path / "pkg" / "src" / "other.py"
    _write(path, "x = 1\ny = 2\n")

    assert CapabilityDerivationRule().run([path]) == []


def test_rule_name_is_stable() -> None:
    """The orchestrator reports per-rule counts under this name."""
    assert CapabilityDerivationRule().name == "capability-sizes"


def test_catches_the_original_defect_verbatim(tmp_path: Path) -> None:
    """Frozen regression: the exact declaration this rule was written for.

    A rule that passes on synthetic input but not on the real thing is a rule
    that would not have prevented the incident. This is Model-Trainer's
    GPT2_CAPABILITIES as it stood before the fix, copied verbatim -- it
    advertised a "tiny" the size table did not implement, so asking for the
    advertised size raised a bare KeyError, and no test noticed because the one
    that looked relevant asserted the constant against a copy of itself.
    """
    path = tmp_path / "services" / "Model-Trainer" / "src" / "backend_factory.py"
    _write(
        path,
        "GPT2_CAPABILITIES: BackendCapabilities = {\n"
        '    "supports_train": True,\n'
        '    "supports_evaluate": True,\n'
        '    "supports_score": True,\n'
        '    "supports_generate": True,\n'
        '    "supports_distributed": False,\n'
        '    "supported_sizes": ("tiny", "small", "medium", "large"),\n'
        "}\n",
    )

    violations = CapabilityDerivationRule().run([path])

    assert [v.kind for v in violations] == [_KIND]
    assert violations[0].line_no == 7
    assert violations[0].line == '"supported_sizes": ("tiny", "small", "medium", "large"),'
