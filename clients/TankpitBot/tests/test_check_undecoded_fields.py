"""Tests for the undecoded-field guard (Phase 5).

Covers each banned pattern, common false-positive scenarios, the
production source files (current steady state must be clean), and the
CLI entry point.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.check_undecoded_fields import (
    DEFAULT_TARGETS,
    Violation,
    find_violations,
    find_violations_in_source,
    run,
)

from scripts import check_undecoded_fields as guard
from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import PathExistsProtocol, ReadTextProtocol


class _FakeFileSystem:
    """Save-and-restore fake for the guard's file-read hooks."""

    def __init__(self) -> None:
        """Initialise with no virtual files registered."""
        self._files: dict[str, str] = {}

    def write(self, path: Path, content: str) -> None:
        """Register a virtual file's contents."""
        self._files[str(path)] = content

    def path_exists(self, path: Path) -> bool:
        """Return True when ``path`` was registered."""
        return str(path) in self._files

    def read_text(self, path: Path) -> str:
        """Return the contents of ``path``."""
        return self._files[str(path)]


def _install_fake_filesystem() -> tuple[_FakeFileSystem, PathExistsProtocol, ReadTextProtocol]:
    """Swap the script hooks for a fake; return originals for restore."""
    fake = _FakeFileSystem()
    original_path_exists: PathExistsProtocol = _test_hooks.path_exists
    original_read_text: ReadTextProtocol = _test_hooks.read_text
    _test_hooks.path_exists = fake.path_exists
    _test_hooks.read_text = fake.read_text
    return (fake, original_path_exists, original_read_text)


class TestFindViolationsInSource:
    """Direct tests for the per-file scanner."""

    def test_clean_typeddict_with_named_fields_returns_empty(self) -> None:
        """A TypedDict with only semantically-named fields is clean."""
        source = (
            "from typing import TypedDict\n"
            "class GoodDict(TypedDict):\n"
            "    msg_type: int\n"
            "    aim_x: int\n"
            "    aim_y: int\n"
        )
        assert find_violations_in_source(Path("dummy.py"), source) == []

    @pytest.mark.parametrize(
        "field_name",
        ["unk1", "unk2", "unk99"],
    )
    def test_flags_unk_digit_fields(self, field_name: str) -> None:
        """Every ``unkN`` field is flagged regardless of digit count."""
        source = f"from typing import TypedDict\nclass BadDict(TypedDict):\n    {field_name}: int\n"
        violations = find_violations_in_source(Path("dummy.py"), source)
        assert len(violations) == 1
        assert violations[0].field_name == field_name
        assert violations[0].typed_dict_name == "BadDict"

    @pytest.mark.parametrize(
        "field_name",
        ["unknown_byte", "unknown_byte0", "unknown_byte_a"],
    )
    def test_flags_unknown_byte_fields(self, field_name: str) -> None:
        """Every ``unknown_byte*`` field is flagged."""
        source = f"from typing import TypedDict\nclass BadDict(TypedDict):\n    {field_name}: int\n"
        violations = find_violations_in_source(Path("dummy.py"), source)
        assert len(violations) == 1
        assert violations[0].field_name == field_name

    @pytest.mark.parametrize(
        "field_name",
        ["padding", "padding0", "padding42"],
    )
    def test_flags_padding_fields(self, field_name: str) -> None:
        """Every ``padding*`` field is flagged."""
        source = f"from typing import TypedDict\nclass BadDict(TypedDict):\n    {field_name}: int\n"
        violations = find_violations_in_source(Path("dummy.py"), source)
        assert len(violations) == 1

    @pytest.mark.parametrize(
        "field_name",
        ["reserved", "reserved0", "reserved_3"],
    )
    def test_flags_reserved_fields(self, field_name: str) -> None:
        """``reserved`` and ``reservedN`` are flagged; ``reserved_*`` is not.

        The pattern is intentionally strict so a field named
        ``reserved_for_promotion_state`` (which would be legitimate)
        passes through.
        """
        source = f"from typing import TypedDict\nclass BadDict(TypedDict):\n    {field_name}: int\n"
        violations = find_violations_in_source(Path("dummy.py"), source)
        # reserved_3 does NOT match ``^reserved\\d*$`` because of the
        # underscore. Verify the expectation explicitly.
        if field_name == "reserved_3":
            assert violations == []
        else:
            assert len(violations) == 1

    def test_ignores_non_typeddict_classes(self) -> None:
        """A regular class with ``unk1`` is not flagged.

        Only TypedDict subclasses encode wire structure; ``unk1`` on a
        helper dataclass or vanilla class is the author's choice.
        """
        source = "class Plain:\n    unk1: int = 0\n"
        assert find_violations_in_source(Path("dummy.py"), source) == []

    def test_ignores_unknown_in_data_collectors(self) -> None:
        """``unknown`` / ``unknown_counts`` are legitimate names elsewhere.

        Diagnostic capture stats collect "unknowns" into TypedDicts and
        must not be flagged. The pattern only matches the literal
        ``unknown_byte*`` form, not ``unknown_counts`` or ``unknown``.
        """
        source = (
            "from typing import TypedDict\n"
            "class StatsDict(TypedDict):\n"
            "    unknown: dict[str, int]\n"
            "    unknown_counts: dict[str, int]\n"
        )
        assert find_violations_in_source(Path("dummy.py"), source) == []

    def test_recognises_attribute_typeddict_base(self) -> None:
        """``typing.TypedDict`` / ``typing_extensions.TypedDict`` are recognised."""
        source = "import typing\nclass BadDict(typing.TypedDict):\n    unk1: int\n"
        violations = find_violations_in_source(Path("dummy.py"), source)
        assert len(violations) == 1
        assert violations[0].field_name == "unk1"

    def test_skips_non_name_targets_inside_typeddict(self) -> None:
        """Inside a TypedDict, attribute-target ``AnnAssign`` is skipped.

        Real TypedDict fields use simple-name targets; an exotic form
        like ``self.unk1: int`` is invalid for TypedDict semantics but
        the AST still parses it. The scanner must skip it rather than
        crash on ``stmt.target.id`` lookup.
        """
        source = (
            "from typing import TypedDict\n"
            "class BadDict(TypedDict):\n"
            "    self.unk1: int\n"
            "    aim_x: int\n"
        )
        # The simple-name field aim_x is clean; the attribute target is skipped.
        assert find_violations_in_source(Path("dummy.py"), source) == []


class TestFindViolations:
    """Tests for ``find_violations`` using the fake filesystem."""

    def setup_method(self) -> None:
        """Install the fake filesystem."""
        (
            self._fake,
            self._original_path_exists,
            self._original_read_text,
        ) = _install_fake_filesystem()

    def teardown_method(self) -> None:
        """Restore the real ``_test_hooks`` bindings."""
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.read_text = self._original_read_text

    def test_aggregates_across_multiple_files(self) -> None:
        """Each path's violations are concatenated in scan order."""
        a = Path("a.py")
        b = Path("b.py")
        self._fake.write(
            a,
            ("from typing import TypedDict\nclass A(TypedDict):\n    unk1: int\n"),
        )
        self._fake.write(
            b,
            ("from typing import TypedDict\nclass B(TypedDict):\n    padding0: int\n"),
        )
        violations = find_violations((a, b))
        assert [v.path for v in violations] == [a, b]
        assert [v.field_name for v in violations] == ["unk1", "padding0"]

    def test_raises_filenotfound_when_path_missing(self) -> None:
        """A missing path raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            find_violations((Path("missing.py"),))


class TestViolationFormat:
    """Tests for the violation formatter."""

    def test_format_includes_path_line_and_field(self) -> None:
        """The formatted line carries every diagnostic field."""
        violation = Violation(
            path=Path("src/dummy.py"),
            line_no=42,
            typed_dict_name="ShootEventDict",
            field_name="unk1",
        )
        formatted = violation.format()
        assert "src" in formatted and "dummy.py" in formatted
        assert ":42:" in formatted
        assert "ShootEventDict.unk1" in formatted


class TestProductionSourcesAreClean:
    """The current steady state: every wire-format TypedDict is clean."""

    def test_default_targets_have_zero_violations(self) -> None:
        """Production scan against ``DEFAULT_TARGETS`` returns no violations.

        This is the regression gate: any future field named ``unk\\d+``,
        ``unknown_byte*``, ``padding*``, or ``reserved\\d*`` in
        ``protocol/types.py`` or ``container/types.py`` will fail this
        test before reaching review.
        """
        violations = find_violations(DEFAULT_TARGETS)
        if violations:
            formatted = "\n".join("  " + v.format() for v in violations)
            raise AssertionError(
                f"production sources have {len(violations)} undecoded-field "
                f"violations:\n{formatted}"
            )


class TestRunEntrypoint:
    """Tests for ``run()`` exit codes and CLI output."""

    def setup_method(self) -> None:
        """Install the fake filesystem."""
        (
            self._fake,
            self._original_path_exists,
            self._original_read_text,
        ) = _install_fake_filesystem()

    def teardown_method(self) -> None:
        """Restore the real ``_test_hooks`` bindings."""
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.read_text = self._original_read_text

    def test_returns_zero_when_clean(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A clean scan exits 0 with a friendly stdout summary."""
        a = Path("a.py")
        self._fake.write(
            a,
            ("from typing import TypedDict\nclass A(TypedDict):\n    aim_x: int\n"),
        )
        assert run((a,)) == 0
        out = capsys.readouterr().out
        assert "clean" in out
        assert "1 files scanned" in out

    def test_returns_one_on_violation_and_writes_to_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A violation exits 1 and every line is printed to stderr."""
        a = Path("a.py")
        self._fake.write(
            a,
            ("from typing import TypedDict\nclass A(TypedDict):\n    unk1: int\n"),
        )
        assert run((a,)) == 1
        err = capsys.readouterr().err
        assert "1 violation" in err
        assert "A.unk1" in err


class TestMain:
    """Tests for ``main()`` and the ``__main__`` runpy entrypoint."""

    def test_main_exits_with_run_code(self) -> None:
        """``main()`` propagates ``run()``'s exit code via ``SystemExit``."""
        with pytest.raises(SystemExit) as exc:
            guard.main()
        assert exc.value.code == 0

    def test_module_entrypoint_runs_main(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``python -m scripts.check_undecoded_fields`` executes ``main``."""
        old_argv = sys.argv
        sys.argv = ["scripts.check_undecoded_fields"]
        try:
            sys.modules.pop("scripts.check_undecoded_fields", None)
            with pytest.raises(SystemExit) as exc:
                runpy.run_module("scripts.check_undecoded_fields", run_name="__main__")
            assert exc.value.code == 0
        finally:
            sys.argv = old_argv
        out = capsys.readouterr().out
        assert "clean" in out
