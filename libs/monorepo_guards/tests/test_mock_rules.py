"""Tests for mock ban guard rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.mock_rules import MockBanRule


def _write_test_file(tmp_path: Path, content: str) -> Path:
    """Write content to a test file in a tests directory."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "test_example.py"
    test_file.write_text(content, encoding="utf-8")
    return test_file


class TestMockBanRuleImports:
    """Tests for mock import detection."""

    def test_detects_import_mock(self, tmp_path: Path) -> None:
        test_file = _write_test_file(tmp_path, "import mock\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "import-mock"
        assert violations[0].file == test_file
        assert violations[0].line_no == 1

    def test_detects_import_unittest_mock(self, tmp_path: Path) -> None:
        test_file = _write_test_file(tmp_path, "import unittest.mock\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "import-mock"

    def test_detects_from_unittest_mock_import(self, tmp_path: Path) -> None:
        test_file = _write_test_file(tmp_path, "from unittest.mock import patch\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "import-mock"

    def test_detects_from_unittest_import_mock(self, tmp_path: Path) -> None:
        test_file = _write_test_file(tmp_path, "from unittest import mock\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "import-mock"

    def test_allows_normal_imports(self, tmp_path: Path) -> None:
        test_file = _write_test_file(tmp_path, "import pytest\nfrom pathlib import Path\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_ignores_relative_import(self, tmp_path: Path) -> None:
        test_file = _write_test_file(tmp_path, "from . import something\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_allows_multiple_unittest_imports_without_mock(self, tmp_path: Path) -> None:
        test_file = _write_test_file(tmp_path, "from unittest import TestCase, TestLoader\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 0


class TestMockBanRuleMonkeypatch:
    """Tests for monkeypatch detection."""

    def test_detects_monkeypatch_setattr(self, tmp_path: Path) -> None:
        code = "def test_x(monkeypatch):\n    monkeypatch.setattr('mod', 'attr', 1)\n"
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "monkeypatch-banned"
        assert violations[0].line == "monkeypatch.setattr"

    def test_detects_monkeypatch_setenv(self, tmp_path: Path) -> None:
        code = "def test_x(monkeypatch):\n    monkeypatch.setenv('KEY', 'val')\n"
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "monkeypatch-banned"
        assert violations[0].line == "monkeypatch.setenv"

    def test_detects_monkeypatch_delattr(self, tmp_path: Path) -> None:
        code = "def test_x(monkeypatch):\n    monkeypatch.delattr('mod', 'attr')\n"
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "monkeypatch-banned"
        assert violations[0].line == "monkeypatch.delattr"

    def test_detects_monkeypatch_delenv(self, tmp_path: Path) -> None:
        code = "def test_x(monkeypatch):\n    monkeypatch.delenv('KEY')\n"
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "monkeypatch-banned"
        assert violations[0].line == "monkeypatch.delenv"

    def test_detects_monkeypatch_setitem(self, tmp_path: Path) -> None:
        code = "def test_x(monkeypatch):\n    monkeypatch.setitem(d, 'k', 'v')\n"
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "monkeypatch-banned"
        assert violations[0].line == "monkeypatch.setitem"

    def test_detects_monkeypatch_delitem(self, tmp_path: Path) -> None:
        code = "def test_x(monkeypatch):\n    monkeypatch.delitem(d, 'k')\n"
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "monkeypatch-banned"
        assert violations[0].line == "monkeypatch.delitem"

    def test_detects_monkeypatch_syspath_prepend(self, tmp_path: Path) -> None:
        code = "def test_x(monkeypatch):\n    monkeypatch.syspath_prepend('/path')\n"
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "monkeypatch-banned"
        assert violations[0].line == "monkeypatch.syspath_prepend"

    def test_allows_other_method_calls(self, tmp_path: Path) -> None:
        code = "def test_x(client):\n    client.get('/path')\n"
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_allows_non_monkeypatch_setattr(self, tmp_path: Path) -> None:
        code = "def test_x(obj):\n    obj.setattr('a', 'b')\n"
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_allows_monkeypatch_non_banned_method(self, tmp_path: Path) -> None:
        code = "def test_x(monkeypatch):\n    monkeypatch.chdir('/tmp')\n"
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 0


class TestMockBanRuleFileHandling:
    """Tests for file handling edge cases."""

    def test_ignores_non_test_files(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        src_file = src_dir / "module.py"
        src_file.write_text("import mock\n", encoding="utf-8")

        rule = MockBanRule()
        violations = rule.run([src_file])

        assert len(violations) == 0

    def test_syntax_error_raises(self, tmp_path: Path) -> None:
        test_file = _write_test_file(tmp_path, "import [\n")

        rule = MockBanRule()
        with pytest.raises(RuntimeError) as exc_info:
            rule.run([test_file])
        assert "failed to parse" in str(exc_info.value)

    def test_detects_multiple_violations(self, tmp_path: Path) -> None:
        code = (
            "import mock\n"
            "from unittest.mock import patch\n"
            "def test_x(monkeypatch):\n"
            "    monkeypatch.setattr('m', 'a', 1)\n"
        )
        test_file = _write_test_file(tmp_path, code)

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 3
        kinds = {v.kind for v in violations}
        assert "import-mock" in kinds
        assert "monkeypatch-banned" in kinds
