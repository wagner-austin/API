"""Tests for mock ban guard rule."""

from __future__ import annotations

from pathlib import Path

from scripts.mock_ban_rule import MockBanRule


class TestMockBanRule:
    """Tests for MockBanRule."""

    def test_detects_import_mock(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("import mock\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "import-mock"

    def test_detects_import_unittest_mock(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("import unittest.mock\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "import-mock"

    def test_detects_from_unittest_mock_import(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("from unittest.mock import patch\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "import-mock"

    def test_detects_from_unittest_import_mock(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("from unittest import mock\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "import-mock"

    def test_allows_normal_imports(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("import pytest\nfrom pathlib import Path\n")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_syntax_error_raises(self, tmp_path: Path) -> None:
        import pytest

        test_file = tmp_path / "test.py"
        test_file.write_text("import [\n", encoding="utf-8")

        rule = MockBanRule()
        with pytest.raises(RuntimeError) as exc_info:
            rule.run([test_file])
        assert "failed to parse" in str(exc_info.value)

    def test_handles_nonexistent_file(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nonexistent.py"

        rule = MockBanRule()
        violations = rule.run([nonexistent])

        # Read errors are silently ignored, returns no violations
        assert len(violations) == 0

    def test_ignores_relative_import(self, tmp_path: Path) -> None:
        """Test that relative imports (module is None) are ignored.

        Covers line 78 where node.module is None for ImportFrom.
        """
        test_file = tmp_path / "test.py"
        test_file.write_text("from . import something\n", encoding="utf-8")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_allows_multiple_unittest_imports_without_mock(self, tmp_path: Path) -> None:
        """Test that multiple imports from unittest without mock are allowed.

        Covers branch 91->90 where the for loop iterates multiple times
        but none of the aliases is 'mock'.
        """
        test_file = tmp_path / "test.py"
        test_file.write_text("from unittest import TestCase, TestLoader\n", encoding="utf-8")

        rule = MockBanRule()
        violations = rule.run([test_file])

        assert len(violations) == 0
