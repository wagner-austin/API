from __future__ import annotations

from pathlib import Path

from monorepo_guards.logging_rules import LoggingRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_logging_rule_flags_print_and_basicconfig(tmp_path: Path) -> None:
    code = "import logging\npri" + "nt('x')\nlogging.basic" + "Config(level=10)\n"
    path = tmp_path / "log_mod.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "print" in kinds
    assert "logging-basicConfig" in kinds


def test_logging_rule_skips_platform_core_logging_module(tmp_path: Path) -> None:
    """Test that platform_core/logging.py is skipped from checks."""
    code = "import logging\nlogger = logging.getLogger(__name__)\n"
    path = tmp_path / "platform_core" / "src" / "platform_core" / "logging.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    assert len(violations) == 0


def test_logging_rule_skips_test_logging_files(tmp_path: Path) -> None:
    """Test that test_logging.py files are skipped from checks."""
    code = "import logging\nlogger = logging.getLogger(__name__)\n"
    path = tmp_path / "tests" / "test_logging.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    assert len(violations) == 0


def test_logging_rule_flags_local_logging_module(tmp_path: Path) -> None:
    """Test that local logging.py files are flagged."""
    code = "# Local logging module\n"
    path = tmp_path / "myservice" / "logging.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    assert len(violations) == 1
    assert violations[0].kind == "local-logging-module"


def test_logging_rule_flags_direct_import_logging(tmp_path: Path) -> None:
    """Test that 'import logging' is flagged."""
    code = "import logging\nx = 1\n"
    path = tmp_path / "service.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "direct-logging-import" in kinds


def test_logging_rule_flags_from_logging_import(tmp_path: Path) -> None:
    """Test that 'from logging import' is flagged."""
    code = "from logging import Logger\nx = 1\n"
    path = tmp_path / "service.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "from-logging-import" in kinds


def test_logging_rule_flags_logging_getlogger(tmp_path: Path) -> None:
    """Test that 'logging.getLogger()' is flagged."""
    code = "import logging\nlogger = logging.getLogger(__name__)\n"
    path = tmp_path / "service.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "logging-getLogger" in kinds


def test_logging_rule_flags_all_violations_together(tmp_path: Path) -> None:
    """Test that all logging violations are detected together."""
    code = (
        "import logging\n"
        "from logging import Logger\n"
        "logger = logging.getLogger(__name__)\n"
        "print('debug')\n"
        "logging.basicConfig(level=10)\n"
    )
    path = tmp_path / "bad.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "direct-logging-import" in kinds
    assert "from-logging-import" in kinds
    assert "logging-getLogger" in kinds
    assert "print" in kinds
    assert "logging-basicConfig" in kinds


def test_logging_rule_finds_import_after_comment(tmp_path: Path) -> None:
    """Test that import logging is found even when preceded by non-matching lines."""
    code = "# This is a comment\n# Another comment\nimport logging\nx = 1\n"
    path = tmp_path / "service.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "direct-logging-import" in kinds
    assert len([v for v in violations if v.kind == "direct-logging-import"]) == 1


def test_logging_rule_finds_from_logging_after_code(tmp_path: Path) -> None:
    """Test that from logging import is found even when preceded by other lines."""
    code = "x = 1\ny = 2\nfrom logging import Logger, Handler\nz = 3\n"
    path = tmp_path / "service.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "from-logging-import" in kinds
    assert len([v for v in violations if v.kind == "from-logging-import"]) == 1


def test_logging_rule_flags_import_logging_with_alias(tmp_path: Path) -> None:
    """Test that 'import logging as log' is flagged and alias is tracked."""
    code = "import logging as log\nlog.basicConfig(level=10)\n"
    path = tmp_path / "service.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "direct-logging-import" in kinds
    assert "logging-basicConfig" in kinds


def test_logging_rule_flags_from_logging_import_with_alias(tmp_path: Path) -> None:
    """Test that 'from logging import getLogger as get_log' is flagged and alias used."""
    code = "from logging import getLogger as get_log\nlogger = get_log(__name__)\n"
    path = tmp_path / "service.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "from-logging-import" in kinds
    assert "logging-getLogger" in kinds


def test_logging_rule_handles_empty_import_segments(tmp_path: Path) -> None:
    """Test that empty segments in from imports are handled correctly."""
    code = "from logging import getLogger,  , Logger\nx = 1\n"
    path = tmp_path / "service.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "from-logging-import" in kinds
