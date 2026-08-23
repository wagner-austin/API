from __future__ import annotations

from pathlib import Path

import pytest

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


def test_logging_rule_ignores_print_inside_a_string_literal(tmp_path: Path) -> None:
    """A mention is not a call.

    A module may legitimately carry the text of a program meant for a
    different interpreter -- a command sent over ssh, a generated script, a
    docstring showing usage. The text form of this check flagged those as
    printing from this process, which is how a remote-environment probe in
    tools/hpc3 was reported for output it never produces here.
    """
    code = 'PROBE = "import sys;print(sys.version)"\nVALUE = 1\n'
    path = tmp_path / "remote_probe.py"
    _write(path, code)

    rule = LoggingRule()
    kinds = {v.kind for v in rule.run([path])}
    assert "print" not in kinds


def test_logging_rule_ignores_print_inside_a_docstring(tmp_path: Path) -> None:
    code = '"""Usage: print(value) writes to stdout."""\nVALUE = 1\n'
    path = tmp_path / "documented.py"
    _write(path, code)

    rule = LoggingRule()
    kinds = {v.kind for v in rule.run([path])}
    assert "print" not in kinds


def test_logging_rule_still_flags_a_real_print_beside_a_mentioned_one(tmp_path: Path) -> None:
    """The fix must not have turned the check off, only made it exact."""
    code = 'PROBE = "print(1)"\npri' + "nt('actually printing')\n"
    path = tmp_path / "mixed.py"
    _write(path, code)

    rule = LoggingRule()
    prints = [v for v in rule.run([path]) if v.kind == "print"]
    assert [v.line_no for v in prints] == [2]


def test_logging_rule_does_not_flag_an_attribute_print(tmp_path: Path) -> None:
    """console.print is the rich console, not the builtin."""
    code = "console = get_console()\nconsole.print('x')\n"
    path = tmp_path / "console_user.py"
    _write(path, code)

    rule = LoggingRule()
    kinds = {v.kind for v in rule.run([path])}
    assert "print" not in kinds


def test_logging_rule_refuses_to_pass_an_unparsable_file(tmp_path: Path) -> None:
    """Reporting a file it could not read as clean is the worse failure."""
    path = tmp_path / "broken.py"
    _write(path, "def (:\n")

    rule = LoggingRule()
    with pytest.raises(RuntimeError, match="failed to parse"):
        rule.run([path])


def test_logging_rule_skips_platform_core_logging_module(tmp_path: Path) -> None:
    """Test that platform_core/logging.py is skipped from checks."""
    code = "import logging\nlogger = logging.getLogger(__name__)\n"
    path = tmp_path / "platform_core" / "src" / "platform_core" / "logging.py"
    _write(path, code)

    rule = LoggingRule()
    violations = rule.run([path])
    assert len(violations) == 0


def test_logging_rule_skips_platform_core_rich_logging_module(tmp_path: Path) -> None:
    """platform_core/rich_logging.py implements the logging layer and is skipped."""
    code = "import logging\nlogger = logging.getLogger(__name__)\n"
    path = tmp_path / "platform_core" / "src" / "platform_core" / "rich_logging.py"
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


def test_logging_rule_ignores_an_import_line_inside_a_string(tmp_path: Path) -> None:
    """Text that looks like an import is not an import.

    This is the same blind spot the print check lost, closed for the import
    scan too. A module documenting the rule it is subject to -- or building a
    command for another interpreter -- registered as importing stdlib logging.
    The malformed segment is what the old text scanner had a defensive branch
    for; nothing that parses can produce it, so both are gone.
    """
    code = 'DOC = """\nfrom logging import getLogger,  , Logger\nimport logging\n"""\nx = 1\n'
    path = tmp_path / "service.py"
    _write(path, code)

    rule = LoggingRule()
    assert rule.run([path]) == []


def test_logging_rule_ignores_a_logging_call_inside_a_string(tmp_path: Path) -> None:
    code = 'import logging\nDOC = "logging.getLogger(__name__)"\n'
    path = tmp_path / "mentioned.py"
    _write(path, code)

    rule = LoggingRule()
    # The import is real and still flagged; the call is only mentioned.
    assert [v.kind for v in rule.run([path])] == ["direct-logging-import"]


def test_logging_rule_finds_an_aliased_module_call(tmp_path: Path) -> None:
    code = "import logging as lg\nlogger = lg.getLogger(__name__)\n"
    path = tmp_path / "aliased.py"
    _write(path, code)

    rule = LoggingRule()
    kinds = {v.kind for v in rule.run([path])}
    assert kinds == {"direct-logging-import", "logging-getLogger"}


def test_logging_rule_finds_an_aliased_function_call(tmp_path: Path) -> None:
    code = "from logging import getLogger as gl\nlogger = gl(__name__)\n"
    path = tmp_path / "aliased_func.py"
    _write(path, code)

    rule = LoggingRule()
    kinds = {v.kind for v in rule.run([path])}
    assert kinds == {"from-logging-import", "logging-getLogger"}


def test_logging_rule_ignores_a_relative_logging_import(tmp_path: Path) -> None:
    """``from .logging import x`` is a local module, not the stdlib one."""
    code = "from .logging import get_logger\nlogger = get_logger(__name__)\n"
    path = tmp_path / "relative.py"
    _write(path, code)

    rule = LoggingRule()
    assert rule.run([path]) == []
