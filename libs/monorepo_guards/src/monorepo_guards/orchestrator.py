from __future__ import annotations

import sys
from pathlib import Path

from monorepo_guards import Rule, RuleReport, Violation
from monorepo_guards.config import GuardConfig
from monorepo_guards.config_helpers_rules import ConfigHelpersRule
from monorepo_guards.config_loader import _decode_monorepo_guard_config
from monorepo_guards.config_rules import ConfigRule
from monorepo_guards.dataclass_rules import DataclassRule
from monorepo_guards.env_rules import EnvRule
from monorepo_guards.error_rules import ErrorsRule
from monorepo_guards.exceptions_rules import ExceptionsRule
from monorepo_guards.fake_redis_rules import FakeRedisRule
from monorepo_guards.httpx_rules import HttpxRule
from monorepo_guards.imports_rules import ImportsRule
from monorepo_guards.json_rules import JsonRule
from monorepo_guards.logging_rules import LoggingRule
from monorepo_guards.mock_rules import MockBanRule
from monorepo_guards.pattern_rules import PatternRule
from monorepo_guards.redis_rules import RedisRule
from monorepo_guards.request_context_rules import RequestContextRule
from monorepo_guards.security_rules import SecurityRule
from monorepo_guards.standardization_rules import StandardizationRule
from monorepo_guards.suppress_rules import SuppressRule
from monorepo_guards.test_quality_rules import MLTestQualityRule, WeakAssertionRule
from monorepo_guards.tests_rules import PolicyTestsRule
from monorepo_guards.typing_rules import TypingRule
from monorepo_guards.util import iter_py_files
from monorepo_guards.validation_rules import ValidationRule
from monorepo_guards.worker_imports_rules import WorkerImportsRule


def _run_with_config(config: GuardConfig) -> int:
    """Internal helper for running guards with a config object (used by tests)."""
    files = iter_py_files(config)
    rules: list[Rule] = [
        ConfigRule(),
        TypingRule(),
        ImportsRule(),
        PolicyTestsRule(),
        PatternRule(config),
        LoggingRule(),
        SuppressRule(),
        ErrorsRule(),
        ExceptionsRule(),
        EnvRule(),
        JsonRule(),
        RedisRule(),
        HttpxRule(),
        ConfigHelpersRule(),
        RequestContextRule(),
        SecurityRule(),
        StandardizationRule(),
        ValidationRule(),
        WorkerImportsRule(),
        WeakAssertionRule(),
        MLTestQualityRule(),
        FakeRedisRule(),
        MockBanRule(),
    ]
    if config.dataclass_ban_segments:
        rules.append(DataclassRule(config))

    reports: list[RuleReport] = []
    violations: list[Violation] = []

    for rule in rules:
        res = rule.run(files)
        reports.append(RuleReport(name=rule.name, violations=len(res)))
        violations.extend(res)

    out_stream = sys.stdout
    err_stream = sys.stderr

    out_stream.write("Guard rule summary:\n")
    for rep in reports:
        out_stream.write(f"  {rep.name}: {rep.violations} violations\n")

    if violations:
        err_stream.write("Guard checks failed:\n")
        for v in violations:
            text = v.line
            if len(text) > 180:
                text = text[:177] + "..."
            err_stream.write(f"  {v.file}:{v.line_no}: kind={v.kind} text={text}\n")
        return 2

    out_stream.write("Guard checks passed: no violations found.\n")
    return 0


def run_for_project(monorepo_root: Path, project_root: Path) -> int:
    config = _decode_monorepo_guard_config(monorepo_root)
    config = config._replace(root=project_root)
    return _run_with_config(config)


__all__ = ["run_for_project"]
