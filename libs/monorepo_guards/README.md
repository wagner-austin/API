# monorepo-guards

Strict, typed guard rules for enforcing code standards across Python projects.

## Installation

```bash
poetry add monorepo-guards --group dev
```

## Usage

Create `scripts/guard.py` in your project:

```python
from pathlib import Path
from monorepo_guards.orchestrator import run_for_project

if __name__ == "__main__":
    monorepo_root = Path(__file__).parent.parent.parent
    project_root = Path(__file__).parent.parent
    exit(run_for_project(monorepo_root, project_root))
```

Run via Makefile:

```bash
make lint
```

## Rules

| Rule | Description |
|------|-------------|
| `TypingRule` | Enforces strict typing (no `Any`, no `cast`, no `object` type) |
| `ImportsRule` | Validates import patterns and ordering |
| `LoggingRule` | Enforces structured logging standards |
| `ExceptionsRule` | Exception handling patterns |
| `DataclassRule` | Dataclass usage validation |
| `PatternRule` | Custom regex pattern matching |
| `SuppressRule` | Validates suppression directives |
| `ConfigRule` | Configuration patterns |
| `ErrorsRule` | Error handling standards |
| `EnvRule` | Environment variable usage |
| `JsonRule` | JSON handling patterns |
| `RedisRule` | Redis usage patterns |
| `HttpxRule` | HTTP client patterns |
| `SecurityRule` | Security pattern checks |
| `ValidationRule` | Input validation patterns |
| `ConfigHelpersRule` | Config helper usage |
| `RequestContextRule` | Request context patterns |
| `StandardizationRule` | Code standardization |
| `WorkerImportsRule` | Worker-specific imports |
| `PolicyTestsRule` | Test policy enforcement |
| `WeakAssertionRule` | Detects weak test assertions (is not None, isinstance, hasattr, len > 0) |
| `MLTestQualityRule` | ML test quality (loss comparisons, forward pass value checks, weight verification) |

## API

```python
from monorepo_guards import Rule, RuleReport, Violation

# Violation: (file, line_no, kind, line)
# RuleReport: (name, violations)
# Rule: Protocol with name property and run(files) method
```

## Configuration

Guards are configured via `monorepo_guard.toml` at the monorepo root. Rules can be enabled/disabled per project.

## Requirements

- Python 3.11+
- 100% test coverage enforced
