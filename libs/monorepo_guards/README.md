# monorepo-guards

Strict, typed guard rules for enforcing code standards across Python projects.

## Installation

```bash
poetry add monorepo-guards --group dev
```

## Quick Start

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

### Typing Rules

| Rule | Description |
|------|-------------|
| `TypingRule` | Enforces strict typing (no `Any`, no `cast`, no `object` type) |
| `DataclassRule` | Dataclass usage validation |

### Import Rules

| Rule | Description |
|------|-------------|
| `ImportsRule` | Validates import patterns and ordering |
| `WorkerImportsRule` | Worker-specific import patterns |

### Code Quality Rules

| Rule | Description |
|------|-------------|
| `LoggingRule` | Enforces structured logging standards |
| `ExceptionsRule` | Exception handling patterns |
| `PatternRule` | Custom regex pattern matching |
| `SuppressRule` | Validates suppression directives |
| `StandardizationRule` | Code standardization checks |

### Configuration Rules

| Rule | Description |
|------|-------------|
| `ConfigRule` | Configuration patterns |
| `ConfigHelpersRule` | Config helper usage |
| `EnvRule` | Environment variable usage |

### Dependency Rules

| Rule | Description |
|------|-------------|
| `EscapingPathDependencyRule` | Forbids a Poetry `path` dependency that resolves outside the monorepo, in any dependency group |

A path dependency is how one project here consumes another, and it is resolved
relative to the project that declares it — so nothing stops it climbing past the
monorepo root into a sibling checkout on one machine. Such a dependency builds
only there, and it stays working long after it should have been removed, because
`poetry install` does not uninstall what the lock file no longer names. Use
`poetry sync` so the environment matches the lock, and this rule so the
dependency cannot be declared in the first place.

### Error Handling Rules

| Rule | Description |
|------|-------------|
| `ErrorsRule` | Error handling standards |
| `JsonRule` | JSON handling patterns |
| `ValidationRule` | Input validation patterns |

### Infrastructure Rules

| Rule | Description |
|------|-------------|
| `RedisRule` | Redis usage patterns |
| `FakeRedisRule` | Fake Redis usage in tests |
| `HttpxRule` | HTTP client patterns |
| `RequestContextRule` | Request context patterns |
| `SecurityRule` | Security pattern checks |

### Test Quality Rules

| Rule | Description |
|------|-------------|
| `TestsRule` | Test file patterns |
| `TestQualityRule` | Test quality checks |
| `MockRule` | Mock usage patterns |
| `WeakAssertionRule` | Detects weak assertions (`is not None`, `isinstance`, `hasattr`, `len > 0`) |
| `MLTestQualityRule` | ML test quality (loss comparisons, forward pass checks, weight verification) |

### Rust Rules

| Rule | Description |
|------|-------------|
| `RustTestRule` | Requires `Result<(), ...>` return types in tests, forbids `.unwrap()` and `.expect()` |
| `RustCargoLintRule` | Enforces strict lints in `Cargo.toml` (`[lints.rust]` and `[lints.clippy]` sections) |
| `RustManualSerializeRule` | Forbids `derive(Serialize, Deserialize)`, requires manual implementation |
| `RustCoverageRule` | Enforces 100% region coverage threshold in Makefile |
| `RustProptestRule` | Requires `proptest` in dev-dependencies for property-based testing |
| `RustExplicitMatchRule` | Forbids `?` operator, requires explicit `match` expressions for full coverage |

## API

```python
from monorepo_guards import Rule, RuleReport, Violation

# Violation: (file, line_no, kind, line)
# RuleReport: (name, violations)
# Rule: Protocol with name property and run(files) method
```

### Rule Protocol

```python
from typing import Protocol
from pathlib import Path

class Rule(Protocol):
    @property
    def name(self) -> str: ...

    def run(self, files: list[Path]) -> list[Violation]: ...
```

### Violation Type

```python
from typing import NamedTuple

class Violation(NamedTuple):
    file: Path
    line_no: int
    kind: str
    line: str
```

### RuleReport Type

```python
class RuleReport(NamedTuple):
    name: str
    violations: int
```

## Configuration

Guards are configured via `monorepo_guard.toml` at the monorepo root:

```toml
[project.my-service]
enabled_rules = ["typing", "imports", "logging"]
disabled_rules = ["dataclass"]

[project.my-service.typing]
allow_any_in_tests = true

[project.my-service.imports]
forbidden_imports = ["os.system", "subprocess.call"]
```

Rules can be enabled/disabled per project.

## Custom Rules

Create custom rules by implementing the `Rule` protocol:

```python
from pathlib import Path
from monorepo_guards import Rule, Violation

class MyCustomRule:
    @property
    def name(self) -> str:
        return "my-custom-rule"

    def run(self, files: list[Path]) -> list[Violation]:
        violations = []
        for file in files:
            content = file.read_text()
            for i, line in enumerate(content.splitlines(), 1):
                if "TODO" in line:
                    violations.append(Violation(file, i, "todo-found", line))
        return violations
```

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- 100% test coverage enforced
