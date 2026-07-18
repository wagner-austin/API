"""Tests for the contract-enforcement guard rule."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.contract_rules import GUARDED_PACKAGES, MUTATION_PREFIXES, run_contract_rules


def _write_module(project_root: Path, package: str, name: str, body: str) -> Path:
    """Create a module inside a guarded package of a fake project tree.

    Args:
        project_root: Fake project root.
        package: Guarded package name (e.g. ``facts``).
        name: Module filename.
        body: Module source text.

    Returns:
        Path to the created module.
    """
    package_root = project_root / "src" / "tankpit_bot" / package
    package_root.mkdir(parents=True, exist_ok=True)
    module_path = package_root / name
    module_path.write_text(body, encoding="utf-8")
    return module_path


def test_missing_packages_yield_zero_violations(tmp_path: Path) -> None:
    """A tree with no guarded packages passes."""
    assert run_contract_rules(tmp_path) == 0


def test_unenforced_public_mutation_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A public apply_* without the decorator is a violation."""
    module_path = _write_module(
        tmp_path,
        "facts",
        "mutators.py",
        "def apply_observation(*, value: int) -> int:\n    return value\n",
    )
    assert run_contract_rules(tmp_path) == 1
    out = capsys.readouterr().out
    assert f"contract_rule_violation {module_path}:1" in out
    assert "'apply_observation' lacks @enforce_contract" in out


def test_decorated_mutation_passes(tmp_path: Path) -> None:
    """@enforce_contract(...) on the mutation satisfies the rule."""
    _write_module(
        tmp_path,
        "facts",
        "mutators.py",
        "@enforce_contract(SomeContract())\n"
        "def apply_observation(*, value: int) -> int:\n"
        "    return value\n",
    )
    assert run_contract_rules(tmp_path) == 0


def test_bare_and_attribute_decorators_are_recognized(tmp_path: Path) -> None:
    """Bare and module-qualified decorator forms both satisfy the rule."""
    _write_module(
        tmp_path,
        "ledger",
        "bare.py",
        "@enforce_contract\ndef record_outcome(*, value: int) -> int:\n    return value\n",
    )
    _write_module(
        tmp_path,
        "memory",
        "qualified.py",
        "@enforcement.enforce_contract(SomeContract())\n"
        "def update_belief(*, value: int) -> int:\n"
        "    return value\n",
    )
    assert run_contract_rules(tmp_path) == 0


def test_other_decorators_do_not_satisfy_the_rule(tmp_path: Path) -> None:
    """A different decorator still leaves the mutation unenforced."""
    _write_module(
        tmp_path,
        "facts",
        "mutators.py",
        "@functools.cache\ndef set_belief(*, value: int) -> int:\n    return value\n",
    )
    assert run_contract_rules(tmp_path) == 1


def test_private_and_non_mutation_functions_are_ignored(tmp_path: Path) -> None:
    """Private helpers and non-mutation names are out of scope."""
    _write_module(
        tmp_path,
        "facts",
        "helpers.py",
        "CONSTANT = 1\n"
        "def _apply_helper(*, value: int) -> int:\n"
        "    return value\n"
        "def make_fact(*, value: int) -> int:\n"
        "    return value\n"
        "def decode_fact(*, value: int) -> int:\n"
        "    return value\n",
    )
    assert run_contract_rules(tmp_path) == 0


def test_async_mutations_are_scanned(tmp_path: Path) -> None:
    """Async mutation functions are held to the same rule."""
    _write_module(
        tmp_path,
        "facts",
        "async_mutators.py",
        "async def mutate_world(*, value: int) -> int:\n    return value\n",
    )
    assert run_contract_rules(tmp_path) == 1


def test_every_mutation_prefix_is_guarded(tmp_path: Path) -> None:
    """Each declared prefix triggers the rule when unenforced."""
    lines = [
        f"def {prefix}example_{index}(*, value: int) -> int:\n    return value\n"
        for index, prefix in enumerate(MUTATION_PREFIXES)
    ]
    _write_module(tmp_path, "facts", "all_prefixes.py", "".join(lines))
    assert run_contract_rules(tmp_path) == len(MUTATION_PREFIXES)


def test_guarded_packages_are_facts_ledger_memory() -> None:
    """The guard covers the three architecture packages."""
    assert GUARDED_PACKAGES == ("facts", "ledger", "memory")
