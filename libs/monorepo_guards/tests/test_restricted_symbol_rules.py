"""Tests for the owner-restricted symbol guard rule."""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.restricted_symbol_rules import RESTRICTED_SYMBOLS, RestrictedSymbolRule


def _write(tmp_path: Path, relative: str, content: str) -> Path:
    """Write a source file at a repo-shaped relative path.

    Args:
        tmp_path: Test-scoped root directory.
        relative: Forward-slash relative path for the file.
        content: File content.

    Returns:
        The written file's path.
    """
    target = tmp_path / Path(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


class TestRestrictedSymbolRule:
    """The raw lock clear is referencable only inside its owner."""

    def test_import_outside_the_owner_is_flagged(self, tmp_path: Path) -> None:
        offender = _write(
            tmp_path,
            "src/tankpit_bot/bot/ai/quad_sweep.py",
            "from tankpit_bot.bot.ai.intent import clear_resource_target\n",
        )

        violations = RestrictedSymbolRule().run([offender])

        assert len(violations) == 1
        assert violations[0].kind == "restricted-symbol-clear_resource_target"
        assert violations[0].file == offender
        assert violations[0].line_no == 1

    def test_call_outside_the_owner_is_flagged(self, tmp_path: Path) -> None:
        offender = _write(
            tmp_path,
            "src/tankpit_bot/bot/ai/forage.py",
            "def f(state):\n    return clear_resource_target(state)\n",
        )

        violations = RestrictedSymbolRule().run([offender])

        assert len(violations) == 1
        assert violations[0].kind == "restricted-symbol-clear_resource_target"
        assert violations[0].line_no == 2

    def test_attribute_access_outside_the_owner_is_flagged(self, tmp_path: Path) -> None:
        offender = _write(
            tmp_path,
            "src/tankpit_bot/bot/ai/helper.py",
            "from tankpit_bot.bot.ai import intent\n"
            "\n"
            "def f(state):\n"
            "    return intent.clear_resource_target(state)\n",
        )

        violations = RestrictedSymbolRule().run([offender])

        assert len(violations) == 1
        assert violations[0].kind == "restricted-symbol-clear_resource_target"
        assert violations[0].line_no == 4

    def test_the_owner_module_may_define_and_call_it(self, tmp_path: Path) -> None:
        owner = _write(
            tmp_path,
            "src/tankpit_bot/bot/ai/intent.py",
            "def clear_resource_target(state):\n"
            "    return state\n"
            "\n"
            "def release_collect_plan(state):\n"
            "    return clear_resource_target(state)\n",
        )

        assert RestrictedSymbolRule().run([owner]) == []

    def test_a_test_file_is_not_exempt(self, tmp_path: Path) -> None:
        offender = _write(
            tmp_path,
            "tests/bot/ai/test_intent.py",
            "from tankpit_bot.bot.ai.intent import clear_resource_target\n",
        )

        violations = RestrictedSymbolRule().run([offender])

        assert len(violations) == 1
        assert violations[0].kind == "restricted-symbol-clear_resource_target"

    def test_prose_mentions_never_trip_the_rule(self, tmp_path: Path) -> None:
        clean = _write(
            tmp_path,
            "src/tankpit_bot/bot/ai/quad_sweep.py",
            '"""The clear_resource_target call was removed 2026-09-02."""\n'
            "# clear_resource_target used to be called here\n"
            "VALUE = 1\n",
        )

        assert RestrictedSymbolRule().run([clean]) == []

    def test_unrestricted_symbols_pass_everywhere(self, tmp_path: Path) -> None:
        clean = _write(
            tmp_path,
            "src/tankpit_bot/bot/ai/forage.py",
            "from tankpit_bot.bot.ai.intent import release_collect_plan\n"
            "def f(state):\n"
            "    return release_collect_plan(state)\n",
        )

        assert RestrictedSymbolRule().run([clean]) == []

    def test_the_table_names_the_lock_clear_and_its_owner(self) -> None:
        assert RESTRICTED_SYMBOLS["clear_resource_target"] == ("bot/ai/intent.py",)
