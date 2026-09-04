"""A command that exits 0 having done nothing, refused for every package at once.

The failure is specific and was hit three times. `python -m <pkg>.cli.<cmd>`
on a module that defines ``entrypoint`` and carries no ``__main__`` block
imports it, defines its functions and exits **0**. No error, no output, no
file. It is indistinguishable from a run that legitimately produced nothing,
which is a state several of these commands can genuinely reach.

``hpc3`` had it in all twelve commands. ``model_trainer`` had it. Both added a
per-package test. ``code_style_eval`` had neither, and it cost a scoring run
over 226 files that an A30 had just spent thirty-three minutes generating.

Three misses is the argument for the rule living here rather than in each
package's own suite.
"""

from __future__ import annotations

import pathlib

from monorepo_guards.entrypoint_rules import EntrypointRule

_ENTRYPOINT = "def entrypoint() -> None:\n    raise SystemExit(main())\n"
_MAIN = "def main() -> int:\n    return 0\n"


def _cli(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    """Write one module into a ``cli`` directory.

    Args:
        tmp_path: The test's temporary directory.
        name: The module's filename.
        body: Its source.

    Returns:
        The written file.
    """
    directory = tmp_path / "cli"
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def _kinds(paths: list[pathlib.Path]) -> list[str]:
    """Run the rule and report the violation kinds it found.

    Args:
        paths: Files to check.

    Returns:
        One kind per violation, in file order.
    """
    return [violation.kind for violation in EntrypointRule().run(paths)]


class TestTheDefectItExistsFor:
    """A command with no main guard does nothing when run as a module."""

    def test_a_command_with_no_main_guard_is_refused(self, tmp_path: pathlib.Path) -> None:
        path = _cli(tmp_path, "evaluate.py", _MAIN + _ENTRYPOINT)

        assert _kinds([path]) == ["entrypoint-unguarded"]

    def test_the_refusal_says_what_running_it_would_do(self, tmp_path: pathlib.Path) -> None:
        """ "Unguarded" means nothing to a reader who has not hit this before."""
        path = _cli(tmp_path, "evaluate.py", _MAIN + _ENTRYPOINT)

        assert "exits 0 having done nothing" in EntrypointRule().run([path])[0].line

    def test_a_guard_calling_main_is_refused(self, tmp_path: pathlib.Path) -> None:
        """`main()` returns an exit code into nothing, so the process exits 0
        whatever the command reported -- the same silence by another route."""
        path = _cli(
            tmp_path,
            "evaluate.py",
            _MAIN + _ENTRYPOINT + 'if __name__ == "__main__":\n    main()\n',
        )

        assert _kinds([path]) == ["entrypoint-misguarded"]

    def test_a_correct_command_passes(self, tmp_path: pathlib.Path) -> None:
        path = _cli(
            tmp_path,
            "evaluate.py",
            _MAIN + _ENTRYPOINT + 'if __name__ == "__main__":\n    entrypoint()\n',
        )

        assert _kinds([path]) == []


class TestWhatItDeliberatelyIgnores:
    """The predicate is a shape, so nothing needs an exemption list."""

    def test_a_library_module_beside_a_command_needs_no_guard(self, tmp_path: pathlib.Path) -> None:
        """`record_reports` is a real example: report helpers, no entry point,
        correctly no guard. Keying on `entrypoint` lets it fall out of scope
        by its own shape rather than by being named somewhere."""
        path = _cli(tmp_path, "report_helpers.py", "def render() -> str:\n    return 'x'\n")

        assert _kinds([path]) == []

    def test_a_private_module_is_not_a_command(self, tmp_path: pathlib.Path) -> None:
        """`_test_hooks` and friends are wiring, not commands."""
        path = _cli(tmp_path, "_test_hooks.py", _MAIN + _ENTRYPOINT)

        assert _kinds([path]) == []

    def test_an_entrypoint_outside_a_cli_directory_is_not_a_command(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Plenty of modules define a function called `entrypoint` without
        being console scripts. A rule that fired on all of them would be
        answered by exemptions rather than by guards."""
        path = tmp_path / "somewhere.py"
        path.write_text(_MAIN + _ENTRYPOINT, encoding="utf-8")

        assert _kinds([path]) == []

    def test_a_nested_function_named_entrypoint_does_not_count(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The console script binds a MODULE-level name; a closure is not it."""
        path = _cli(
            tmp_path,
            "evaluate.py",
            "def outer() -> None:\n    def entrypoint() -> None:\n        return None\n",
        )

        assert _kinds([path]) == []


class TestNearMissGuards:
    """A conditional that merely LOOKS like the main guard is not one.

    Every case here produces a module that imports, defines its entry point,
    and still exits 0 doing nothing when run with `python -m` -- while
    carrying a line a reader skims straight past as the guard. That is
    strictly worse than having no guard at all, so each is refused.
    """

    def test_a_module_level_conditional_that_is_not_a_comparison(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _cli(tmp_path, "evaluate.py", _MAIN + _ENTRYPOINT + "if True:\n    entrypoint()\n")

        assert _kinds([path]) == ["entrypoint-unguarded"]

    def test_a_chained_comparison_is_not_the_guard(self, tmp_path: pathlib.Path) -> None:
        """`__name__ == "__main__" == x` is a three-term chain, and what it
        compares against is no longer readable off the first comparator."""
        path = _cli(
            tmp_path,
            "evaluate.py",
            _MAIN + _ENTRYPOINT + 'if __name__ == "__main__" == "x":\n    entrypoint()\n',
        )

        assert _kinds([path]) == ["entrypoint-unguarded"]

    def test_comparing_a_literal_rather_than_a_name(self, tmp_path: pathlib.Path) -> None:
        path = _cli(
            tmp_path,
            "evaluate.py",
            _MAIN + _ENTRYPOINT + 'if "__name__" == "__main__":\n    entrypoint()\n',
        )

        assert _kinds([path]) == ["entrypoint-unguarded"]

    def test_comparing_the_wrong_name(self, tmp_path: pathlib.Path) -> None:
        path = _cli(
            tmp_path,
            "evaluate.py",
            _MAIN + _ENTRYPOINT + 'if __file__ == "__main__":\n    entrypoint()\n',
        )

        assert _kinds([path]) == ["entrypoint-unguarded"]

    def test_comparing_against_a_name_rather_than_the_string(self, tmp_path: pathlib.Path) -> None:
        """An indirection here is unresolvable by parsing, and a guard that
        needed imports to evaluate would not be a guard."""
        path = _cli(
            tmp_path,
            "evaluate.py",
            _MAIN + _ENTRYPOINT + "if __name__ == MAIN:\n    entrypoint()\n",
        )

        assert _kinds([path]) == ["entrypoint-unguarded"]

    def test_a_typo_in_the_dunder_is_refused(self, tmp_path: pathlib.Path) -> None:
        """The one that actually gets written. `__mian__` never equals
        `__name__`, so the block is dead and the command is silent."""
        path = _cli(
            tmp_path,
            "evaluate.py",
            _MAIN + _ENTRYPOINT + 'if __name__ == "__mian__":\n    entrypoint()\n',
        )

        assert _kinds([path]) == ["entrypoint-unguarded"]


class TestManyFiles:
    """The rule reports every offender, not the first."""

    def test_two_broken_commands_both_report(self, tmp_path: pathlib.Path) -> None:
        first = _cli(tmp_path, "evaluate.py", _MAIN + _ENTRYPOINT)
        second = _cli(tmp_path, "compare.py", _MAIN + _ENTRYPOINT)

        assert _kinds(sorted([first, second])) == [
            "entrypoint-unguarded",
            "entrypoint-unguarded",
        ]

    def test_a_clean_directory_reports_nothing(self, tmp_path: pathlib.Path) -> None:
        good = _MAIN + _ENTRYPOINT + 'if __name__ == "__main__":\n    entrypoint()\n'
        paths = [_cli(tmp_path, "evaluate.py", good), _cli(tmp_path, "compare.py", good)]

        assert _kinds(paths) == []
