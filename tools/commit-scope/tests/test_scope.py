"""The decision itself, over the shapes that caused the two incidents."""

from __future__ import annotations

from commit_scope.contracts import ScopeQuestion
from commit_scope.scope import covers, decide, refuses


class TestCovers:
    """Whether one declared entry covers one staged path."""

    def test_exact_file(self) -> None:
        """The simplest declaration: one file."""
        assert covers("libs/a.py", "libs/a.py")

    def test_directory_prefix(self) -> None:
        """A declared directory covers what is under it."""
        assert covers("libs/platform_core/src/x.py", "libs/platform_core")

    def test_sibling_package_is_not_covered(self) -> None:
        """The separator is load-bearing.

        Without it, declaring ``libs/platform_core`` would silently cover
        ``libs/platform_core_extras`` -- a different package, and exactly the
        kind of neighbour a sweep picks up.
        """
        assert not covers("libs/platform_core_extras/x.py", "libs/platform_core")

    def test_unrelated_path_is_not_covered(self) -> None:
        """A declaration does not cover the rest of the tree."""
        assert not covers("tools/hpc3/x.py", "libs/platform_core")


class TestDecideUndeclared:
    """No declaration is not an empty declaration."""

    def test_reports_undeclared_and_accuses_nothing(self) -> None:
        """With no statement of intent there is nothing to be outside of."""
        question: ScopeQuestion = {"staged": ("a.py", "b.py"), "scope": ()}
        decision = decide(question)
        assert not decision["declared"]
        assert decision["out_of_scope"] == ()
        assert decision["staged"] == ("a.py", "b.py")

    def test_never_refuses(self) -> None:
        """Refusing every undeclared commit would block every human here."""
        assert not refuses(decide({"staged": ("a.py",), "scope": ()}))


class TestDecideDeclared:
    """The measured sweep, and its resolution."""

    def test_names_the_swept_paths_and_refuses(self) -> None:
        """Reproduces 09d2a04b: another session's files in a shared index."""
        question: ScopeQuestion = {
            "staged": (
                "tools/commit-scope/src/commit_scope/scope.py",
                "clients/TankpitBot/src/tankpit_bot/bot/base.py",
                "clients/TankpitBot/src/tankpit_bot/bot/config.py",
            ),
            "scope": ("tools/commit-scope",),
        }
        decision = decide(question)
        assert refuses(decision)
        assert decision["out_of_scope"] == (
            "clients/TankpitBot/src/tankpit_bot/bot/base.py",
            "clients/TankpitBot/src/tankpit_bot/bot/config.py",
        )

    def test_accepts_once_the_sweep_is_excluded(self) -> None:
        """The same commit, staged as its author intended."""
        question: ScopeQuestion = {
            "staged": ("tools/commit-scope/src/commit_scope/scope.py",),
            "scope": ("tools/commit-scope",),
        }
        assert not refuses(decide(question))

    def test_reports_a_declared_entry_that_matched_nothing(self) -> None:
        """A typo is visible without being fatal."""
        question: ScopeQuestion = {"staged": ("a.py",), "scope": ("a.py", "typo/b.py")}
        decision = decide(question)
        assert decision["unmatched"] == ("typo/b.py",)

    def test_an_unmatched_entry_alone_does_not_refuse(self) -> None:
        """A superfluous entry cannot admit anything, so it cannot fail open."""
        question: ScopeQuestion = {"staged": ("a.py",), "scope": ("a.py", "typo/b.py")}
        assert not refuses(decide(question))

    def test_a_declared_commit_with_an_empty_index_is_accepted(self) -> None:
        """Nothing staged cannot be out of scope, however narrow the scope."""
        decision = decide({"staged": (), "scope": ("libs/platform_core",)})
        assert not refuses(decision)
        assert decision["unmatched"] == ("libs/platform_core",)

    def test_multiple_entries_are_a_union(self) -> None:
        """Two declared trees admit paths from either."""
        question: ScopeQuestion = {
            "staged": ("libs/a/x.py", "tools/b/y.py"),
            "scope": ("libs/a", "tools/b"),
        }
        assert not refuses(decide(question))
