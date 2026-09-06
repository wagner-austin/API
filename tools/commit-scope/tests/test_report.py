"""What the author reads, asserted rather than eyeballed."""

from __future__ import annotations

from commit_scope.contracts import ScopeDecision
from commit_scope.report import render


def _decision(
    *,
    declared: bool,
    staged: tuple[str, ...],
    out_of_scope: tuple[str, ...] = (),
    unmatched: tuple[str, ...] = (),
) -> ScopeDecision:
    """Build a decision without going through the decider.

    Args:
        declared: Whether a scope was declared.
        staged: The staged set.
        out_of_scope: Paths outside the declaration.
        unmatched: Declared entries nothing matched.

    Returns:
        The decision.
    """
    return {
        "declared": declared,
        "staged": staged,
        "out_of_scope": out_of_scope,
        "unmatched": unmatched,
    }


class TestReceipt:
    """The undeclared case must show what is about to ship."""

    def test_lists_every_staged_path_with_a_count(self) -> None:
        """Both incidents were visible at this moment and unlooked at.

        The paths are asserted as a contiguous block in staged order, not by
        membership: a receipt that printed them in some other order, or
        interleaved with prose, would still contain each line while being
        unreadable at the moment it matters.
        """
        lines = render(_decision(declared=False, staged=("a.py", "b/c.py")))
        assert lines[:3] == (
            "=== commit-scope: staging receipt (2 path(s)) ===",
            "    a.py",
            "    b/c.py",
        )

    def test_says_nothing_is_enforced(self) -> None:
        """It is a narrowing, and must not read as a close."""
        text = "\n".join(render(_decision(declared=False, staged=("a.py",))))
        assert "Nothing was declared, so nothing is enforced" in text

    def test_names_both_ways_to_get_protection(self) -> None:
        """The declaration, and git's own native close."""
        text = "\n".join(render(_decision(declared=False, staged=("a.py",))))
        assert "COMMIT_SCOPE" in text
        assert "git commit -- <paths>" in text


class TestAccepted:
    """A declared commit that may proceed."""

    def test_is_one_line_when_everything_matched(self) -> None:
        """A passing check must not add noise to every commit."""
        lines = render(_decision(declared=True, staged=("a.py",)))
        assert lines == ("=== commit-scope: staged scope OK (1 path(s)) ===",)

    def test_reports_an_unmatched_entry_as_non_fatal(self) -> None:
        """A typo is surfaced, and named as not fatal so it is not feared."""
        text = "\n".join(
            render(_decision(declared=True, staged=("a.py",), unmatched=("typo/b.py",)))
        )
        assert "likely a typo, not fatal" in text
        assert "    typo/b.py" in text


class TestRefusal:
    """The refusal has to survive being read mid-commit."""

    def test_names_every_intruding_path(self) -> None:
        """The author needs to see whose work is in their index."""
        text = "\n".join(
            render(
                _decision(
                    declared=True,
                    staged=("mine.py", "theirs.py"),
                    out_of_scope=("theirs.py",),
                )
            )
        )
        assert "COMMIT BLOCKED" in text
        assert "    theirs.py" in text

    def test_does_not_accuse_the_author_s_own_path(self) -> None:
        """Listing the author's own file would obscure the finding."""
        lines = render(
            _decision(
                declared=True,
                staged=("mine.py", "theirs.py"),
                out_of_scope=("theirs.py",),
            )
        )
        assert "    mine.py" not in lines

    def test_forbids_the_tempting_wrong_fix(self) -> None:
        """`git add`-ing the intruders converts a near miss into the incident."""
        text = "\n".join(render(_decision(declared=True, staged=("a.py",), out_of_scope=("a.py",))))
        assert "Do NOT `git add` them" in text

    def test_carries_the_amend_prohibition(self) -> None:
        """--only does not save an amend, and that is not obvious."""
        text = "\n".join(render(_decision(declared=True, staged=("a.py",), out_of_scope=("a.py",))))
        assert "Never --amend in a shared tree" in text
