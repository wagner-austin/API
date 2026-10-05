"""Tests for the submit-time check of a run's declared commit.

The probe is executed by a REAL bash against a REAL git repository built in
the test, with the four histories that matter: the declared commit is HEAD,
is behind HEAD, is on a branch HEAD never merged, or is absent from the clone
altogether -- which is cleargbm P6 rung 5, whose ``20d9159`` resolved to
nothing in ``/pub/wagnera3/api``. A probe asserted only as a string would pass
with a shell error in it; one that runs is checked against what git says.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.code_claim import CodeClaim
from hpc3.core.code_claim import (
    CheckoutReading,
    check_code_claim,
    explain_mismatch,
    parse_probe,
    probe_command,
)
from tests.bash_discovery import posix_bash
from tests.conftest import FakeRun

_WALL_SECONDS = 60
"""Bound on each git or bash call; every one of them answers in milliseconds."""

_HEAD = "80221ea1056e08aacd3f5ee01e6c599e166970be"
_OTHER = "20d9159a5a8c1c1d6a4c9ab0fb2b1e2b0a3c9d11"


def _git(repo: pathlib.Path, *arguments: str) -> str:
    """Run git in a repository and return its standard output.

    Args:
        repo: The repository.
        *arguments: Git's arguments.

    Returns:
        Standard output, stripped.
    """
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=_WALL_SECONDS,
    )
    return result.stdout.strip()


def _commit(repo: pathlib.Path, message: str) -> str:
    """Make an empty commit and return its full sha.

    Args:
        repo: The repository.
        message: The commit message.

    Returns:
        The new commit's sha.
    """
    _git(repo, "commit", "--allow-empty", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


class _History:
    """A checkout whose HEAD has an ancestor and an unmerged sibling."""

    def __init__(self, repo: pathlib.Path) -> None:
        """Build the history: base <- head on main, and side off base.

        Args:
            repo: An empty directory to initialise.
        """
        self.repo = repo
        _git(repo, "init", "-q", "-b", "main")
        self.base = _commit(repo, "base")
        _git(repo, "checkout", "-q", "-b", "side")
        self.side = _commit(repo, "side")
        _git(repo, "checkout", "-q", "main")
        self.head = _commit(repo, "head")

    def probe(self, commit: str) -> str:
        """Run the probe against this checkout in a real bash.

        Args:
            commit: The declared commit.

        Returns:
            The probe's standard output.
        """
        claim = CodeClaim(tree=self.repo.as_posix(), commit=commit)
        result = subprocess.run(
            [posix_bash(), "-c", probe_command(claim)],
            capture_output=True,
            text=True,
            check=True,
            timeout=_WALL_SECONDS,
        )
        return result.stdout

    def read(self, commit: str) -> CheckoutReading:
        """Run the probe and parse what it printed.

        Args:
            commit: The declared commit.

        Returns:
            The checkout's reading.
        """
        return parse_probe(self.probe(commit), CodeClaim(tree=str(self.repo), commit=commit))


class TestTheProbeAgainstARealCheckout:
    def test_head_declared_resolves_to_head(self, tmp_path: pathlib.Path) -> None:
        history = _History(tmp_path)
        reading = history.read(history.head[:7])
        assert reading == {"head": history.head, "declared": history.head, "ancestor": "0"}

    def test_an_ancestor_is_resolved_and_reported_as_one(self, tmp_path: pathlib.Path) -> None:
        history = _History(tmp_path)
        reading = history.read(history.base)
        assert reading == {"head": history.head, "declared": history.base, "ancestor": "0"}

    def test_an_unmerged_commit_is_resolved_and_reported_as_not_one(
        self, tmp_path: pathlib.Path
    ) -> None:
        history = _History(tmp_path)
        reading = history.read(history.side[:12])
        assert reading == {"head": history.head, "declared": history.side, "ancestor": "1"}

    def test_a_commit_the_clone_never_fetched_resolves_to_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Rung 5: absent is an answer about the claim, so the probe still succeeds."""
        history = _History(tmp_path)
        reading = history.read("20d9159")
        assert reading == {"head": history.head, "declared": "", "ancestor": ""}

    def test_a_missing_checkout_makes_the_probe_itself_fail(self, tmp_path: pathlib.Path) -> None:
        claim = CodeClaim(tree=(tmp_path / "absent").as_posix(), commit="80221ea")
        result = subprocess.run(
            [posix_bash(), "-c", probe_command(claim)],
            capture_output=True,
            text=True,
            check=False,
            timeout=_WALL_SECONDS,
        )
        assert result.returncode != 0
        assert result.stdout == ""

    def test_a_tree_with_a_quote_in_it_reaches_cd_as_one_argument(self) -> None:
        command = probe_command(CodeClaim(tree="/pub/it's here", commit="80221ea"))
        assert command.startswith("cd '/pub/it'\"'\"'s here' && ")


_CLAIM = CodeClaim(tree="/pub/wagnera3/api", commit="20d9159")


class TestParsingTheProbe:
    def test_a_banner_before_the_line_is_skipped(self) -> None:
        output = f"Welcome to HPC3\nhead={_HEAD} declared={_HEAD} ancestor=0\n"
        assert parse_probe(output, _CLAIM) == {
            "head": _HEAD,
            "declared": _HEAD,
            "ancestor": "0",
        }

    @pytest.mark.parametrize(
        "output",
        [
            "",
            "Welcome to HPC3\n",
            f"head={_HEAD} ancestor=0\n",
            f"head={_HEAD} declared={_HEAD}\n",
            f"head= declared={_HEAD} ancestor=0\n",
        ],
    )
    def test_output_without_the_probe_shape_is_unreadable_not_a_verdict(self, output: str) -> None:
        with pytest.raises(AppError) as caught:
            parse_probe(output, _CLAIM)
        assert caught.value.code is Hpc3ErrorCode.REPO_COMMIT_PROBE_UNREADABLE
        assert "Asking /pub/wagnera3/api about 20d9159" in caught.value.message


class TestExplainingAMismatch:
    def _explain(self, declared: str, ancestor: str) -> str:
        return explain_mismatch(
            _CLAIM, CheckoutReading(head=_HEAD, declared=declared, ancestor=ancestor)
        )

    def test_every_message_names_the_tree_both_commits_and_the_repair(self) -> None:
        message = self._explain("", "")
        assert f"repo_commit 20d9159 for /pub/wagnera3/api, but that checkout is at {_HEAD}" in (
            message
        )
        assert message.endswith("then submit again.")

    def test_an_unresolved_commit_says_the_clone_never_fetched_it(self) -> None:
        assert "resolves no single commit named 20d9159" in self._explain("", "")

    def test_an_ancestor_says_the_checkout_moved_past_it(self) -> None:
        assert "has moved past it" in self._explain(_OTHER, "0")

    def test_a_non_ancestor_says_the_checkout_does_not_contain_it(self) -> None:
        message = self._explain(_OTHER, "1")
        assert f"{_OTHER} is present but NOT in the checkout's history" in message

    def test_an_unanswered_relation_is_never_reported_as_a_no(self) -> None:
        """Exit 128 is git declining to answer; the message must not pick a side."""
        message = self._explain(_OTHER, "128")
        assert "Which way they differ is unknown" in message
        assert "exited 128" in message
        assert "NOT in the checkout's history" not in message


class TestCheckingAClaim:
    def test_a_run_making_no_claim_asks_the_cluster_nothing(self, fake_run: FakeRun) -> None:
        assert check_code_claim("hpc3", {"phase": "P6"}) is None
        assert fake_run.calls == []

    def test_a_claim_matching_head_returns_head(self, fake_run: FakeRun) -> None:
        fake_run.add("rev-parse", stdout=f"head={_HEAD} declared={_HEAD} ancestor=0\n")
        experiment = {"repo_commit": "80221ea", "repo_tree": "/pub/wagnera3/api"}

        assert check_code_claim("hpc3", experiment) == _HEAD
        assert fake_run.commands() == [
            probe_command(CodeClaim(tree="/pub/wagnera3/api", commit="80221ea"))
        ]

    def test_rung_five_is_refused(self, fake_run: FakeRun) -> None:
        fake_run.add("rev-parse", stdout=f"head={_HEAD} declared= ancestor=\n")
        experiment = {"repo_commit": "20d9159", "repo_tree": "/pub/wagnera3/api"}

        with pytest.raises(AppError) as caught:
            check_code_claim("hpc3", experiment)
        assert caught.value.code is Hpc3ErrorCode.REPO_COMMIT_NOT_HEAD
        assert "resolves no single commit named 20d9159" in caught.value.message

    def test_a_checkout_that_is_not_there_fails_as_a_remote_command(
        self, fake_run: FakeRun
    ) -> None:
        fake_run.add("rev-parse", returncode=1, stderr="cd: /pub/x: No such file or directory")
        experiment = {"repo_commit": "80221ea", "repo_tree": "/pub/x"}

        with pytest.raises(AppError) as caught:
            check_code_claim("hpc3", experiment)
        assert caught.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED
        assert "No such file or directory" in caught.value.message
