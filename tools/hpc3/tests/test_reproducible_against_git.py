"""The reproducibility check, driven against REAL git rather than a script.

WHY THIS FILE EXISTS, AND IT IS NOT REDUNDANT WITH ``test_reproducible.py``.
That file scripts git's answers through the ``run`` hook, which is the right
way to test what the module does with an answer -- and it is structurally
incapable of testing whether the module ASKS THE RIGHT QUESTION.

The first version of ``working_blob`` called ``git hash-object`` without
``--no-filters``. That applies the same clean filter as ``git add``, so a CRLF
working file is normalised to LF before hashing and the command returns
exactly HEAD's blob hash. The check therefore passed on the ONLY case it was
written for. Every unit test passed too, because they told git what to say.

It was caught by running the module against this repository, where it reported
three known-bad files as clean. This file is that catch, turned into
something that runs every time: a real repository, a real CRLF mismatch, and
an assertion that the filtered form would MISS it.
"""

from __future__ import annotations

import pathlib
import subprocess

from hpc3.contracts.stage import StagedFile, StageManifest
from hpc3.core.reproducible import unreproducible

_LF = b'{\n  "seed": 0\n}\n'
_CRLF = _LF.replace(b"\n", b"\r\n")


def _git(repo: pathlib.Path, *args: str) -> str:
    """Run git in a repository and return its standard output.

    Args:
        repo: Working directory for the command.
        args: Arguments following ``git``.

    Returns:
        Standard output, stripped.

    Raises:
        AssertionError: If git reports failure. A test whose setup silently
            failed would assert against a repository that does not exist.
    """
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False, text=True
    )
    assert done.returncode == 0, f"git {args}: {done.stderr}"
    return done.stdout.strip()


def _repo_with_crlf_mismatch(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a repository whose working file differs from its committed blob.

    The file is COMMITTED with LF and then rewritten on disk with CRLF, which
    is exactly the state a Windows checkout of an LF blob produces and exactly
    the state that made three of this repository's staged digests
    unreproducible.

    ``core.autocrlf`` is set explicitly rather than inherited, so the contrast
    this file asserts does not depend on the machine's global git config.

    Args:
        tmp_path: Directory to build the repository in.

    Returns:
        The repository path.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "core.autocrlf", "true")
    (repo / "spec.json").write_bytes(_LF)
    _git(repo, "add", "spec.json")
    _git(repo, "commit", "-q", "-m", "spec")
    (repo / "spec.json").write_bytes(_CRLF)
    return repo


def _manifest(name: str) -> StageManifest:
    """Build a one-file manifest.

    Args:
        name: The file to name.

    Returns:
        The manifest.
    """
    return StageManifest(
        destination="/pub/wagnera3/x",
        files=[StagedFile(name=name, sha256="0" * 64, size_bytes=1)],
        provenance={"what": "test", "why": "test"},
    )


class TestAgainstRealGit:
    """The question actually asked, not the answer handled."""

    def test_a_crlf_working_file_is_refused(self, tmp_path: pathlib.Path) -> None:
        """The motivating defect, end to end, with no scripted output.

        Args:
            tmp_path: Directory for the repository.
        """
        repo = _repo_with_crlf_mismatch(tmp_path)

        assert unreproducible(repo, _manifest("spec.json")) == ("spec.json",)

    def test_the_filtered_form_would_have_missed_it(self, tmp_path: pathlib.Path) -> None:
        """Pins WHY ``--no-filters`` is there, against the real command.

        Without the flag git normalises CRLF to LF before hashing and returns
        HEAD's own blob, so the comparison this module makes would be between
        a value and itself. This asserts the two forms genuinely disagree on
        this file -- if a future git made them agree, the check would be
        silently inert and this test is the only thing that would say so.

        Args:
            tmp_path: Directory for the repository.
        """
        repo = _repo_with_crlf_mismatch(tmp_path)

        head = _git(repo, "rev-parse", "HEAD:./spec.json")
        filtered = _git(repo, "hash-object", "spec.json")
        unfiltered = _git(repo, "hash-object", "--no-filters", "spec.json")

        assert filtered == head
        assert unfiltered != head

    def test_a_matching_working_file_passes(self, tmp_path: pathlib.Path) -> None:
        """The ordinary case, so the check is not simply always-refusing.

        Args:
            tmp_path: Directory for the repository.
        """
        repo = _repo_with_crlf_mismatch(tmp_path)
        (repo / "spec.json").write_bytes(_LF)

        assert unreproducible(repo, _manifest("spec.json")) == ()

    def test_an_untracked_file_passes(self, tmp_path: pathlib.Path) -> None:
        """Every corpus this package stages lives outside the repository.

        Args:
            tmp_path: Directory for the repository.
        """
        repo = _repo_with_crlf_mismatch(tmp_path)
        (repo / "corpus.jsonl").write_bytes(_CRLF)

        assert unreproducible(repo, _manifest("corpus.jsonl")) == ()
