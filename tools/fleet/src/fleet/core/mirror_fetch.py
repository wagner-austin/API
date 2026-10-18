"""Fetching what a job exports into the hub's bare mirror of its remote.

The fetch half of :mod:`fleet.core.export`: a project's commit
(:func:`fetch_commit`) and a companion's ref (:func:`fetch_ref`), each from
the remote the registry declares, into the mirror the archive or the bundle
is then read out of.

A LOST FETCH IS SENT AGAIN BEFORE THE JOB IS REFUSED (MCPs board task
dfcdfb48). Until then one non-zero fetch refused the job ``EXPORT_FAILED``
and the submitter had to notice and resubmit, though nothing was wrong with
the commit. ``dispatch_list(status=refused)`` on 2026-10-18 held six in three
days: on diphtheria ``gnutls_handshake() failed: The TLS connection was
non-properly terminated`` (job 5aadc355, which passed as b083f28b at the
same sha), on austinpc ``RPC failed; curl 56 Recv failure: Connection was
reset`` four times in one afternoon, and once ``timed out after 600 s``. So
each fetch is sent up to :data:`FETCH_ATTEMPTS` times, every attempt within
one shared :data:`FETCH_BUDGET_SECONDS`, which bounds the export as a single
fetch's deadline did: the claim path it runs in has a node runner's whole
serve behind it, and a wait of three full deadlines would outlast it. A
remote that answers it does not serve the object is not lost: that refusal
is the commit's or the registry's, and it is raised at once.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Final

from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks

#: The local ref a companion's declared ref is fetched into, inside that
#: companion's own mirror.
#:
#: Fetched into a ref rather than read off ``FETCH_HEAD``, so the commit is
#: REFERENCED between the fetch and the bundle: an unreferenced tip is
#: exactly what ``git gc`` is entitled to remove, and the window would be
#: rare enough to be undebuggable.
COMPANION_REF: Final = "refs/fleet/companion"

#: One fetch attempt's deadline, in seconds. A first fetch of a large
#: repository over the operator's uplink is minutes; ten holds it and ends a
#: hung remote before the tick's own bound does.
FETCH_TIMEOUT_SECONDS: Final = 600

#: How many times one fetch is sent before the job is refused.
FETCH_ATTEMPTS: Final = 3

#: The time every attempt of one fetch shares, in seconds: a lost connection
#: is sent again twice, a fetch that timed out at its full deadline once more
#: with what is left.
FETCH_BUDGET_SECONDS: Final = 900

#: The pause before a lost fetch is sent again, in seconds; no attempt is
#: sent when no more than this is left of the budget.
FETCH_PAUSE_SECONDS: Final = 10

#: The deadline of a git command that reads only the local mirror, in seconds.
INIT_TIMEOUT_SECONDS: Final = 60

#: What git says when the remote will not serve the object asked for: the
#: three spellings across protocol versions and server settings. Any of
#: them means the sha is not fetchable from the remote, which for a project
#: on GitHub means it was never pushed.
NOT_ON_REMOTE_MARKERS: Final = (
    "couldn't find remote ref",
    "unadvertised object",
    "not our ref",
)


def _words(stderr: str) -> str:
    """git's own words from one command, bounded.

    Args:
        stderr: What the command wrote to standard error.

    Returns:
        The first 600 characters, stripped, or a note that it wrote nothing.
    """
    return stderr.strip()[:600] if stderr.strip() else "(git wrote nothing to stderr)"


def git_failure(code: FleetErrorCode, verb: str, stderr: str) -> AppError[FleetErrorCode]:
    """Compose the refusal for a git command that exited non-zero.

    Args:
        code: The refusal's code.
        verb: What was being done, for the message.
        stderr: git's own words, carried verbatim and bounded.

    Returns:
        The error to raise.
    """
    return AppError(code=code, message=f"{verb}: {_words(stderr)}")


def _fetched(
    argv: tuple[str, ...], *, verb: str, not_served: Callable[[str], AppError[FleetErrorCode]]
) -> None:
    """Send one fetch until it is done, the remote refuses it, or its attempts run out.

    Args:
        argv: The ``git fetch`` command.
        verb: What is fetched where, for the refusal.
        not_served: The refusal for a remote that does not serve what was
            asked for, given git's words.

    Raises:
        AppError: ``not_served``'s refusal, at the first attempt git answers
            with one of :data:`NOT_ON_REMOTE_MARKERS`; ``EXPORT_FAILED``
            naming every attempt's words when none was done.
    """
    deadline = _test_hooks.now() + FETCH_BUDGET_SECONDS
    said: list[str] = []
    while True:
        timeout = min(FETCH_TIMEOUT_SECONDS, deadline - _test_hooks.now())
        fetched = _test_hooks.run(argv, timeout_seconds=timeout)
        if fetched["returncode"] == 0:
            return
        stderr = fetched["stderr"]
        if any(marker in stderr for marker in NOT_ON_REMOTE_MARKERS):
            raise not_served(stderr)
        said.append(f"attempt {len(said) + 1}: {_words(stderr)}")
        if len(said) == FETCH_ATTEMPTS or deadline - _test_hooks.now() <= FETCH_PAUSE_SECONDS:
            break
        _test_hooks.sleep(FETCH_PAUSE_SECONDS)
    raise AppError(
        code=FleetErrorCode.EXPORT_FAILED,
        message=(
            f"{verb}: no attempt of {len(said)} within {FETCH_BUDGET_SECONDS} s was done; "
            + "; ".join(said)
        ),
    )


def has_commit(mirror: pathlib.Path, sha: str) -> bool:
    """Whether the mirror already holds the commit.

    Args:
        mirror: The mirror's path.
        sha: The commit.

    Returns:
        True iff ``git cat-file -e <sha>^{{commit}}`` exits 0.
    """
    probe = _test_hooks.run(
        ("git", "-C", str(mirror), "cat-file", "-e", f"{sha}^{{commit}}"),
        timeout_seconds=INIT_TIMEOUT_SECONDS,
    )
    return probe["returncode"] == 0


def fetch_commit(mirror: pathlib.Path, remote: str, sha: str) -> None:
    """Fetch one commit from the remote into the mirror.

    Skipped when the mirror already holds it: a second job for the same sha
    (a re-run, or two packages of one commit) costs no round trip.

    Args:
        mirror: The mirror's path, already initialised.
        remote: The project's declared remote.
        sha: The commit, forty hex.

    Raises:
        AppError: ``SHA_NOT_ON_REMOTE`` when the remote will not serve the
            object (one of :data:`NOT_ON_REMOTE_MARKERS` in git's words),
            ``EXPORT_FAILED`` when no attempt was done, every attempt's
            words verbatim.
    """
    if has_commit(mirror, sha):
        return

    def not_served(stderr: str) -> AppError[FleetErrorCode]:
        return AppError(
            code=FleetErrorCode.SHA_NOT_ON_REMOTE,
            message=(
                f"{remote} does not serve {sha}; a commit the remote has never seen is one "
                f"git push away from checkable, and this runner exports only what the remote "
                f"serves. git said: {stderr.strip()[:300]}"
            ),
        )

    _fetched(
        ("git", "-C", str(mirror), "fetch", "--quiet", "--no-tags", remote, sha),
        verb=f"git fetch {remote} {sha} into {mirror}",
        not_served=not_served,
    )


def fetch_ref(mirror: pathlib.Path, remote: str, ref: str) -> str:
    """Fetch a ref's tip into the mirror and answer the commit it names.

    Never skipped the way :func:`fetch_commit` is: a sha the mirror holds is
    that sha forever, while a ref's tip is a moving answer and the question
    a companion asks is what it points at NOW.

    Args:
        mirror: The companion's mirror, already initialised.
        remote: The companion's declared remote.
        ref: The declared ref, ``main`` or ``refs/heads/main``.

    Returns:
        The forty-hex commit the ref names.

    Raises:
        AppError: ``COMPANION_REF_NOT_ON_REMOTE`` when the remote does not
            serve the ref (one of :data:`NOT_ON_REMOTE_MARKERS` in git's
            words), which is a registry line to correct rather than a
            commit to push; ``EXPORT_FAILED`` when no attempt was done, or
            for a fetched ref that does not resolve to a commit, git's
            words verbatim.
    """

    def not_served(stderr: str) -> AppError[FleetErrorCode]:
        return AppError(
            code=FleetErrorCode.COMPANION_REF_NOT_ON_REMOTE,
            message=(
                f"{remote} does not serve {ref!r}, which the registry declares as a "
                f"companion ref; every job of the project carrying it fails here until "
                f"that line names a ref the remote has. git said: {stderr.strip()[:300]}"
            ),
        )

    _fetched(
        (
            "git",
            "-C",
            str(mirror),
            "fetch",
            "--quiet",
            "--no-tags",
            "--force",
            remote,
            f"+{ref}:{COMPANION_REF}",
        ),
        verb=f"git fetch {remote} {ref} into {mirror}",
        not_served=not_served,
    )
    resolved = _test_hooks.run(
        ("git", "-C", str(mirror), "rev-parse", f"{COMPANION_REF}^{{commit}}"),
        timeout_seconds=INIT_TIMEOUT_SECONDS,
    )
    if resolved["returncode"] != 0:
        raise git_failure(
            FleetErrorCode.EXPORT_FAILED,
            f"git rev-parse {COMPANION_REF} in {mirror}",
            resolved["stderr"],
        )
    return resolved["stdout"].strip()


__all__ = [
    "COMPANION_REF",
    "FETCH_ATTEMPTS",
    "FETCH_BUDGET_SECONDS",
    "FETCH_PAUSE_SECONDS",
    "FETCH_TIMEOUT_SECONDS",
    "INIT_TIMEOUT_SECONDS",
    "NOT_ON_REMOTE_MARKERS",
    "fetch_commit",
    "fetch_ref",
    "git_failure",
    "has_commit",
]
