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

A REFUSED CREDENTIAL IS NOT LOST EITHER (MCPs board task 579a5ab0). On
2026-10-20 diphtheria's gh login was rejected, its credential helper answered
nothing, and every job of the proof of ae154507f was sent three times and
refused ``EXPORT_FAILED`` with git's ``could not read Username``, which read
as a network fault and was broadcast as the repository having gone private.
Each further attempt sends the same credential, so the first one git
answers with a :data:`CREDENTIAL_REFUSED_MARKERS` spelling is refused at once
as ``REMOTE_CREDENTIAL_REJECTED``, naming the control host's credential as
what must change.

A COMMIT IS FETCHED INTO A REF, ONE EXPORT AT A TIME PER MIRROR (MCPs board
task 86f70457). Until then :func:`fetch_commit` fetched a bare sha with no
destination, so a project mirror never held a ref: every fetch had nothing to
offer the remote as common history and received the whole repository again,
and nothing kept several node runners from fetching one commit into one mirror
at once. On diphtheria on 2026-10-22 six runners each received MCPs' 133470
objects for the same commit, the MCPs mirror held 3.5 GB with no ref and
1.6 GB of ``tmp_pack`` left by fetches that died, and the host ran at load 40
on 8 threads with its swap full. So the commit lands in :data:`FETCHED_REF`,
whose history every later fetch offers, and the probe and the fetch run
inside the mirror's :data:`FETCH_TURN`, so a second export of the same commit
waits for the first and then finds it held.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Final

from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, file_turn

#: The local ref a companion's declared ref is fetched into, inside that
#: companion's own mirror.
#:
#: Fetched into a ref rather than read off ``FETCH_HEAD``, so the commit is
#: REFERENCED between the fetch and the bundle: an unreferenced tip is
#: exactly what ``git gc`` is entitled to remove, and the window would be
#: rare enough to be undebuggable.
COMPANION_REF: Final = "refs/fleet/companion"

#: The local ref a project's commit is fetched into, inside its mirror.
#:
#: One ref, moved by each fetch, rather than one per commit: what a fetch
#: needs is a tip whose history it can offer the remote, and the commits the
#: fleet checks are on the remote's history, so the newest one carries nearly
#: all of the next one's objects. A commit it moves off is still held, which
#: is what :func:`has_commit` asks, until ``git gc`` prunes it; fetched again
#: then, it costs only what this ref's history lacks.
FETCHED_REF: Final = "refs/fleet/fetched"

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

#: A mirror's fetch turn as :mod:`fleet.core.file_turn` holds it. Inside it
#: an export probes the mirror and fetches at most one commit, which ends
#: within one probe's deadline and the fetch's budget, so a lock older than
#: both with a probe's deadline to spare was left by a process that died
#: inside its turn.
FETCH_TURN: Final = file_turn.TurnBounds(
    what="mirror fetch turn",
    waited_for="another export fetching into the same mirror",
    stale_seconds=FETCH_BUDGET_SECONDS + 2 * INIT_TIMEOUT_SECONDS,
    poll_seconds=FETCH_PAUSE_SECONDS,
)

#: What git says when the remote will not serve the object asked for: the
#: three spellings across protocol versions and server settings. Any of
#: them means the sha is not fetchable from the remote, which for a project
#: on GitHub means it was never pushed.
NOT_ON_REMOTE_MARKERS: Final = (
    "couldn't find remote ref",
    "unadvertised object",
    "not our ref",
)

#: What git says when the remote refused the hub's credential or asked for
#: one the hub could not give, as git 2.43 printed each on 2026-10-20: no
#: credential answered by the helper (prompts are unavailable to the
#: runner), an https credential the remote refused, and an ssh key it does
#: not know. GitHub's ``Repository not found`` is not one: it is also what
#: a missing repository answers, so it stays git's own words under
#: ``EXPORT_FAILED``.
CREDENTIAL_REFUSED_MARKERS: Final = (
    "could not read Username for",
    "Authentication failed for",
    "Permission denied (publickey)",
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


def _credential_refused(remote: str, stderr: str) -> AppError[FleetErrorCode]:
    """Compose the refusal for a remote that refused the hub's credential.

    Args:
        remote: The remote the fetch was sent to.
        stderr: git's own words, carried verbatim and bounded.

    Returns:
        ``REMOTE_CREDENTIAL_REJECTED``, naming the remote and the control
        host's credential as what must change.
    """
    return AppError(
        code=FleetErrorCode.REMOTE_CREDENTIAL_REJECTED,
        message=(
            f"{remote} refused the credential this hub's git sent it, or asked for one "
            f"its credential helper did not answer; nothing from it can be exported "
            f"until that credential is renewed on the control host, and sending the "
            f"fetch again would send the same one. git said: {stderr.strip()[:300]}"
        ),
    )


def _fetched(
    argv: tuple[str, ...],
    *,
    remote: str,
    verb: str,
    not_served: Callable[[str], AppError[FleetErrorCode]],
) -> None:
    """Send one fetch until it is done, the remote refuses it, or its attempts run out.

    Args:
        argv: The ``git fetch`` command.
        remote: The remote ``argv`` fetches from, for a refused credential.
        verb: What is fetched where, for the refusal.
        not_served: The refusal for a remote that does not serve what was
            asked for, given git's words.

    Raises:
        AppError: ``REMOTE_CREDENTIAL_REJECTED`` at the first attempt git
            answers with one of :data:`CREDENTIAL_REFUSED_MARKERS`;
            ``not_served``'s refusal at the first it answers with one of
            :data:`NOT_ON_REMOTE_MARKERS`; ``EXPORT_FAILED`` naming every
            attempt's words when none was done.
    """
    deadline = _test_hooks.now() + FETCH_BUDGET_SECONDS
    said: list[str] = []
    while True:
        timeout = min(FETCH_TIMEOUT_SECONDS, deadline - _test_hooks.now())
        fetched = _test_hooks.run(argv, timeout_seconds=timeout)
        if fetched["returncode"] == 0:
            return
        stderr = fetched["stderr"]
        if any(marker in stderr for marker in CREDENTIAL_REFUSED_MARKERS):
            raise _credential_refused(remote, stderr)
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


def turn_lock(mirror: pathlib.Path) -> pathlib.Path:
    """Where a mirror's fetch turn is held.

    Beside the mirror rather than in it, so nothing git reads or repacks
    holds a file git did not write.

    Args:
        mirror: The mirror's path.

    Returns:
        ``<mirror>.fetch.lock`` in the mirrors directory.
    """
    return mirror.with_name(f"{mirror.name}.fetch.lock")


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
    """Fetch one commit from the remote into the mirror's :data:`FETCHED_REF`.

    Skipped when the mirror already holds it: a second job for the same sha
    (a re-run, or two packages of one commit) costs no round trip. The probe
    and the fetch share one :data:`FETCH_TURN`, so a job that asks while
    another fetches the same commit waits for it and is then skipped.

    Args:
        mirror: The mirror's path, already initialised.
        remote: The project's declared remote.
        sha: The commit, forty hex.

    Raises:
        AppError: ``REMOTE_CREDENTIAL_REJECTED`` when the remote refused
            the hub's credential (one of :data:`CREDENTIAL_REFUSED_MARKERS`),
            ``SHA_NOT_ON_REMOTE`` when it will not serve the object (one of
            :data:`NOT_ON_REMOTE_MARKERS` in git's words), ``EXPORT_FAILED``
            when no attempt was done, every attempt's words verbatim.
    """

    def not_served(stderr: str) -> AppError[FleetErrorCode]:
        return AppError(
            code=FleetErrorCode.SHA_NOT_ON_REMOTE,
            message=(
                f"{remote} does not serve {sha}; a commit the remote has never seen is one "
                f"git push away from checkable, and this runner exports only what the remote "
                f"serves. git said: {stderr.strip()[:300]}"
            ),
        )

    with file_turn.held(turn_lock(mirror), holder=f"the export of {sha}", bounds=FETCH_TURN):
        if has_commit(mirror, sha):
            return
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
                f"+{sha}:{FETCHED_REF}",
            ),
            remote=remote,
            verb=f"git fetch {remote} {sha} into {mirror}",
            not_served=not_served,
        )


def fetch_ref(mirror: pathlib.Path, remote: str, ref: str) -> str:
    """Fetch a ref's tip into the mirror and answer the commit it names.

    Never skipped the way :func:`fetch_commit` is: a sha the mirror holds is
    that sha forever, while a ref's tip is a moving answer and the question
    a companion asks is what it points at NOW. The fetch and the read of the
    ref share one :data:`FETCH_TURN`, as :func:`fetch_commit`'s do, so two
    jobs never write the one ref at once.

    Args:
        mirror: The companion's mirror, already initialised.
        remote: The companion's declared remote.
        ref: The declared ref, ``main`` or ``refs/heads/main``.

    Returns:
        The forty-hex commit the ref names.

    Raises:
        AppError: ``REMOTE_CREDENTIAL_REJECTED`` when the remote refused
            the hub's credential (one of :data:`CREDENTIAL_REFUSED_MARKERS`);
            ``COMPANION_REF_NOT_ON_REMOTE`` when it does not serve the ref
            (one of :data:`NOT_ON_REMOTE_MARKERS` in git's words), which is
            a registry line to correct rather than a commit to push;
            ``EXPORT_FAILED`` when no attempt was done, or
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

    with file_turn.held(turn_lock(mirror), holder=f"the fetch of {ref}", bounds=FETCH_TURN):
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
            remote=remote,
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
    "CREDENTIAL_REFUSED_MARKERS",
    "FETCHED_REF",
    "FETCH_ATTEMPTS",
    "FETCH_BUDGET_SECONDS",
    "FETCH_PAUSE_SECONDS",
    "FETCH_TIMEOUT_SECONDS",
    "FETCH_TURN",
    "INIT_TIMEOUT_SECONDS",
    "NOT_ON_REMOTE_MARKERS",
    "fetch_commit",
    "fetch_ref",
    "git_failure",
    "has_commit",
    "turn_lock",
]
