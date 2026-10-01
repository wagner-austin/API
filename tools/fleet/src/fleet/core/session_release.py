"""The sealed session-audit release a hub session verb runs, and nothing else.

WHAT WENT WRONG (MCPs board tasks f4cd489f and c7c2527d). Every session verb
this runner executes (restart, revive, both kills, compact and exit) first
ran ``poetry -C <mcps>/packages/session-audit run session-audit``, whose
editable install is the MCPs main checkout's working tree: a tree no
worktree publish advances and that carries other sessions' uncommitted
edits. From 2026-09-24 the runner put a ``git archive`` of
``refs/remotes/origin/main`` first on ``PYTHONPATH`` instead, which fixed
the stale tree but still ran whatever commit had been pushed last, checked
or not, with the checkout's own environment. On 2026-09-28 the operator
ruled (card 6b697bb4, 'build sealed releases for both') that the fleet-agent
runs only a sealed release, the way the supervisors, the fleet audit and the
hub cleanup already did.

WHAT THIS DOES. MCPs ``scripts/ops/prepare-release.ps1 -Kind session-verbs``
checks a release out at a full commit, builds session-audit's own
environment inside it, runs its checks and seals it; MCPs
``scripts/ops/register-session-verbs.ps1`` refuses anything but such a
release and points :data:`POINTER_NAME` in the release store at it. Before
every verb, :func:`active_session_release` reads that pointer, runs the
release's own ``scripts/ops/verify-release.ps1`` (MCPs ``Read-Release``,
which recomputes the payload digest and compares it and the detached
commit with the seal), and requires the verifier to name exactly the
release the pointer named. The verb then runs that release's session-audit
with its sealed environment and its registry. The job's closing detail
names the release id and revision.

NO FALLBACK. No pointer, a malformed one, a release that fails its seal or
a verifier whose answer is not the pointer's release refuses the job with a
named code, and nothing runs; running the checkout or ``origin/main`` on the
day the release is missing is exactly what the ruling removed.

WHY POWERSHELL FOR THE CHECK. The seal is MCPs' digest
(``scripts/lib/release.ps1``), one implementation for every release kind; a
second copy of it here would be the fork that drifts first. The verifier is
inside the payload it checks, as every kind's registration is.
"""

from __future__ import annotations

import pathlib
import re
from typing import Final, TypedDict

from platform_core.json_utils import load_json_str

from fleet.core import _test_hooks
from fleet.core.commit_tree import step_refusal

#: The release kind a session verb runs, as MCPs ``release-kinds.ps1`` names it.
KIND: Final = "session-verbs"

#: The release store under ``%LOCALAPPDATA%``, as MCPs
#: ``register-session-verbs.ps1`` writes it.
STORE: Final = pathlib.PurePosixPath("Corvis/session-verbs-releases")

#: The pointer file in the store naming the active release.
POINTER_NAME: Final = "active.json"

#: The pointer's keys, in the order MCPs writes them.
POINTER_KEYS: Final[tuple[str, ...]] = ("kind", "release", "revision", "root")

#: The verifier inside a release, relative to its checkout.
VERIFIER: Final = pathlib.PurePosixPath("scripts/ops/verify-release.ps1")

#: The source registry inside a release, passed to every verb.
REGISTRY_DIR: Final = pathlib.PurePosixPath("mcp-shared/src/source-registry")

#: The verifier's deadline. It hashes the whole payload, the private
#: environment included, which took seconds when measured; a verifier still
#: running after five minutes has stopped answering.
VERIFY_TIMEOUT_SECONDS: Final[int] = 300

#: A full commit id, as a release receipt records it.
REVISION_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")

#: Detail prefix when the environment names no local application data root.
NO_APPDATA_CODE: Final = "SESSION_RELEASE_NO_APPDATA"

#: Detail prefix when no release has been activated.
NOT_ACTIVE_CODE: Final = "SESSION_RELEASE_NOT_ACTIVE"

#: Detail prefix when the pointer is not the record MCPs writes.
POINTER_INVALID_CODE: Final = "SESSION_RELEASE_POINTER_INVALID"

#: Detail prefix when the release fails its seal.
UNSEALED_CODE: Final = "SESSION_RELEASE_UNSEALED"

#: Detail prefix when the verifier names a release other than the pointer's.
ANSWER_MISMATCH_CODE: Final = "SESSION_RELEASE_ANSWER_MISMATCH"


class SessionRelease(TypedDict):
    """The verified release one session verb runs.

    Attributes:
        release: The release id, the candidate directory MCPs created.
        revision: The full commit the release was checked out at.
        root: The release's checkout.
        registry_dir: Its source registry, passed as ``--registry-dir``.
    """

    release: str
    revision: str
    root: pathlib.Path
    registry_dir: str


def pointer_path() -> pathlib.Path | str:
    """Locate the active-release pointer.

    Returns:
        ``%LOCALAPPDATA%/Corvis/session-verbs-releases/active.json``, or a
        refusal detail when the environment names no ``LOCALAPPDATA``.
    """
    appdata = _test_hooks.env("LOCALAPPDATA")
    # The env seam answers None for unset and blank alike.
    if appdata is None:
        return (
            f"{NO_APPDATA_CODE}: LOCALAPPDATA is unset in this runner's environment, so the "
            "session-verbs release store cannot be found; nothing was run"
        )
    return pathlib.Path(appdata) / STORE / POINTER_NAME


def _pointed(path: pathlib.Path) -> SessionRelease | str:
    """Read the pointer into the release it names, unverified.

    Args:
        path: The pointer file.

    Returns:
        The release the pointer names, or a refusal detail.
    """
    if not _test_hooks.file_exists(path):
        return (
            f"{NOT_ACTIVE_CODE}: {path} does not exist; activate a sealed session-verbs "
            "release with MCPs make register-session-verbs RELEASE=<root>; nothing was run"
        )
    record = load_json_str(_test_hooks.read_text(path))
    if not isinstance(record, dict) or tuple(record) != POINTER_KEYS:
        return f"{POINTER_INVALID_CODE}: {path} is not an object of {', '.join(POINTER_KEYS)}"
    kind, release, revision, root = (record[key] for key in POINTER_KEYS)
    if not (
        isinstance(kind, str)
        and isinstance(release, str)
        and isinstance(revision, str)
        and isinstance(root, str)
    ):
        return f"{POINTER_INVALID_CODE}: {path} must hold strings, got {record!r}"
    if kind != KIND or REVISION_PATTERN.fullmatch(revision) is None or release == "":
        return (
            f"{POINTER_INVALID_CODE}: {path} names kind {kind!r}, release {release!r} and "
            f"revision {revision!r}; a {KIND} release with an id and a full commit is required"
        )
    checkout = pathlib.Path(root)
    return SessionRelease(
        release=release,
        revision=revision,
        root=checkout,
        registry_dir=str(checkout / REGISTRY_DIR),
    )


def active_session_release() -> SessionRelease | str:
    """Find the active session-verbs release and verify its seal.

    Returns:
        The verified release, or a ``CODE: message`` refusal detail for the
        queue when there is none, its pointer is malformed, its seal does
        not verify, or the verifier names another release.
    """
    path = pointer_path()
    if isinstance(path, str):
        return path
    release = _pointed(path)
    if isinstance(release, str):
        return release
    root = release["root"]
    verifier = root / VERIFIER
    result = _test_hooks.run(
        (
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(verifier),
            "-Root",
            str(root),
            "-Kind",
            KIND,
        ),
        timeout_seconds=VERIFY_TIMEOUT_SECONDS,
    )
    if result["returncode"] != 0:
        return step_refusal(UNSEALED_CODE, str(verifier), result)
    expected = (
        f"RELEASE_VERIFIED kind={KIND} release={release['release']} "
        f"revision={release['revision']} root={root}"
    )
    answered = result["stdout"].strip()
    if answered != expected:
        return (
            f"{ANSWER_MISMATCH_CODE}: {verifier} answered {answered[:400]!r} where the "
            f"pointer {path} requires {expected!r}; nothing was run"
        )
    return release


__all__ = [
    "ANSWER_MISMATCH_CODE",
    "KIND",
    "NOT_ACTIVE_CODE",
    "NO_APPDATA_CODE",
    "POINTER_INVALID_CODE",
    "POINTER_KEYS",
    "POINTER_NAME",
    "REGISTRY_DIR",
    "REVISION_PATTERN",
    "STORE",
    "UNSEALED_CODE",
    "VERIFIER",
    "VERIFY_TIMEOUT_SECONDS",
    "SessionRelease",
    "active_session_release",
    "pointer_path",
]
