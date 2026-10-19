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

WHAT THIS DOES. MCPs cuts a session-verbs release at a full commit, builds
session-audit's own environment inside it, runs its checks, seals it, and
points :data:`POINTER_NAME` in the kind's store at it. Before every verb,
:func:`active_session_release` reads that pointer, has the release's own
maketools ``verify-release`` check its seal (:mod:`fleet.core.sealed_release`,
the check a netboot arm makes too), and requires the verified release and
revision to be exactly the pointer's. The verb then runs that release's
session-audit with its sealed environment and its registry. The job's
closing detail names the release id and revision.

THE STORE IS THE PLATFORM'S (MCPs board task 315111e4). The sessions duty
moved to diphtheria, a Linux host, on 2026-10-18, and its first kill was
refused because this module read the store only from ``LOCALAPPDATA``, which
Linux does not set. MCPs keeps every kind's store under one per-user base per
platform (maketools ``release/store.py`` ``STORE_BASE``), and
:data:`STORE_BASES` says the same: ``<LOCALAPPDATA>/Corvis`` on Windows,
``<HOME>/.local/share/corvis`` on Linux. This package cannot import
maketools, which lives in another repository and environment, so the pair is
written here and the verifier, which is maketools, is the release's own.

NO FALLBACK. An unknown platform, an unset base variable, no pointer, a
malformed one, a release that fails its seal or a verified release that is
not the pointer's refuses the job with a named code, and nothing runs;
running the checkout or ``origin/main`` on the day the release is missing is
exactly what the ruling removed.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Final, TypedDict

from platform_core.json_utils import load_json_str

from fleet.core import _test_hooks, sealed_release

#: The release kind a session verb runs, as MCPs ``release-kinds.json`` names it.
KIND: Final = "session-verbs"

#: Where each platform keeps every release kind's store: the environment
#: variable naming the per-user root, and the base under it, as MCPs
#: maketools ``release/store.py`` ``STORE_BASE`` lays them out.
STORE_BASES: Final[Mapping[str, tuple[str, pathlib.PurePosixPath]]] = {
    "win32": ("LOCALAPPDATA", pathlib.PurePosixPath("Corvis")),
    "linux": ("HOME", pathlib.PurePosixPath(".local/share/corvis")),
}

#: The kind's store under its platform's base.
STORE: Final = pathlib.PurePosixPath(f"{KIND}-releases")

#: The pointer file in the store naming the active release.
POINTER_NAME: Final = "active.json"

#: The pointer's keys, in the order MCPs writes them.
POINTER_KEYS: Final[tuple[str, ...]] = ("kind", "release", "revision", "root")

#: The source registry inside a release, passed to every verb.
REGISTRY_DIR: Final = pathlib.PurePosixPath("mcp-shared/src/source-registry")

#: Detail prefix when this process runs on a platform with no store base.
PLATFORM_UNSUPPORTED_CODE: Final = "SESSION_RELEASE_PLATFORM_UNSUPPORTED"

#: Detail prefix when the environment names no root for the platform's store.
NO_STORE_BASE_CODE: Final = "SESSION_RELEASE_NO_STORE_BASE"

#: Detail prefix when no release has been activated.
NOT_ACTIVE_CODE: Final = "SESSION_RELEASE_NOT_ACTIVE"

#: Detail prefix when the pointer is not the record MCPs writes.
POINTER_INVALID_CODE: Final = "SESSION_RELEASE_POINTER_INVALID"

#: Detail prefix when the verified release or revision is not the pointer's.
POINTER_MISMATCH_CODE: Final = "SESSION_RELEASE_POINTER_MISMATCH"


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


def pointer_path(operating_system: str) -> pathlib.Path | str:
    """Locate the active-release pointer in this platform's store.

    Args:
        operating_system: The platform this runner is on, as
            ``sys.platform`` names it; the caller passes it so one suite
            reads every platform's store.

    Returns:
        ``<base>/session-verbs-releases/active.json`` under the platform's
        :data:`STORE_BASES` root, or a refusal detail when the platform has
        no store base or the environment names no root for it.
    """
    base = STORE_BASES.get(operating_system)
    if base is None:
        return (
            f"{PLATFORM_UNSUPPORTED_CODE}: this runner is on {operating_system!r}, and a "
            f"session-verbs release store is kept only on {', '.join(STORE_BASES)}; "
            "nothing was run"
        )
    variable, relative = base
    root = _test_hooks.env(variable)
    # The env seam answers None for unset and blank alike.
    if root is None:
        return (
            f"{NO_STORE_BASE_CODE}: {variable} is unset in this runner's environment, so the "
            f"{operating_system} session-verbs release store cannot be found; nothing was run"
        )
    return pathlib.Path(root) / relative / STORE / POINTER_NAME


def _pointed(path: pathlib.Path) -> SessionRelease | str:
    """Read the pointer into the release it names, unverified.

    Args:
        path: The pointer file.

    Returns:
        The release the pointer names, or a refusal detail.
    """
    if not _test_hooks.file_exists(path):
        return (
            f"{NOT_ACTIVE_CODE}: {path} does not exist, so no sealed session-verbs release "
            "is active on this machine; MCPs activates one on the host of the sessions "
            "duty; nothing was run"
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
    if kind != KIND or sealed_release.REVISION_PATTERN.fullmatch(revision) is None or release == "":
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


def active_session_release(operating_system: str) -> SessionRelease | str:
    """Find the active session-verbs release and verify its seal.

    Args:
        operating_system: The platform this runner is on, as
            ``sys.platform`` names it, which decides where its store is.

    Returns:
        The verified release, or a ``CODE: message`` refusal detail for the
        queue when there is none, its pointer is malformed, its seal does
        not verify, or the release its verifier verified is not the
        pointer's.
    """
    path = pointer_path(operating_system)
    if isinstance(path, str):
        return path
    release = _pointed(path)
    if isinstance(release, str):
        return release
    verified = sealed_release.verify_release(release["root"], KIND)
    if isinstance(verified, str):
        return verified
    if (verified["release"], verified["revision"]) != (release["release"], release["revision"]):
        return (
            f"{POINTER_MISMATCH_CODE}: the pointer {path} names release {release['release']} "
            f"at revision {release['revision']}, and its root's own verifier verified release "
            f"{verified['release']} at revision {verified['revision']}; nothing was run"
        )
    return release


__all__ = [
    "KIND",
    "NOT_ACTIVE_CODE",
    "NO_STORE_BASE_CODE",
    "PLATFORM_UNSUPPORTED_CODE",
    "POINTER_INVALID_CODE",
    "POINTER_KEYS",
    "POINTER_MISMATCH_CODE",
    "POINTER_NAME",
    "REGISTRY_DIR",
    "STORE",
    "STORE_BASES",
    "SessionRelease",
    "active_session_release",
    "pointer_path",
]
