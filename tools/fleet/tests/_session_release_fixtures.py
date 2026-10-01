"""The sealed session-verbs release a session verb runs, planted for the suites.

Shared by ``test_session_release`` (finding and verifying the release) and
``test_agent_restart`` (the tick that does so before every session verb), so
the pointer MCPs ``register-session-verbs.ps1`` writes, the verifier call and
its answer are written down once. The pointer is planted for real under a
per-test ``LOCALAPPDATA``: the release is found through the real
``file_exists`` and ``read_text`` hooks, so what a test plants is what the
code reads.
"""

from __future__ import annotations

import pathlib
from typing import Final

from platform_core.json_utils import JSONObject, dump_json_str

from fleet.core import _test_hooks, session_release
from fleet.core.session_release import SessionRelease
from tests._queue_fakes import FakeEnv, queue_env
from tests.conftest import ok

#: The release id: a candidate directory name as MCPs ``release-build.ps1``
#: makes it, a GUID without hyphens.
RELEASE_ID: Final = "3f0c9d5e8a7b4c21b6e0f1d2a3c4b5e6"

#: The commit the release was checked out at.
REVISION: Final = "58efcb92d6c1e0a4b3f2e1d0c9b8a7f6e5d4c3b2"


def release_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Where the planted release's checkout is, as MCPs lays a store out.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        ``<LOCALAPPDATA>/Corvis/session-verbs-releases/<id>/MCPs``.
    """
    return appdata(tmp_path) / session_release.STORE / RELEASE_ID / "MCPs"


def appdata(tmp_path: pathlib.Path) -> pathlib.Path:
    """The per-test ``LOCALAPPDATA``.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The directory the environment names.
    """
    return tmp_path / "appdata"


def pointer_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """The active-release pointer under the per-test ``LOCALAPPDATA``.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The ``active.json`` path.
    """
    return appdata(tmp_path) / session_release.STORE / session_release.POINTER_NAME


def point_environment(tmp_path: pathlib.Path) -> None:
    """Give the agent's environment the queue's variables and the test's ``LOCALAPPDATA``.

    Args:
        tmp_path: The test's temporary directory.
    """
    _test_hooks.env = FakeEnv({**queue_env().values, "LOCALAPPDATA": str(appdata(tmp_path))})


def write_pointer(tmp_path: pathlib.Path, record: JSONObject) -> pathlib.Path:
    """Write a pointer record, well formed or not, where the agent reads it.

    Args:
        tmp_path: The test's temporary directory.
        record: The JSON object to write.

    Returns:
        The pointer path.
    """
    point_environment(tmp_path)
    path = pointer_file(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json_str(record), encoding="utf-8")
    return path


def pointer_record(tmp_path: pathlib.Path) -> JSONObject:
    """The pointer MCPs ``register-session-verbs.ps1`` writes for the planted release.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The record, its keys in MCPs' order.
    """
    return {
        "kind": session_release.KIND,
        "release": RELEASE_ID,
        "revision": REVISION,
        "root": str(release_root(tmp_path)),
    }


def plant_release(tmp_path: pathlib.Path) -> SessionRelease:
    """Activate the planted release, as a registration would.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The release :func:`~fleet.core.session_release.active_session_release`
        returns once its verifier answers :func:`verify_reply`.
    """
    write_pointer(tmp_path, pointer_record(tmp_path))
    root = release_root(tmp_path)
    return SessionRelease(
        release=RELEASE_ID,
        revision=REVISION,
        root=root,
        registry_dir=str(root / session_release.REGISTRY_DIR),
    )


def verify_call(tmp_path: pathlib.Path) -> tuple[str, ...]:
    """The verifier invocation the agent runs before a verb.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The release's own ``verify-release.ps1`` against its root and kind.
    """
    root = release_root(tmp_path)
    return (
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(root / session_release.VERIFIER),
        "-Root",
        str(root),
        "-Kind",
        session_release.KIND,
    )


def verified_line(tmp_path: pathlib.Path) -> str:
    """The line MCPs ``verify-release.ps1`` prints for the planted release.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        ``RELEASE_VERIFIED kind=... release=... revision=... root=...``.
    """
    return (
        f"RELEASE_VERIFIED kind={session_release.KIND} release={RELEASE_ID} "
        f"revision={REVISION} root={release_root(tmp_path)}"
    )


def verify_reply(tmp_path: pathlib.Path) -> _test_hooks.CommandResult:
    """The verifier's successful answer.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        Exit 0 with :func:`verified_line`.
    """
    return ok(f"{verified_line(tmp_path)}\n")


__all__ = [
    "RELEASE_ID",
    "REVISION",
    "appdata",
    "plant_release",
    "point_environment",
    "pointer_file",
    "pointer_record",
    "release_root",
    "verified_line",
    "verify_call",
    "verify_reply",
    "write_pointer",
]
