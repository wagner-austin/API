"""The sealed session-verbs release a session verb runs, planted for the suites.

Shared by ``test_session_release`` (finding and verifying the release) and
``test_agent_restart`` (the tick that does so before every session verb), so
the pointer MCPs writes, the verifier call and its answer are written down
once. The pointer is planted for real in the store under a per-test
``HOME``: the release is found through the real
``file_exists`` and ``read_text`` hooks, so what a test plants is what the
code reads.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Final

from platform_core.json_utils import JSONObject, dump_json_str

from fleet.core import _test_hooks
from fleet.core.session_release import SessionRelease
from tests._queue_fakes import FakeEnv, queue_env
from tests.conftest import ok

#: The release id: a candidate directory name as MCPs makes it, a GUID
#: without hyphens.
RELEASE_ID: Final = "3f0c9d5e8a7b4c21b6e0f1d2a3c4b5e6"

#: The commit the release was checked out at.
REVISION: Final = "58efcb92d6c1e0a4b3f2e1d0c9b8a7f6e5d4c3b2"

#: The store under ``HOME``, as MCPs maketools ``release/store.py`` lays it
#: out, written out rather than read from the module under test.
STORE_PARTS: Final[tuple[str, ...]] = (".local", "share", "corvis", "session-verbs-releases")


def store_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """The per-test directory ``HOME`` names.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The directory the environment names.
    """
    return tmp_path / "root"


def store(tmp_path: pathlib.Path) -> pathlib.Path:
    """The kind's store.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        ``<root>/.local/share/corvis/session-verbs-releases``.
    """
    return store_root(tmp_path).joinpath(*STORE_PARTS)


def release_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Where the planted release's checkout is, as MCPs lays a store out.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        ``<store>/<id>/MCPs``.
    """
    return store(tmp_path) / RELEASE_ID / "MCPs"


def pointer_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """The active-release pointer in the per-test store.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The ``active.json`` path.
    """
    return store(tmp_path) / "active.json"


def point_environment(tmp_path: pathlib.Path) -> None:
    """Give the agent the queue's variables and the store's ``HOME``.

    Args:
        tmp_path: The test's temporary directory.
    """
    _test_hooks.env = FakeEnv({**queue_env().values, "HOME": str(store_root(tmp_path))})


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
    """The pointer MCPs writes for the planted release.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The record, its keys in MCPs' order.
    """
    return {
        "kind": "session-verbs",
        "release": RELEASE_ID,
        "revision": REVISION,
        "root": str(release_root(tmp_path)),
    }


def plant_release(tmp_path: pathlib.Path) -> SessionRelease:
    """Activate the planted release, as MCPs would.

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
        registry_dir=str(root / "mcp-shared" / "src" / "source-registry"),
    )


def verifier(tmp_path: pathlib.Path) -> pathlib.Path:
    """The planted release's own maketools launcher.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        ``<root>/packages/maketools/scripts/run.py``.
    """
    return release_root(tmp_path) / "packages" / "maketools" / "scripts" / "run.py"


def verify_call(tmp_path: pathlib.Path) -> tuple[str, ...]:
    """The verifier invocation the agent runs before a verb.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        This interpreter, isolated, on the release's own maketools
        ``verify-release`` against its root and kind.
    """
    root = release_root(tmp_path)
    return (
        sys.executable,
        "-I",
        "-B",
        str(verifier(tmp_path)),
        "verify-release",
        "--root",
        str(root),
        "--kind",
        "session-verbs",
    )


def verified_line(tmp_path: pathlib.Path) -> str:
    """The line MCPs maketools ``verify-release`` prints for the planted release.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        ``RELEASE_VERIFIED kind=... release=... revision=... root=...``.
    """
    return (
        f"RELEASE_VERIFIED kind=session-verbs release={RELEASE_ID} "
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
    "STORE_PARTS",
    "plant_release",
    "point_environment",
    "pointer_file",
    "pointer_record",
    "release_root",
    "store",
    "store_root",
    "verified_line",
    "verifier",
    "verify_call",
    "verify_reply",
    "write_pointer",
]
