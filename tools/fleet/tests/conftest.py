"""Shared fakes and the hook reset that keeps tests independent.

:class:`FakeRun` and :class:`FakeClock` are FAKES, not mocks. Each implements
the same Protocol the production implementation does and records what it was
asked to do, so an assertion is about the commands this package builds rather
than about a patching library's call-recording API. Nothing here patches
anything: the hooks in :mod:`fleet.core._test_hooks` are module-level names,
and a test rebinds them.

HOOKS ARE RESET BEFORE AND AFTER EVERY TEST. A rebinding that leaked would
produce a test that fails only when it runs after a specific other one, and
``-n auto`` reorders freely -- so the symptom would be an intermittent failure
whose cause is invisible in the failing test.

THE CLOCK IS A FAKE BECAUSE THE PACKAGE IS ABOUT EXPIRY. Every question about
whether a resource is free is a question about the time, and a test that could
not move the clock could only assert that an unexpired lease is unexpired --
the case that never breaks.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator, Sequence

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.core import _test_hooks, manifest


class FakeRun:
    """A command runner that answers from a script and records the calls.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`.

    Attributes:
        calls: Every argv it was given, in order.
        stdin: Every stdin payload it was given, in order, with None for the
            calls that had none. A separate list rather than a field on the
            call, so a test asserting on argv does not have to mention bytes.
    """

    calls: list[tuple[str, ...]]
    stdin: list[bytes | None]
    _replies: list[_test_hooks.CommandResult]

    def __init__(self, replies: Sequence[_test_hooks.CommandResult]) -> None:
        """Build a runner that will answer with these results in order.

        Args:
            replies: One result per expected call. Running out is an error
                rather than a default: a test that made more calls than it
                declared has changed behaviour it did not mean to assert on.
        """
        self.calls = []
        self.stdin = []
        self._replies = list(replies)

    def __call__(
        self, argv: Sequence[str], *, stdin_bytes: bytes | None = None
    ) -> _test_hooks.CommandResult:
        """Record a call and answer with the next scripted result.

        Args:
            argv: The command.
            stdin_bytes: Its standard input, or None.

        Returns:
            The next scripted result.

        Raises:
            AssertionError: If more calls are made than results were given.
        """
        self.calls.append(tuple(argv))
        self.stdin.append(stdin_bytes)
        assert self._replies, f"unscripted call: {list(argv)}"
        return self._replies.pop(0)


class FakeClock:
    """A clock a test moves by hand.

    Satisfies :class:`~fleet.core._test_hooks.NowProtocol`.

    Attributes:
        seconds: The current time, whole seconds since the epoch. Assign to
            it to move time.
    """

    seconds: int

    def __init__(self, seconds: int) -> None:
        """Start the clock.

        Args:
            seconds: Initial time, whole seconds since the epoch.
        """
        self.seconds = seconds

    def __call__(self) -> int:
        """Read the current time.

        Returns:
            Whatever ``seconds`` currently holds.
        """
        return self.seconds


def ok(stdout: str) -> _test_hooks.CommandResult:
    """Build a successful command result.

    Args:
        stdout: What the command printed.

    Returns:
        The result, exit status zero and empty stderr.
    """
    return _test_hooks.CommandResult(returncode=0, stdout=stdout, stderr="")


def failed(returncode: int, stderr: str) -> _test_hooks.CommandResult:
    """Build a failing command result.

    Args:
        returncode: The exit status.
        stderr: What the command wrote to standard error.

    Returns:
        The result, with empty stdout.
    """
    return _test_hooks.CommandResult(returncode=returncode, stdout="", stderr=stderr)


#: The pinned clock every end-to-end dispatch test runs against.
DEMO_NOW = 1_757_000_000

#: The project the demo monorepo contains.
DEMO_PROJECT = "libs/demo"

#: The library ``libs/demo`` declares a path dependency on.
#:
#: It exists so the staging tests exercise a MULTI-MEMBER archive. A demo
#: project with no dependencies would have passed every test in this suite
#: while ``dispatch.start`` staged one directory -- which is exactly what
#: shipped, and what the first dispatch to a real node found could not build.
DEMO_DEPENDENCY = "libs/base"

#: The run id a dispatch of :data:`DEMO_PROJECT` gets at :data:`DEMO_NOW`.
DEMO_RUN_ID = f"libs-demo-{DEMO_NOW}"

#: What a healthy capacity probe answers.
PROBE_OK = "free_ram_gb=27.0\nfree_disk_gb=860.0\n"


def dispatch_replies(archive_digest: str) -> list[_test_hooks.CommandResult]:
    """Every command a successful dispatch runs, in order.

    Written out rather than indexed into, because the sequence is the thing
    under test: the archive step runs ``tar`` through the SAME hook as ssh, so
    a list built by patching one position silently misaligns the moment a step
    is added. Naming each call makes that visible -- and it did: the launch
    grew from one script to two when the build was split out of the
    registration, and this list is where that had to be accounted for.

    Args:
        archive_digest: What the node should report having reassembled. The
            real digest for a success; anything else exercises the refusal.

    Returns:
        One result per call.
    """
    return [
        ok(""),  # probe: send script
        ok(PROBE_OK),  # probe: run it
        ok(""),  # tar, locally
        ok(""),  # stage: send mkdir script
        ok(""),  # stage: run mkdir
        ok(""),  # stage: send the base64 payload
        ok(""),  # stage: send reassemble script
        ok(archive_digest),  # stage: run reassemble
        ok(""),  # stage: send extract script
        ok(""),  # stage: run extract
        ok(""),  # launch: send the build script
        ok(""),  # launch: send the registration script
        ok("launched"),  # launch: run the registration script
    ]


def workspace_document() -> JSONObject:
    """Build a one-node, one-project workspace as JSON.

    Returns:
        The document, ready to serialise.
    """
    return {
        "nodes": {
            "lavender": {
                "host": "lavender",
                "stage_root": "C:/fleet/stage",
                "logical_cores": 16,
                "ram_gb": 32.0,
                "gpu": None,
                "budget": {
                    "reserved_cores": 2,
                    "reserved_ram_gb": 4.0,
                    "worker_ram_gb": 1.1,
                    "max_concurrent_runs": 2,
                    "max_disk_gb": 20.0,
                },
            }
        },
        "projects": {
            DEMO_PROJECT: {
                "worker_ram_gb": 1.1,
                "minimum_workers": 2,
                "expected_minutes": 5,
            }
        },
        "ledger": "ledger.jsonl",
        "feed": "feed.jsonl",
        "leases": "leases.json",
    }


@pytest.fixture(name="repo")
def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a tiny monorepo shaped like the real one.

    A real tree rather than a fixture archive, because the archive step runs
    the real ``tar`` and a fabricated one would test nothing about it.

    It has the three things that make a project here NOT self-contained: a
    manifest declaring a sibling by path, and the two shared directories every
    Makefile and guard shim reaches for. A flatter fixture is what let a
    single-directory dispatch look correct for as long as it did.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The repo root.
    """
    root = tmp_path / "repo"
    project = root / DEMO_PROJECT
    project.mkdir(parents=True)
    project.joinpath("Makefile").write_text("check:\n\techo ok\n", encoding="utf-8")
    project.joinpath("pyproject.toml").write_text(
        "[tool.poetry.dependencies]\n"
        f'base = {{ path = "../{pathlib.PurePosixPath(DEMO_DEPENDENCY).name}" }}\n',
        encoding="utf-8",
    )
    project.joinpath(".venv").mkdir()
    project.joinpath(".venv", "huge.bin").write_text("x" * 4096, encoding="utf-8")

    dependency = root / DEMO_DEPENDENCY
    dependency.mkdir(parents=True)
    dependency.joinpath("pyproject.toml").write_text(
        "[tool.poetry.dependencies]\n", encoding="utf-8"
    )

    for directory in manifest.SHARED_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
        (root / directory / "placeholder.txt").write_text("present\n", encoding="utf-8")
    for name in manifest.SHARED_FILES:
        (root / name).write_text("# present\n", encoding="utf-8")
    return root


@pytest.fixture(name="config_path")
def _config_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a workspace document and pin the clock.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path to the written document.
    """
    _test_hooks.now = FakeClock(DEMO_NOW)
    path = tmp_path / "fleet.json"
    path.write_text(dump_json_str(workspace_document()), encoding="utf-8")
    return path


@pytest.fixture(name="reset_hooks", autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Put every hook back to its real implementation around each test."""
    _restore()
    yield None
    _restore()


def _restore() -> None:
    """Rebind every hook to the implementation it starts life with."""
    _test_hooks.run = _test_hooks._default_run
    _test_hooks.now = _test_hooks._default_now
    _test_hooks.read_text = _test_hooks._default_read_text
    _test_hooks.read_bytes = _test_hooks._default_read_bytes
    _test_hooks.file_exists = _test_hooks._default_file_exists
    _test_hooks.directory_exists = _test_hooks._default_directory_exists
    _test_hooks.append_text = _test_hooks._default_append_text
    _test_hooks.write_text = _test_hooks._default_write_text
