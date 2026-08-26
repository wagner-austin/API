"""Shared fixtures and the fake command runner.

Hooks are reset before and after every test so a rebinding made by one test
cannot leak into another. Leakage here is particularly hard to diagnose: the
symptom is a test that fails only when it runs after a specific other test,
and ``-n auto`` reorders freely.

:class:`FakeRun` is a fake, not a mock. It implements the same protocol the
production runner does and records what it was asked to do, so assertions are
about the commands this package builds rather than about a patching library's
call-recording API.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Generator, Mapping, Sequence

import pytest
from platform_core.json_utils import JSONValue, dump_json_str

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.clusters.hpc3 import HPC3
from hpc3.contracts.cluster import ClusterFacts
from hpc3.core import _test_hooks as core_hooks
from hpc3.core._test_hooks import CommandResult


class RecordedCall:
    """One invocation the fake runner received.

    Attributes:
        argv: Executable and arguments, exactly as passed.
        stdin_bytes: Bytes offered on standard input, or None.
    """

    __slots__ = ("argv", "stdin_bytes")

    def __init__(self, argv: Sequence[str], stdin_bytes: bytes | None) -> None:
        """Record one invocation.

        Args:
            argv: Executable and arguments.
            stdin_bytes: Bytes offered on standard input, or None.
        """
        self.argv = tuple(argv)
        self.stdin_bytes = stdin_bytes

    @property
    def remote_command(self) -> str:
        """The command sent to the remote host.

        Returns:
            The final argv element, which is the command ``ssh`` executes.
        """
        return self.argv[-1]


class FakeRun:
    """A scripted stand-in for the real command runner.

    Responses are matched by substring against the remote command, in the
    order they were added, and the first match wins. A command matching no
    rule returns success with empty output, which keeps a test's setup to the
    commands it actually cares about.
    """

    def __init__(self) -> None:
        """Start with no rules and no recorded calls."""
        self.calls: list[RecordedCall] = []
        self._rules: list[tuple[str, CommandResult, bool]] = []

    def add(
        self,
        contains: str,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        once: bool = False,
    ) -> None:
        """Script a response for commands containing a substring.

        Args:
            contains: Substring matched against the remote command.
            stdout: Standard output to return.
            stderr: Standard error to return.
            returncode: Exit status to return.
            once: Consume the rule after it matches, so a later identical
                command falls through to the next rule. Needed wherever the
                same command is issued twice around a state change -- reading
                a job's state before and after cancelling it, for instance,
                where returning the same answer both times would let a test
                pass against behaviour that cannot happen.
        """
        self._rules.append(
            (contains, CommandResult(returncode=returncode, stdout=stdout, stderr=stderr), once)
        )

    def __call__(self, argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
        """Record the invocation and return its scripted response.

        Args:
            argv: Executable and arguments.
            stdin_bytes: Bytes offered on standard input, or None.

        Returns:
            The first matching scripted response, or empty success.
        """
        call = RecordedCall(argv, stdin_bytes)
        self.calls.append(call)
        for index, (contains, result, once) in enumerate(self._rules):
            if contains in call.remote_command:
                if once:
                    del self._rules[index]
                return result
        return CommandResult(returncode=0, stdout="", stderr="")

    def commands(self) -> list[str]:
        """List every remote command received, in order.

        Returns:
            The commands, for order-sensitive assertions.
        """
        return [call.remote_command for call in self.calls]


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Rebind every hook to production before and after each test."""
    core_hooks.reset_hooks()
    cli_hooks.reset_hooks()
    yield
    core_hooks.reset_hooks()
    cli_hooks.reset_hooks()


def _make_emitted() -> Generator[list[str], None, None]:
    """Capture CLI report lines instead of writing them to stdout.

    Yields:
        The list the CLI's ``emit`` hook appends to, in emission order.
    """
    lines: list[str] = []
    cli_hooks.emit = lines.append
    yield lines
    cli_hooks.reset_hooks()


class LoggedEvent:
    """One structured audit event the fake sink received.

    Attributes:
        event: Event name.
        fields: The structured context, exactly as emitted.
    """

    __slots__ = ("event", "fields")

    def __init__(self, event: str, fields: Mapping[str, str | int | float | bool]) -> None:
        """Record one event.

        Args:
            event: Event name.
            fields: Structured context.
        """
        self.event = event
        self.fields = dict(fields)


def _make_errors() -> Generator[list[str], None, None]:
    """Capture CLI refusal lines instead of writing them to stderr.

    Yields:
        The list the CLI's ``emit_error`` hook appends to, in order.
    """
    lines: list[str] = []
    cli_hooks.emit_error = lines.append
    yield lines
    cli_hooks.reset_hooks()


def _make_logged() -> Generator[list[LoggedEvent], None, None]:
    """Capture audit events instead of writing them to the platform logger.

    Yields:
        The list the core's ``log_event`` hook appends to, in emission order.
    """
    events: list[LoggedEvent] = []

    def _record(event: str, fields: Mapping[str, str | int | float | bool]) -> None:
        events.append(LoggedEvent(event, fields))

    core_hooks.log_event = _record
    yield events
    core_hooks.reset_hooks()


FROZEN_NOW = "2026-08-22T16:00:00+00:00"


def _make_frozen_clock() -> Generator[str, None, None]:
    """Pin the CLI clock so ledger timestamps are assertable.

    Yields:
        The timestamp every submission will record.
    """

    def _now() -> str:
        return FROZEN_NOW

    cli_hooks.now_iso = _now
    yield FROZEN_NOW
    cli_hooks.reset_hooks()


def _make_fake_run() -> Generator[FakeRun, None, None]:
    """Install the fake command runner for the duration of a test.

    Yields:
        The runner, for scripting responses and asserting on calls.
    """
    fake = FakeRun()
    core_hooks.run = fake
    yield fake
    core_hooks.reset_hooks()


def _make_argv() -> Generator[list[str], None, None]:
    """Give a test control of ``sys.argv`` and restore it afterwards.

    Yields:
        The live argument list, for the test to replace in place.
    """
    original = list(sys.argv)
    yield sys.argv
    sys.argv[:] = original


PREFLIGHT_LINE = (
    "sbatch: Job 1 to start at 2026-08-22T03:23:00 a using 4 processors "
    "on nodes hpc3-gpu-16-02 in partition free-gpu"
)

ABL_PINNED_DISTRIBUTIONS = (
    "torch==2.6.0+cu124\ntransformers==4.46.3\nnumpy==2.1.3\ntyping_extensions==4.12.2\n"
)
"""What ``/pub/wagnera3/envs/abl-pinned`` reports, as measured on the cluster.

The ablation's arms were produced against exactly these versions, so a probe
answering anything else means the run would not be comparable to them. The
``typing_extensions`` entry is here deliberately: it is the underscore spelling
the distribution actually reports, and normalisation has to survive it.
"""


def script_healthy_cluster(fake: FakeRun, *, job_id: str = "55519937") -> None:
    """Script a cluster that admits and accepts everything.

    Submission now preflights unconditionally, so every submit test needs the
    environment probe and the ``--test-only`` verdict scripted, not only the
    ``sbatch``. Centralised here so that adding a step to the submit path
    updates every test at once rather than one failing test at a time.

    Args:
        fake: The runner to script.
        job_id: Id the real submission should report.
    """
    fake.add("test -d", stdout="PRESENT\n")
    fake.add("importlib.metadata", stdout=ABL_PINNED_DISTRIBUTIONS)
    fake.add("--test-only", stdout=PREFLIGHT_LINE + "\nrc=0\n")
    fake.add("sbatch", stdout=f"Submitted batch job {job_id}\n")


def cluster() -> ClusterFacts:
    """The cluster nearly every test validates against.

    Args:
        None.

    Returns:
        HPC3's measured facts, so a test asserts against the same numbers the
        real machine reported rather than a fixture's invention.
    """
    return HPC3


def gpus(model: str, count: int = 1) -> dict[str, JSONValue]:
    """Build a GPU request payload.

    Args:
        model: GPU model to pin.
        count: GPUs requested. Defaults to one, which is what nearly every
            test wants and what keeps the interesting cases visible.

    Returns:
        The request, ready to place in a spec or project payload. A test
        wanting a CPU-only job writes ``None`` instead of calling this.
    """
    return {"model": model, "count": count}


def project_config(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build one project's resource defaults.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        The defaults, ready to place in a workspace's project table.
    """
    config: dict[str, JSONValue] = {
        "partition": "free-gpu",
        "gpu": gpus("A100"),
        "cpus": 8,
        "mem_gb": 96,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "image": None,
        "env_path": "/pub/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
    }
    config.update(overrides)
    return config


def ledger_row(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build one raw ledger record, before decoding.

    One builder rather than the four near-identical private ones this
    replaced. Those forked the moment the entry grew a field: making
    ``image_digest`` required broke 25 tests across four files that each
    spelled the same row out by hand, and every one of them had to be edited
    to say the same new thing.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        A record ``decode_ledger_entry`` accepts, with both index fields
        explicitly null -- the state of a row that does not record them.

    """
    row: dict[str, JSONValue] = {
        "job_id": "101",
        "project": "abl",
        "name": "abl.arm-b-42",
        "host": "hpc3",
        "partition": "free-gpu",
        "submitted_at": "2026-08-22T16:00:00+00:00",
        "log_dir": "/pub/logs",
        "deterministic": False,
        "experiment": {"arm": "B"},
        "image_digest": None,
        "artifact": None,
    }
    row.update(overrides)
    return row


def budget_document(
    *, gpu_hours: float = 100.0, units: float = 0.0, account: str = ""
) -> dict[str, JSONValue]:
    """Build a budget for a workspace document.

    Args:
        gpu_hours: GPU-hour cap.
        account: Slurm account to bill. Empty by default, pairing with the
            zero cap: a workspace that cannot spend has nothing to spend from.
        units: Service-unit cap. Defaults to zero, which is the free-work-only
            posture and what a workspace has until someone raises it
            deliberately. It defaulted to 1000 while service units were never
            projected and the number could not affect an outcome; now that a
            declared budget is what admits a billed partition, a generous
            default would quietly make every test workspace one that can
            spend.

    Returns:
        The budget object.
    """
    return {
        "max_gpu_hours": gpu_hours,
        "max_service_units": units,
        "charge_account": account,
    }


def workspace_document(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid workspace document.

    Args:
        **overrides: Top-level fields to replace, including ``budget`` (see
            :func:`budget_document`) and ``projects``.

    Returns:
        The document.
    """
    document: dict[str, JSONValue] = {
        "cluster": "hpc3",
        "host": "hpc3",
        "root": "/pub/w",
        "ledger": "ledger.jsonl",
        "quiet_seconds": 1800,
        "budget": budget_document(),
        "projects": {"abl": project_config()},
    }
    document.update(overrides)
    return document


def write_workspace(path: pathlib.Path, document: dict[str, JSONValue] | None = None) -> str:
    """Write a workspace document for a CLI to read.

    Every command now reads one, so nearly every CLI test needs this.
    Centralised so a change to the workspace shape updates all of them at
    once rather than one failing test at a time.

    Args:
        path: File to write.
        document: The document; defaults to :func:`workspace_document`.

    Returns:
        The path as a string, ready to pass as ``--config``.
    """
    payload = document if document is not None else workspace_document()
    write_file(path, dump_json_str(payload).encode("utf-8"))
    return str(path)


def write_file(path: pathlib.Path, payload: bytes) -> None:
    """Write bytes, creating parent directories.

    Args:
        path: File to write.
        payload: Exact bytes to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


# The call form resolves pytest's overloaded decorator to a concrete type;
# the bare @pytest.fixture expression carries Any under disallow_any_expr.
argv = pytest.fixture(_make_argv)
emitted = pytest.fixture(_make_emitted)
errors = pytest.fixture(_make_errors)
fake_run = pytest.fixture(_make_fake_run)
frozen_clock = pytest.fixture(_make_frozen_clock)
logged = pytest.fixture(_make_logged)
