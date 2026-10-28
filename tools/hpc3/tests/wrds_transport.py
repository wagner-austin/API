"""Script exact WRDS metadata replies while retaining real local file effects."""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Sequence

from platform_core.json_utils import JSONObject, dump_json_str

from hpc3.contracts.wrds_run import WrdsRun, encode_wrds_runtime, execution_document, run_identity
from hpc3.core import _test_hooks
from hpc3.core._test_hooks import CommandResult
from hpc3.core.wrds_bundle import worker_bundle
from hpc3.core.wrds_jobs import attempt_directory
from tests.conftest import FakeRun
from tests.test_wrds_scheduler import DONE, EMPTY, HELD

NOW = "2026-10-28T01:00:00+00:00"


def metadata(fake: FakeRun, path: str, text: str) -> None:
    """Provide cat and SHA256 replies for the same exact bytes.

    Args:
        fake: Existing process seam fake.
        path: Remote metadata file.
        text: File contents.
    """
    fake.add("cat -- " + path, stdout=text)
    fake.add(
        "sha256sum '" + path + "'",
        stdout=hashlib.sha256(text.encode()).hexdigest() + "  " + path + "\n",
    )


def prepare(fake: FakeRun, run: WrdsRun, source: pathlib.Path, attempt: str = "one") -> str:
    """Provide admission, staged-byte digests and a held scheduler receipt.

    Args:
        fake: Transport fake.
        run: Experiment.
        source: Real local code directory.
        attempt: Attempt label.

    Returns:
        Remote attempt directory.
    """
    directory = attempt_directory(run, attempt)
    probe = run["root"] + "/tools/probe.123"
    worker_sha = hashlib.sha256(worker_bundle()).hexdigest()
    fake.add("mktemp", stdout=probe + "\n")
    fake.add(" inspect", stdout=dump_json_str(encode_wrds_runtime(run["runtime"])))
    for path in (probe + "/worker.pyz", directory + "/worker.pyz"):
        fake.add("sha256sum '" + path + "'", stdout=worker_sha + "  " + path + "\n")
    for item in run["code"]:
        path = directory + "/code/" + item["name"]
        fake.add("sha256sum '" + path + "'", stdout=item["sha256"] + "  " + path + "\n")
    text = dump_json_str(execution_document(run)) + "\n"
    metadata(fake, directory + "/run.json", text)
    submission: JSONObject = {
        "identity": run_identity(run),
        "attempt": attempt,
        "worker_sha256": worker_sha,
        "created_at": NOW,
    }
    metadata(fake, directory + "/submission.json", dump_json_str(submission) + "\n")
    fake.add("cat " + directory + "/job.id", stdout="123\n")
    fake.add("qstat", stdout=HELD)
    return directory


class CopyTransport(FakeRun):
    """The scripted scheduler plus actual file copying for collect tests."""

    def __init__(self, outputs: pathlib.Path) -> None:
        super().__init__()
        self.outputs = outputs

    def __call__(self, argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
        result = super().__call__(argv, stdin_bytes=stdin_bytes)
        if argv[0] == "scp" and result["returncode"] == 0:
            target = pathlib.Path(argv[-1])
            target.write_bytes((self.outputs / target.name).read_bytes())
        return result


def completed(fake: FakeRun, run: WrdsRun, result: JSONObject, attempt: str = "one") -> str:
    """Prepare terminal accounting and a matching worker attestation.

    Args:
        fake: Transport fake.
        run: Experiment definition.
        result: Real worker completion, or a deliberately corrupted copy.
        attempt: Attempt label.

    Returns:
        Remote directory.
    """
    directory = attempt_directory(run, attempt)
    metadata(fake, directory + "/run.json", dump_json_str(execution_document(run)) + "\n")
    metadata(fake, directory + "/completion.json", dump_json_str(result) + "\n")
    submission: JSONObject = {
        "identity": run_identity(run),
        "attempt": attempt,
        "worker_sha256": hashlib.sha256(b"fixture worker identity").hexdigest(),
    }
    metadata(fake, directory + "/submission.json", dump_json_str(submission))
    fake.add("cat " + directory + "/job.id", stdout="123\n")
    fake.add("qstat", stdout=EMPTY)
    fake.add("qacct", stdout=DONE)
    _test_hooks.run = fake
    return directory


__all__ = ["NOW", "CopyTransport", "completed", "metadata", "prepare"]
