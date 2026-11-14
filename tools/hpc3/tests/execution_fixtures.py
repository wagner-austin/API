"""Benchmark artifacts served through the existing remote-command test seam."""

from __future__ import annotations

import shlex
from datetime import UTC, datetime

from platform_core.comparability import encode_run_fingerprint
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str
from platform_core.run_record import Observation, encode_run_record, run_record
from platform_core.testing import sample_run_fingerprint

from hpc3.contracts.ledger import LedgerEntry, decode_ledger_entry
from hpc3.core.digest import sha256_hex
from hpc3.core.logs import log_path
from tests.conftest import FakeRun, cluster, ledger_row

IMAGE = "a" * 64
START = "2026-10-28T16:54:02"
END = "2026-10-28T16:54:13"
INSTANT = int(datetime.fromisoformat(START).replace(tzinfo=UTC).timestamp()) + 5
CHECKSUMS = {"/pub/runtime.sif": IMAGE}


def entry(job: str = "101", /, **changes: JSONValue) -> LedgerEntry:
    """Build a recorded benchmark submission.

    Args:
        job: Allocation id.
        **changes: Explicit invalid or alternate ledger fields.

    Returns:
        Decoded ledger entry.
    """
    fields: JSONObject = {
        "job_id": job,
        "artifact": f"/pub/result-{job}.json",
        "image_digest": IMAGE,
        "experiment": {"arm": "B", "repetition": job},
    }
    fields.update(changes)
    return decode_ledger_entry(ledger_row(**fields), cluster())


def accounting_row(value: LedgerEntry, **changes: str) -> str:
    """Render a native allocation row with explicit fault overrides.

    Args:
        value: Submission being measured.
        **changes: Accounting column replacements.

    Returns:
        Pipe-delimited sacct output.
    """
    columns = {
        "job": value["job_id"],
        "name": value["name"],
        "partition": value["partition"],
        "state": "COMPLETED",
        "elapsed": "11",
        "tres": "billing=2,cpu=2,mem=4G,node=1",
        "node": "hpc3-17-01",
        "exit": "0:0",
        "start": START,
        "end": END,
    }
    columns.update(changes)
    return "|".join(columns.values()) + "\n"


def documents(
    *,
    score: float = 0.25,
    image: str = IMAGE,
    driver: str = "580.82.07",
    observations: tuple[Observation, ...] | None = None,
) -> tuple[JSONObject, JSONObject]:
    """Build the benchmark's manifest and shared record through their real codecs.

    Args:
        score: Result and observation value.
        image: Runtime fingerprint identity.
        driver: Independently varied comparability axis.
        observations: Explicit alternate observations.

    Returns:
        Manifest and sidecar JSON documents.
    """
    fingerprint = sample_run_fingerprint(image_digest=image, driver_version=driver)
    results: list[JSONValue] = [{"score": score}]
    manifest: JSONObject = {
        "results": results,
        "fingerprint": encode_run_fingerprint(fingerprint),
        "dataset": {"sha256": "b" * 64, "n_rows": 5},
    }
    measured = (Observation(name="score", value=score),) if observations is None else observations
    record = run_record(
        "paired-benchmark",
        "test",
        fingerprint,
        measured,
        sha256_hex(dump_json_str(results).encode()),
    )
    return manifest, encode_run_record(record)


def serve_text(fake: FakeRun, path: str, text: str) -> None:
    """Serve one verified text artifact.

    Args:
        fake: Existing command seam.
        path: Remote absolute path.
        text: Exact returned content.
    """
    fake.add(shlex.join(["cat", "--", path]), stdout=text, once=True)
    fake.add(f"sha256sum '{path}'", stdout=f"{sha256_hex(text.encode())}  {path}\n")


def serve(
    fake: FakeRun,
    value: LedgerEntry,
    manifest: JSONObject,
    record: JSONObject,
    *,
    instant: int = INSTANT,
    checksum_log: str = "/pub/runtime.sif: OK\n",
) -> None:
    """Serve all measured allocation, artifacts and input identities.

    Args:
        fake: Existing command seam.
        value: Recorded submission.
        manifest: Benchmark results document.
        record: Shared numerical record.
        instant: Result modification time.
        checksum_log: Captured compute-side checks.
    """
    path = value["artifact"]
    assert path is not None
    fake.add(f"-j {value['job_id']} ", stdout=accounting_row(value))
    serve_text(fake, path, dump_json_str(manifest))
    serve_text(fake, path + ".runrecord.json", dump_json_str(record))
    fake.add(f"stat -c %Y -- {path}", stdout=str(instant))
    serve_text(fake, log_path(value), checksum_log)
    fake.add("sha256sum '/pub/runtime.sif'", stdout=f"{IMAGE}  /pub/runtime.sif\n")


#: The one Slurm name every member of the fixture array runs under.
ARRAY_NAME = "abl.sweep"


def array_member(index: int, array: str = "101", /, **changes: JSONValue) -> LedgerEntry:
    """A recorded member of an array, as array submission records it.

    Its ledger name is the array's plus ``-shard<index>``, its experiment
    records that suffix as ``member``, and its artifact is
    ``/pub/out/shard-<index>.json`` whichever array ran it, as a campaign's
    resubmission of the same member declares the same artifact.

    Args:
        index: Its member number.
        array: The array's id, 101 unless a case needs a second array.
        **changes: Alternate ledger fields.

    Returns:
        The decoded entry.
    """
    fields: JSONObject = {
        "name": f"{ARRAY_NAME}-shard{index}",
        "experiment": {"member": f"shard{index}"},
        "artifact": f"/pub/out/shard-{index}.json",
    }
    fields.update(changes)
    return entry(f"{array}_{index}", **fields)


def serve_member(fake: FakeRun, value: LedgerEntry, data: bytes, **accounting: str) -> None:
    """Serve one member's accounting row under the array's name, digest and instant.

    Args:
        fake: The remote-command seam.
        value: The member.
        data: Its artifact's bytes on the cluster.
        **accounting: Accounting column replacements.
    """
    path = value["artifact"]
    assert path is not None
    columns = {"name": ARRAY_NAME, **accounting}
    fake.add(f"-j {value['job_id']} ", stdout=accounting_row(value, **columns))
    fake.add(f"sha256sum '{path}'", stdout=f"{sha256_hex(data)}  {path}\n")
    fake.add(f"stat -c %Y -- {path}", stdout=str(INSTANT))


__all__ = [
    "ARRAY_NAME",
    "CHECKSUMS",
    "END",
    "IMAGE",
    "INSTANT",
    "START",
    "accounting_row",
    "array_member",
    "documents",
    "entry",
    "serve",
    "serve_member",
    "serve_text",
]
