"""Run metadata-only WRDS inventories using the existing cluster transport.

WRDS uses Grid Engine, so this workflow does not forge Slurm ledger entries.
Its receipt retains the real scheduler id, source digest and artifact digest.
The full /wrds filesystem is scanned; SQL-only libraries remain a separate
catalog requiring database authentication.
"""

from __future__ import annotations

import hashlib
import io
import pathlib
import shlex
import zipfile

from platform_core import json_utils
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str, load_json_str

from hpc3.contracts.wrds_catalog import CatalogEntry, decode_catalog_entry
from hpc3.core import _test_hooks, wrds_scan
from hpc3.core.digest import check_remote_digest, parse_remote_digest
from hpc3.core.remote import put_bytes, remote_digest, run_remote


def probe_archive() -> bytes:
    """Package the scanner and its exact shared serializer into a deterministic zipapp.

    Returns:
        Executable archive requiring only Python's standard library on WRDS.
    """
    serializer = pathlib.Path(json_utils.__file__)
    members = {
        "__main__.py": pathlib.Path(wrds_scan.__file__),
        "platform_core/__init__.py": serializer.with_name("__init__.py"),
        "platform_core/json_utils.py": serializer,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, path in members.items():
            archive.writestr(zipfile.ZipInfo(name), _test_hooks.read_bytes(path))
    return buffer.getvalue()


def read_catalog(text: str) -> list[CatalogEntry]:
    """Validate the complete artifact, refusing a lost or premature footer.

    Args:
        text: Probe JSONL artifact.

    Returns:
        All observed paths, excluding the completion marker.

    Raises:
        JSONTypeError: If the artifact is incomplete or has an interior footer.
    """
    records = [decode_catalog_entry(load_json_str(line)) for line in text.splitlines()]
    if not records or records[-1]["kind"] != "complete":
        raise JSONTypeError("WRDS_CATALOG_INCOMPLETE: no completion marker")
    if any(record["kind"] == "complete" for record in records[:-1]):
        raise JSONTypeError("WRDS_CATALOG_INCOMPLETE: premature completion marker")
    return records[:-1]


def inventory(host: str, remote_dir: str, output: pathlib.Path) -> list[CatalogEntry]:
    """Submit a bounded metadata job and save verified catalog and receipt locally.

    Args:
        host: WRDS SSH alias.
        remote_dir: New absolute directory in the user's WRDS home.
        output: New local JSONL path. Its parent must already exist.

    Returns:
        Validated path records. Only metadata is transported.

    Raises:
        ValueError: If paths are unsafe or a scheduler id cannot be read.
        FileExistsError: If output or its receipt exists.
        AppError: If transport or the Grid Engine job fails.
        JSONTypeError: If the returned artifact is incomplete.
    """
    remote = pathlib.PurePosixPath(remote_dir)
    if not remote.is_absolute() or ".." in remote.parts or not str(remote).startswith("/home/"):
        raise ValueError("WRDS_CATALOG_PATH: remote directory must be beneath /home")
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if _test_hooks.file_exists(output) or _test_hooks.file_exists(receipt_path):
        raise FileExistsError(str(output))
    source = probe_archive()
    source_sha = hashlib.sha256(source).hexdigest()
    run_remote(host, shlex.join(["mkdir", "-m", "700", "--", str(remote)]))
    probe = str(remote / "scan.pyz")
    artifact = str(remote / "catalog.jsonl")
    put_bytes(host, probe, source)
    check_remote_digest(probe, source_sha, parse_remote_digest(remote_digest(host, probe), probe))
    command = shlex.join(
        [
            "qsub",
            "-terse",
            "-sync",
            "y",
            "-b",
            "y",
            "-N",
            "wrds_catalog",
            "-l",
            "h_rt=3600",
            "-o",
            str(remote / "job.out"),
            "-e",
            str(remote / "job.err"),
            "python3",
            probe,
            "--root",
            "/wrds",
            "--output",
            artifact,
        ]
    )
    scheduler = run_remote(host, command)
    job_id = scheduler.splitlines()[0] if scheduler.splitlines() else ""
    if not job_id.isdecimal():
        raise ValueError(f"WRDS_JOB_ID: unrecognized scheduler receipt {scheduler!r}")
    text = run_remote(host, shlex.join(["cat", "--", artifact]))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    check_remote_digest(
        artifact, digest, parse_remote_digest(remote_digest(host, artifact), artifact)
    )
    entries = read_catalog(text)
    receipt: dict[str, JSONValue] = {
        "host": host,
        "scheduler": "grid-engine",
        "job_id": job_id,
        "source_sha256": source_sha,
        "catalog_sha256": digest,
        "remote_catalog": artifact,
        "scope": "/wrds filesystem metadata, including historical trees",
        "sql_catalog": "not inventoried; separate database authentication required",
    }
    _test_hooks.write_text(output, text)
    _test_hooks.write_text(receipt_path, dump_json_str(receipt) + "\n")
    return entries


__all__ = ["inventory", "probe_archive", "read_catalog"]
