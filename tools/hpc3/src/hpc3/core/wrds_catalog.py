"""Run metadata-only WRDS inventories using the existing cluster transport.

WRDS uses Grid Engine, so this workflow does not forge Slurm ledger entries.
Its receipt retains the real scheduler id, source digest and artifact digest.
The full /wrds filesystem is scanned; SQL-only libraries are the separate
catalog :mod:`hpc3.core.wrds_sql` reads.
"""

from __future__ import annotations

import io
import pathlib
import zipfile

from platform_core import json_utils
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from hpc3.contracts.wrds_catalog import (
    CatalogEntry,
    CatalogReceipt,
    CatalogRun,
    decode_catalog_entry,
    encode_catalog_receipt,
)
from hpc3.core import _test_hooks, artifacts, wrds_grid, wrds_scan


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


def inventory(host: str, remote_dir: str, output: pathlib.Path) -> CatalogRun:
    """Submit a bounded metadata job and save verified catalog and receipt locally.

    Args:
        host: WRDS SSH alias.
        remote_dir: New absolute directory in the user's WRDS home.
        output: New local JSONL path. Its parent must already exist.

    Returns:
        Validated path records and the saved receipt. Only metadata is
        transported.

    Raises:
        ValueError: If paths are unsafe or a scheduler id cannot be read.
        FileExistsError: If output or its receipt exists.
        AppError: If transport or the Grid Engine job fails.
        JSONTypeError: If the returned artifact is incomplete.
    """
    remote = wrds_grid.remote_directory(remote_dir)
    wrds_grid.refuse_existing([output, wrds_grid.receipt_path(output)])
    wrds_grid.create_directory(host, remote)
    probe = str(remote / "scan.pyz")
    remote_catalog = str(remote / "catalog.jsonl")
    source_sha = wrds_grid.upload(host, probe, probe_archive())
    command = ["python3", probe, "--root", "/wrds", "--output", remote_catalog]
    job_id = wrds_grid.submit(host, remote, "wrds_catalog", command, environment=False)
    artifact = artifacts.download(host, remote_catalog)
    entries = read_catalog(artifact["text"])
    receipt = CatalogReceipt(
        host=host,
        scheduler="grid-engine",
        job_id=job_id,
        source_sha256=source_sha,
        catalog_sha256=artifact["sha256"],
        remote_catalog=remote_catalog,
        scope="/wrds filesystem metadata, including historical trees",
    )
    _test_hooks.write_text(output, artifact["text"])
    _test_hooks.write_text(
        wrds_grid.receipt_path(output), dump_json_str(encode_catalog_receipt(receipt)) + "\n"
    )
    return CatalogRun(entries=entries, receipt=receipt)


__all__ = ["inventory", "probe_archive", "read_catalog"]
