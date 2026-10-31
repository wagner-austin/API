"""Extract both CRSP partitions in one snapshot, verify them and publish only ciphertext.

Plaintext exists only in a new mode-0700 directory under the scheduler's
private scratch. Nothing reaches the output directory until every query has
succeeded, both partitions have passed the existing reader and the partition
check, and both bundles are encrypted; the bundles are then renamed into place
and the receipt is written last. A refusal at any step leaves no usable
bundle, and the private directory is removed as the error propagates.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory

from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.run_record import Observation, encode_observation
from platform_core.stage_manifest import StagedFile

from ..evaluation.evidence import require_private_directory
from . import _test_hooks as hooks
from .crsp_extract import CrspExtract, CrspRelation, encode_crsp_extract, load_crsp_sources
from .crsp_extraction_records import (
    CRSP_RELATIONS,
    CrspExtractionReceipt,
    CrspLimits,
    CrspPartitionReceipt,
    encode_crsp_extraction_receipt,
)
from .crsp_partition import check_crsp_partition
from .crsp_protocol import CrspProtocol, decode_crsp_protocol
from .crsp_queries import (
    CrspScope,
    CrspSource,
    crsp_driver_sql,
    crsp_relation_sql,
    crsp_scopes,
    synthetic_source,
    vendor_source,
)
from .encrypted_storage import encrypt_bundle, staged_record
from .research_keys import ResearchAccess

#: psql without startup files or password prompts, stopping at the first error.
PSQL = ("psql", "-X", "-w", "-q", "-v", "ON_ERROR_STOP=1", "-f", "extract.sql")
#: The source name of a session whose relations are inline synthetic definitions.
SYNTHETIC_SOURCE = "synthetic"


def _record(path: Path) -> StagedFile:
    """Hash one written file by streaming it."""
    with path.open("rb") as stream:
        return staged_record(stream, path.name)


def _snapshot(root: Path) -> tuple[str, str]:
    """Require identical snapshot readings from the transaction's first and last statements.

    Returns:
        The snapshot and the transaction start time, as the server printed them.

    Raises:
        ValueError: CRSP-011 when the readings differ or are malformed.
    """
    readings = [
        list(csv.reader(io.StringIO((root / name).read_text("utf-8"))))
        for name in ("snapshot-first.csv", "snapshot-last.csv")
    ]
    if readings[0] != readings[1] or len(readings[0]) != 1 or len(readings[0][0]) != 2:
        raise ValueError("[CRSP-011] all relations must come from one database snapshot")
    return readings[0][0][0], readings[0][0][1]


def _seal_manifest(directory: Path, extracted_at: str, limits: CrspLimits) -> StagedFile:
    """Write the existing four-relation manifest after bounding the plaintext.

    Raises:
        ValueError: CRSP-012 when the CSVs exceed the partition budget, which is
            checked before the whole-file reader loads them.
    """
    relations = {
        name: CrspRelation(
            csv=_record(directory / (name + ".csv")), query=_record(directory / (name + ".sql"))
        )
        for name in CRSP_RELATIONS
    }
    plaintext = sum(item["csv"]["size_bytes"] for item in relations.values())
    if plaintext > limits["max_partition_bytes"]:
        raise ValueError("[CRSP-012] partition plaintext exceeds its byte budget")
    manifest = CrspExtract(
        extracted_at=extracted_at,
        protocol=_record(directory / "protocol.json"),
        membership=relations["membership"],
        returns=relations["returns"],
        calendar=relations["calendar"],
        delists=relations["delists"],
    )
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(encode_crsp_extract(manifest)))
    return _record(directory / "manifest.json")


def _partition(
    directory: Path, scope: CrspScope, extracted_at: str, limits: CrspLimits
) -> tuple[StagedFile, tuple[StagedFile, ...], tuple[int, ...]]:
    """Seal, reread and check one partition before anything is encrypted.

    Returns:
        The manifest identity, every file to encrypt and the four row counts.
    """
    manifest = _seal_manifest(directory, extracted_at, limits)
    sources = load_crsp_sources(directory / "manifest.json")
    check_crsp_partition(sources, scope, limits["max_rows"])
    files = tuple(_record(path) for path in sorted(directory.iterdir()))
    counts = (
        len(sources.membership),
        len(sources.returns),
        len(sources.calendar),
        len(sources.delists),
    )
    return manifest, files, counts


def _prepare(
    root: Path,
    protocol_bytes: bytes,
    protocol: CrspProtocol,
    scopes: tuple[CrspScope, CrspScope],
    source: CrspSource,
    rows: int,
) -> None:
    """Write each partition's exact protocol copy and relation queries."""
    for scope in scopes:
        directory = root / scope.partition
        directory.mkdir(mode=0o700)
        (directory / "protocol.json").write_bytes(protocol_bytes)
        for name, sql in crsp_relation_sql(protocol, scope, source, rows).items():
            (directory / (name + ".sql")).write_text(sql, encoding="utf-8")


def _refuse_existing(output: Path, scopes: tuple[CrspScope, CrspScope]) -> None:
    """Refuse before any query when a published or pending name already exists."""
    names = ["extraction.json", "metrics.json"]
    for scope in scopes:
        names += [scope.partition, ".pending-" + scope.partition]
    for name in names:
        if (output / name).exists():
            raise FileExistsError(f"[CRSP-011] refusing to replace {name}")


def _encrypt(
    root: Path,
    output: Path,
    scopes: tuple[CrspScope, CrspScope],
    access: ResearchAccess,
    sealed: list[tuple[StagedFile, tuple[StagedFile, ...], tuple[int, ...]]],
) -> list[StagedFile]:
    """Encrypt each checked partition to its own recipient under a pending name.

    Returns:
        Each pending bundle's metadata identity, in partition order.
    """
    recipients = (access["validation_recipient"], access["final_recipient"])
    bundles: list[StagedFile] = []
    for scope, recipient, (manifest, files, _) in zip(scopes, recipients, sealed, strict=True):
        pending = output / (".pending-" + scope.partition)
        encrypt_bundle(
            root / scope.partition,
            pending,
            files=files,
            manifest=manifest,
            recipient=recipient,
            partition=scope.partition,
        )
        bundles.append(_record(pending / "bundle.json"))
    return bundles


def extract_crsp(
    protocol_path: Path,
    access: ResearchAccess,
    *,
    source: str,
    setup: Path | None,
    scratch: Path,
    output: Path,
    limits: CrspLimits,
) -> CrspExtractionReceipt:
    """Produce the train-validation and final bundles from one read-only snapshot.

    Args:
        protocol_path: Frozen protocol; its exact bytes enter both partitions.
        access: Evaluator access settings; only the public recipients are used.
        source: The protocol's vendor library, or ``synthetic``.
        setup: Synthetic ``WITH`` definitions, required exactly for ``synthetic``.
        scratch: Scheduler-owned private scratch for plaintext.
        output: Existing directory receiving only ciphertext and metadata.
        limits: Row, byte and time bounds.

    Returns:
        The receipt also written to ``extraction.json``.

    Raises:
        ValueError: CRSP-010 for an invalid source or future coverage, CRSP-011
            for a failed client or inconsistent snapshot, CRSP-012 for evidence
            exceeding a limit or lying outside its partition; reader, calendar
            and encryption refusals propagate.
        FileExistsError: A published name already exists; nothing is replaced.
        TimeoutExpired: The client exceeded ``limits["seconds"]``.
        OSError: Storage fails; private plaintext is removed on unwinding.
    """
    if (source == SYNTHETIC_SOURCE) != (setup is not None):
        raise ValueError("[CRSP-010] definitions are required exactly for a synthetic source")
    protocol_bytes = protocol_path.read_bytes()
    protocol = decode_crsp_protocol(narrow_json_to_dict(load_json_str(protocol_bytes.decode())))
    setup_sha256: str | None
    if setup is None:
        if source != protocol["library"]:
            raise ValueError("[CRSP-010] the source must be the protocol's library")
        relations = vendor_source(protocol)
        setup_sha256 = None
    else:
        relations = synthetic_source(setup.read_text("utf-8"))
        setup_sha256 = _record(setup)["sha256"]
    started = hooks.clock()
    if protocol["coverage_end"] >= started[:10]:
        raise ValueError("[CRSP-010] coverage must end before the extraction day")
    scopes = crsp_scopes(protocol)
    _refuse_existing(output, scopes)
    with TemporaryDirectory(prefix="research-plaintext-", dir=scratch) as temporary:
        root = Path(temporary)
        require_private_directory(root)
        (root / "extract.sql").write_text(crsp_driver_sql(limits["seconds"]), encoding="utf-8")
        _prepare(root, protocol_bytes, protocol, scopes, relations, limits["max_rows"])
        code = hooks.bounded_runner(
            PSQL, root, seconds=limits["seconds"], max_file_bytes=limits["max_relation_bytes"]
        )
        if code != 0:
            raise ValueError(f"[CRSP-011] psql exited {code}; no bundle was published")
        completed = hooks.clock()
        snapshot, transaction = _snapshot(root)
        sealed = [_partition(root / scope.partition, scope, started, limits) for scope in scopes]
        bundles = _encrypt(root, output, scopes, access, sealed)
        driver_sha256 = _record(root / "extract.sql")["sha256"]
    for scope in scopes:
        (output / (".pending-" + scope.partition)).rename(output / scope.partition)
    partitions = [
        CrspPartitionReceipt(
            partition=scope.partition,
            observation_end=scope.observation_end,
            calendar_end=scope.calendar_end,
            manifest=manifest,
            bundle=bundle,
            membership_rows=counts[0],
            returns_rows=counts[1],
            calendar_rows=counts[2],
            delists_rows=counts[3],
            plaintext_bytes=sum(item["size_bytes"] for item in files),
        )
        for scope, bundle, (manifest, files, counts) in zip(scopes, bundles, sealed, strict=True)
    ]
    receipt = CrspExtractionReceipt(
        source=source,
        setup_sha256=setup_sha256,
        protocol=_record(protocol_path),
        driver_sha256=driver_sha256,
        started_at=started,
        completed_at=completed,
        snapshot=snapshot,
        transaction_started_at=transaction,
        limits=limits,
        train_validation=partitions[0],
        final=partitions[1],
        plaintext_removed=not root.exists(),
    )
    observations = [
        Observation(name=f"{scope.partition.replace('-', '_')}_{name}_rows", value=float(count))
        for scope, (_, _, counts) in zip(scopes, sealed, strict=True)
        for name, count in zip(CRSP_RELATIONS, counts, strict=True)
    ]
    observations.append(Observation(name="plaintext_removed", value=float(not root.exists())))
    with (output / "metrics.json").open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str([encode_observation(item) for item in observations]))
    with (output / "extraction.json").open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(encode_crsp_extraction_receipt(receipt)))
    return receipt


__all__ = ["PSQL", "SYNTHETIC_SOURCE", "extract_crsp"]
