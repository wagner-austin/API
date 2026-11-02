"""Independent historical metadata and real encrypted relation bytes for revisions."""

from pathlib import Path
from typing import NamedTuple

from platform_core.json_utils import dump_json_str, require_dict
from platform_core.stage_manifest import encode_staged_file

from covenant_ml.markets.crsp_extract import decode_crsp_extract, encode_crsp_extract
from covenant_ml.markets.encrypted_storage import encrypt_bundle
from covenant_ml.markets.research_keys import ResearchAccess

from ._crsp_fixtures import panel_protocol, source_rows, write_sources
from ._encrypted_fixtures import create_access, file_record
from .test_crsp_extraction_records import receipt_document


class RevisionCase(NamedTuple):
    """Public revision inputs; keys are separate test-owned synthetic files."""

    outputs: Path
    receipt_digest: str
    previous: Path
    protocol: Path
    access: ResearchAccess


def revision_case(root: Path, *, unchanged: bool = False) -> RevisionCase:
    """Seal real files with historical rule metadata and independently encoded manifests."""
    access = create_access(root)
    current = panel_protocol()
    protocol = root / "current.json"
    protocol.write_text(dump_json_str(current))
    old = dict(current)
    if not unchanged:
        old["version"] = 2
        del old["held_return_rule"]
    previous = root / "previous.json"
    previous.write_text(dump_json_str(old))
    outputs = root / "outputs"
    outputs.mkdir()
    (outputs / "metrics.json").write_text("[]")
    receipt = receipt_document()
    receipt["started_at"] = "2026-10-29T00:00:00+00:00"
    receipt["completed_at"] = "2026-10-29T00:01:00+00:00"
    rows = source_rows()
    for key, partition, recipient in (
        ("train_validation", "train-validation", access["validation_recipient"]),
        ("final", "final", access["final_recipient"]),
    ):
        source = root / (partition + "-source")
        source.mkdir()
        manifest, document = write_sources(source, rows, old)
        manifest.write_text(dump_json_str(encode_crsp_extract(decode_crsp_extract(document))))
        files = tuple(file_record(path) for path in sorted(source.iterdir()))
        encrypt_bundle(
            source,
            outputs / partition,
            files=files,
            manifest=file_record(manifest),
            recipient=recipient,
            partition=partition,
        )
        item = require_dict(receipt, key)
        item["bundle"] = encode_staged_file(file_record(outputs / partition / "bundle.json"))
        item["manifest"] = encode_staged_file(file_record(manifest))
        item["plaintext_bytes"] = sum(file["size_bytes"] for file in files)
        for role, records in rows.items():
            item[role + "_rows"] = len(records)
        receipt["protocol"] = encode_staged_file(file_record(source / "protocol.json"))
    receipt_path = outputs / "extraction.json"
    receipt_path.write_text(dump_json_str(receipt))
    return RevisionCase(outputs, file_record(receipt_path)["sha256"], previous, protocol, access)


__all__ = ["RevisionCase", "revision_case"]
