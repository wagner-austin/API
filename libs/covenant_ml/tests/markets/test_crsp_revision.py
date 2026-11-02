"""A revision proves old metadata, preserves ciphertext and never needs final keys."""

from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)
from platform_core.stage_manifest import encode_staged_file

from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.crsp_revision import revise_crsp_protocol, verify_crsp_revision
from covenant_ml.markets.crsp_revision_records import (
    decode_crsp_protocol_revision,
    encode_crsp_protocol_revision,
)
from covenant_ml.markets.encrypted_storage import decrypted_bundle
from covenant_ml.markets.validation import read_document

from ._crsp_revision_fixtures import revision_case
from ._encrypted_fixtures import evaluator_keys, file_record


def test_metadata_revision_preserves_originals_and_needs_no_keys(tmp_path: Path) -> None:
    case = revision_case(tmp_path)
    keys = evaluator_keys(tmp_path)
    identities = (keys.acquire("train-validation"), keys.acquire("final"))
    for path in (tmp_path / "keys").iterdir():
        path.unlink()
    before = {
        str(path.relative_to(case.outputs)): path.read_bytes()
        for path in case.outputs.rglob("*")
        if path.is_file()
    }
    destination = tmp_path / "revision"
    revision = revise_crsp_protocol(*case, destination)
    assert decode_crsp_protocol_revision(encode_crsp_protocol_revision(revision)) == revision
    report = verify_crsp_revision(*case, destination)
    assert report["kind"] == "CRSP_PROTOCOL_REVISION_EVIDENCE"
    assert report["opened_observations"] is False
    assert report["retained_relation_ciphertexts"] == 16
    assert report["revision_sha256"] == file_record(destination / "revision.json")["sha256"]
    assert revision["extraction_sha256"] == case.receipt_digest
    assert revision["extracted_at"] == "2026-10-29T00:00:00+00:00"
    after = {
        str(path.relative_to(case.outputs)): path.read_bytes()
        for path in case.outputs.rglob("*")
        if path.is_file()
    }
    assert after == before
    for partition, item, identity in zip(
        ("train-validation", "final"),
        (revision["train_validation"], revision["final"]),
        identities,
        strict=True,
    ):
        assert item["current"]["sha256"] != item["previous"]["sha256"]
        assert item["manifest"]["sha256"] != item["previous_manifest"]["sha256"]
        with decrypted_bundle(
            destination / partition / "bundle.json",
            item["current"]["sha256"],
            partition=partition,
            identity=identity,
            scratch=tmp_path,
            max_plaintext_bytes=10_000_000,
        ) as manifest:
            sources = load_crsp_sources(manifest)
            assert sources.protocol["version"] == 3
            assert len(sources.membership) == 3
            assert sources.manifest["extracted_at"] == revision["extracted_at"]
    assert not list(tmp_path.glob("crsp-revision-metadata-*"))
    assert not list(tmp_path.glob("research-plaintext-*"))


@pytest.mark.parametrize(
    "fault", ["receipt-pin", "old-protocol", "same", "date", "old-extra", "recipient", "occupied"]
)
def test_incompatible_or_unpinned_sources_never_publish_revision(
    tmp_path: Path, fault: str
) -> None:
    case = revision_case(tmp_path, unchanged=fault == "same")
    destination = tmp_path / "refused"
    digest = case.receipt_digest
    access = case.access
    if fault == "receipt-pin":
        digest = "0" * 64
    elif fault == "old-protocol":
        case.previous.write_text("{}")
    elif fault == "date":
        document = read_document(case.protocol)
        document["history_start"] = "2017-01-01"
        case.protocol.write_text(dump_json_str(document))
    elif fault == "old-extra":
        document = read_document(case.previous)
        document["unexpected"] = True
        case.previous.write_text(dump_json_str(document))
        receipt = read_document(case.outputs / "extraction.json")
        record = file_record(case.previous)
        record["name"] = "protocol.json"
        receipt["protocol"] = encode_staged_file(record)
        (case.outputs / "extraction.json").write_text(dump_json_str(receipt))
        digest = file_record(case.outputs / "extraction.json")["sha256"]
    elif fault == "recipient":
        access = {**case.access, "validation_recipient": case.access["final_recipient"]}
    elif fault == "occupied":
        destination.mkdir()
        (destination / "keep").write_text("keep")
    with pytest.raises((ValueError, FileExistsError)):
        revise_crsp_protocol(
            case.outputs, digest, case.previous, case.protocol, access, destination
        )
    assert not (destination / "revision.json").exists()
    if fault == "occupied":
        assert (destination / "keep").read_text() == "keep"


@pytest.mark.parametrize(
    "field", ["bundle-size", "protocol-pin", "manifest-pin", "missing-relation"]
)
def test_original_bundle_must_reconstruct_the_exact_recorded_manifest(
    tmp_path: Path, field: str
) -> None:
    case = revision_case(tmp_path)
    path = case.outputs / "train-validation/bundle.json"
    bundle = read_document(path)
    receipt_path = case.outputs / "extraction.json"
    receipt = read_document(receipt_path)
    item = require_dict(receipt, "train_validation")
    files = [narrow_json_to_dict(value) for value in require_list(bundle, "files")]
    if field == "bundle-size":
        require_dict(item, "bundle")["size_bytes"] = 1
    else:
        name = "protocol.json" if field == "protocol-pin" else "manifest.json"
        if field == "missing-relation":
            files = [
                value
                for value in files
                if require_dict(value, "plaintext")["name"] != "returns.csv"
            ]
            (path.parent / "returns.csv.age").unlink()
            bundle["files"] = list[JSONValue](files)
        else:
            for value in files:
                if require_dict(value, "plaintext")["name"] == name:
                    require_dict(value, "plaintext")["sha256"] = "0" * 64
            if field == "manifest-pin":
                require_dict(bundle, "manifest")["sha256"] = "0" * 64
                require_dict(item, "manifest")["sha256"] = "0" * 64
        path.write_text(dump_json_str(bundle))
        item["bundle"] = encode_staged_file(file_record(path))
    receipt_path.write_text(dump_json_str(receipt))
    with pytest.raises(ValueError, match=r"CRSP_REVISION|CRSP-006"):
        revise_crsp_protocol(
            case.outputs,
            file_record(receipt_path)["sha256"],
            case.previous,
            case.protocol,
            case.access,
            tmp_path / "refused",
        )
    assert not (tmp_path / "refused").exists()
