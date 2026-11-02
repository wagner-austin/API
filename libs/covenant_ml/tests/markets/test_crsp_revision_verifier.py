"""The metadata-only reread rejects independently altered transformation claims."""

from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str, narrow_json_to_dict, require_dict, require_list
from platform_core.stage_manifest import encode_staged_file

from covenant_ml.markets.crsp_extract import crsp_extract_from_records
from covenant_ml.markets.crsp_revision import revise_crsp_protocol, verify_crsp_revision
from covenant_ml.markets.crsp_revision_records import decode_crsp_protocol_revision
from covenant_ml.markets.validation import read_document

from ._crsp_revision_fixtures import revision_case
from ._encrypted_fixtures import file_record


@pytest.mark.parametrize("fault", ["extra", "root-link", "receipt-link", "receipt-directory"])
def test_revision_root_is_an_exact_regular_inventory(tmp_path: Path, fault: str) -> None:
    case = revision_case(tmp_path)
    destination = tmp_path / "revised"
    revise_crsp_protocol(*case, destination)
    receipt = destination / "revision.json"
    if fault == "extra":
        (destination / "extra").write_text("extra")
    elif fault == "root-link":
        destination.rename(tmp_path / "elsewhere")
        destination.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    elif fault == "receipt-link":
        receipt.rename(tmp_path / "receipt-copy")
        receipt.symlink_to(tmp_path / "receipt-copy")
    else:
        receipt.unlink()
        receipt.mkdir()
    with pytest.raises(ValueError, match="CRSP_REVISION_INVENTORY"):
        verify_crsp_revision(*case, destination)


@pytest.mark.parametrize(
    "field",
    [
        "extraction_sha256",
        "extracted_at",
        "previous_protocol",
        "protocol",
        "previous",
        "previous_manifest",
        "current",
    ],
)
def test_revision_receipt_is_checked_against_original_inputs(tmp_path: Path, field: str) -> None:
    case = revision_case(tmp_path)
    destination = tmp_path / "revised"
    revise_crsp_protocol(*case, destination)
    path = destination / "revision.json"
    document = read_document(path)
    if field == "extraction_sha256":
        document[field] = "0" * 64
    elif field == "extracted_at":
        document[field] = "2026-10-30T00:00:00+00:00"
    elif field in {"previous_protocol", "protocol"}:
        require_dict(document, field)["size_bytes"] = 1
    else:
        require_dict(require_dict(document, "train_validation"), field)["size_bytes"] = 1
    path.write_text(dump_json_str(document))
    with pytest.raises(ValueError, match="CRSP_REVISION_LINEAGE"):
        verify_crsp_revision(*case, destination)


@pytest.mark.parametrize(
    "fault", ["partition", "recipient", "manifest", "inventory", "retained", "protocol"]
)
def test_revised_bundle_cannot_change_its_original_data_or_metadata(
    tmp_path: Path, fault: str
) -> None:
    case = revision_case(tmp_path)
    destination = tmp_path / "revised"
    revise_crsp_protocol(*case, destination)
    path = destination / "train-validation/bundle.json"
    document = read_document(path)
    files = [narrow_json_to_dict(value) for value in require_list(document, "files")]
    if fault == "partition":
        document["partition"] = "final"
    elif fault == "recipient":
        document["recipient"] = case.access["final_recipient"]
    elif fault == "inventory":
        document["files"] = [
            value for value in files if require_dict(value, "plaintext")["name"] != "returns.csv"
        ]
        (path.parent / "returns.csv.age").unlink()
    else:
        name = {
            "manifest": "manifest.json",
            "retained": "returns.csv",
            "protocol": "protocol.json",
        }[fault]
        for value in files:
            if require_dict(value, "plaintext")["name"] == name:
                if fault == "retained":
                    ciphertext = path.parent / "returns.csv.age"
                    raw = ciphertext.read_bytes()
                    ciphertext.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
                    value["ciphertext"] = encode_staged_file(file_record(ciphertext))
                else:
                    require_dict(value, "plaintext")["sha256"] = "0" * 64
        if fault == "manifest":
            require_dict(document, "manifest")["sha256"] = "0" * 64
    path.write_text(dump_json_str(document))
    receipt = destination / "revision.json"
    lineage = read_document(receipt)
    require_dict(lineage, "train_validation")["current"] = encode_staged_file(file_record(path))
    receipt.write_text(dump_json_str(lineage))
    with pytest.raises(ValueError, match=r"CRSP_REVISION_LINEAGE|CRSP_REVISION_FILES"):
        verify_crsp_revision(*case, destination)


@pytest.mark.parametrize("fault", ["extra", "nested-extra", "time"])
def test_revision_codec_refuses_unknown_fields_and_naive_time(tmp_path: Path, fault: str) -> None:
    case = revision_case(tmp_path)
    destination = tmp_path / "revised"
    revise_crsp_protocol(*case, destination)
    document = read_document(destination / "revision.json")
    if fault == "extra":
        document["extra"] = True
    elif fault == "nested-extra":
        require_dict(document, "final")["extra"] = True
    else:
        document["extracted_at"] = "2026-10-29T00:00:00"
    with pytest.raises(ValueError, match=r"CRSP_REVISION_FIELDS|CRSP_REVISION_TIME"):
        decode_crsp_protocol_revision(document)


def test_canonical_manifest_rejects_eight_wrong_names(tmp_path: Path) -> None:
    revision_case(tmp_path)
    source = tmp_path / "train-validation-source"
    records = tuple(file_record(path) for path in sorted(source.glob("*.csv"))) * 2
    with pytest.raises(ValueError, match="eight canonical relation files"):
        crsp_extract_from_records(
            "2026-10-29T00:00:00+00:00", file_record(source / "protocol.json"), records
        )
