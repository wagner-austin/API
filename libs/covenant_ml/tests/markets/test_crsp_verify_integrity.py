"""Real ciphertext inventories and evaluator limits are checked before plaintext exists."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from covenant_ml.markets.crsp_verify import verify_crsp_extraction

from ._crsp_extraction_fixtures import (
    Extraction,
    FakeDatabase,
    database_rows,
    extraction,
    installed,
    limits,
    run_vendor,
)
from ._encrypted_fixtures import evaluator_keys


@pytest.fixture()
def encrypted_case(tmp_path: Path) -> Extraction:
    """Use the existing database twin and real age backend to seal both partitions."""
    case = extraction(tmp_path)
    with installed(FakeDatabase(database_rows())):
        run_vendor(case, limits())
    return case


@pytest.mark.parametrize("ciphertext_only", [True, False])
@pytest.mark.parametrize("partition", ["train-validation", "final"])
@pytest.mark.parametrize("damage", ["missing", "extra", "changed", "size", "symlink", "dangling"])
def test_ciphertext_damage_in_either_partition_is_refused(
    encrypted_case: Extraction, partition: str, damage: str, ciphertext_only: bool
) -> None:
    case = encrypted_case
    path = case.output / partition / "returns.csv.age"
    original = path.read_bytes()
    if damage == "missing":
        path.unlink()
    elif damage == "extra":
        (path.parent / "extra.age").write_bytes(original)
    elif damage == "changed":
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    elif damage == "size":
        path.write_bytes(original + b"extra")
    else:
        target = case.root / "cipher-copy.age"
        if damage == "symlink":
            target.write_bytes(original)
        path.unlink()
        path.symlink_to(target)
    expected = (
        "ENCRYPTED_DIGEST"
        if damage in {"changed", "size"}
        else "ENCRYPTED_INVENTORY"
        if damage in {"missing", "extra"}
        else r"\[CRSP-013\] symlink"
    )
    scratch = case.root / "private-scratch"
    scratch.mkdir(mode=0o700)
    with pytest.raises(ValueError, match=expected):
        verify_crsp_extraction(
            case.root,
            evaluator_keys(case.root),
            scratch=scratch,
            open_final=False,
            ciphertext_only=ciphertext_only,
        )
    assert list(scratch.iterdir()) == []


def test_final_ciphertext_is_checked_before_validation_decryption(
    encrypted_case: Extraction,
) -> None:
    case = encrypted_case
    path = case.output / "final" / "returns.csv.age"
    path.write_bytes(path.read_bytes() + b"changed")
    # A decrypt attempt would fail with FileNotFoundError at this absent scratch.
    scratch = case.root / "never-created-scratch"
    with pytest.raises(ValueError, match="ENCRYPTED_DIGEST"):
        verify_crsp_extraction(
            case.root,
            evaluator_keys(case.root),
            scratch=scratch,
            open_final=False,
            ciphertext_only=False,
        )
    assert not scratch.exists()


@pytest.mark.parametrize("damage", ["directory", "fifo", "partition-link", "metadata-link"])
def test_nonregular_and_nested_inventory_is_refused(
    encrypted_case: Extraction, damage: str
) -> None:
    case = encrypted_case
    if damage == "directory":
        (case.output / "final" / "empty").mkdir()
    elif damage == "fifo":
        path = case.output / "final" / "returns.csv.age"
        path.unlink()
        os.mkfifo(path)
    elif damage == "partition-link":
        partition = case.output / "final"
        target = case.root / "final-ciphertext"
        partition.rename(target)
        partition.symlink_to(target, target_is_directory=True)
    else:
        metadata = case.output / "extraction.json"
        target = case.root / "receipt-copy.json"
        metadata.rename(target)
        metadata.symlink_to(target)
    with pytest.raises(ValueError, match=r"\[CRSP-013\]"):
        verify_crsp_extraction(
            case.root,
            evaluator_keys(case.root),
            scratch=case.root / "absent",
            open_final=False,
            ciphertext_only=False,
        )
    assert not (case.root / "absent").exists()


@pytest.mark.parametrize("damage", ["symlink", "missing", "file", "metadata", "partition-file"])
def test_collection_root_and_required_metadata_are_checked(
    encrypted_case: Extraction, damage: str
) -> None:
    case = encrypted_case
    if damage == "metadata":
        (case.output / "metrics.json").unlink()
    elif damage == "partition-file":
        (case.output / "final").rename(case.root / "moved-final")
        (case.output / "final").write_bytes(b"not a directory")
    else:
        target = case.root / "moved-outputs"
        case.output.rename(target)
        if damage == "symlink":
            case.output.symlink_to(target, target_is_directory=True)
        elif damage == "file":
            case.output.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match=r"\[CRSP-013\]"):
        verify_crsp_extraction(
            case.root,
            evaluator_keys(case.root),
            scratch=case.root / "absent",
            open_final=False,
            ciphertext_only=False,
        )
    assert not (case.root / "absent").exists()


@pytest.mark.parametrize("scratch_exists", [True, False])
def test_evaluator_owned_byte_budget_prevents_any_plaintext(
    encrypted_case: Extraction, scratch_exists: bool
) -> None:
    case = encrypted_case
    access_path = case.root / "access.json"
    access = narrow_json_to_dict(load_json_str(access_path.read_text()))
    access["max_plaintext_bytes"] = 1
    access_path.write_text(dump_json_str(access))
    scratch = case.root / "never-created-scratch"
    if scratch_exists:
        scratch.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="ENCRYPTED_LIMIT"):
        verify_crsp_extraction(
            case.root,
            evaluator_keys(case.root),
            scratch=scratch,
            open_final=False,
            ciphertext_only=False,
        )
    assert scratch.exists() == scratch_exists
    if scratch_exists:
        assert list(scratch.iterdir()) == []


def test_complete_final_inventory_is_verified_without_its_key(encrypted_case: Extraction) -> None:
    case = encrypted_case
    (case.root / "keys" / "final.agekey").unlink()
    scratch = case.root / "private-scratch"
    scratch.mkdir(mode=0o700)
    result = verify_crsp_extraction(
        case.root,
        evaluator_keys(case.root),
        scratch=scratch,
        open_final=False,
        ciphertext_only=False,
    )
    assert result["ciphertext_files"] == 20
    assert list(scratch.iterdir()) == []
