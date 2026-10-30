"""Real key files, restrictive permissions and partition-scoped key acquisition."""

import os
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue

from covenant_ml.markets.research_keys import (
    EvaluatorKeys,
    create_evaluator_keys,
    decode_research_access,
    encode_research_access,
)


def test_distinct_identities_and_public_only_config(tmp_path: Path) -> None:
    root = tmp_path / "private"
    access = create_evaluator_keys(root, 1024)
    assert decode_research_access(encode_research_access(access)) == access
    assert root.stat().st_mode & 0o777 == 0o700
    keys = EvaluatorKeys(access)
    for partition, expected in (
        ("train-validation", access["validation_recipient"]),
        ("final", access["final_recipient"]),
    ):
        path = root / (partition + ".agekey")
        assert path.stat().st_mode & 0o777 == 0o600
        identity = keys.acquire(partition)
        assert str(identity.to_public()) == expected
        assert str(identity) not in encode_research_access(access).values()
    assert access["validation_recipient"] != access["final_recipient"]
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError):
        create_evaluator_keys(root, 1024)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra", True),
        ("directory", "relative"),
        ("directory", "/x/../y"),
        ("max_plaintext_bytes", 0),
    ],
)
def test_invalid_settings_never_read_keys(tmp_path: Path, field: str, value: JSONValue) -> None:
    document = encode_research_access(create_evaluator_keys(tmp_path / "keys", 100))
    document[field] = value
    with pytest.raises(ValueError, match="RESEARCH_KEYS_CONFIG"):
        decode_research_access(document)


def test_same_identity_cannot_authorize_both_partitions(tmp_path: Path) -> None:
    document = encode_research_access(create_evaluator_keys(tmp_path / "keys", 100))
    document["final_recipient"] = document["validation_recipient"]
    with pytest.raises(ValueError, match="distinct recipients"):
        decode_research_access(document)


@pytest.mark.parametrize(
    "failure",
    [
        "partition",
        "directory_mode",
        "directory_symlink",
        "file_mode",
        "file_symlink",
        "fifo",
        "prefix",
        "length",
        "wrong_key",
    ],
)
def test_real_key_access_refusals(tmp_path: Path, failure: str) -> None:
    root = tmp_path / "keys"
    access = create_evaluator_keys(root, 100)
    path = root / "final.agekey"
    if failure == "directory_mode":
        root.chmod(0o755)
    elif failure == "directory_symlink":
        target = tmp_path / "moved"
        root.rename(target)
        root.symlink_to(target, target_is_directory=True)
    elif failure == "file_mode":
        path.chmod(0o644)
    elif failure == "file_symlink":
        path.unlink()
        path.symlink_to(root / "train-validation.agekey")
    elif failure == "fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    elif failure == "prefix":
        path.write_text("not-a-key")
    elif failure == "length":
        path.write_text("AGE-SECRET-KEY-1short")
    elif failure == "wrong_key":
        path.write_bytes((root / "train-validation.agekey").read_bytes())
    with pytest.raises((ValueError, OSError)):
        EvaluatorKeys(access).acquire("unknown" if failure == "partition" else "final")


def test_validation_worker_does_not_need_a_final_key(tmp_path: Path) -> None:
    access = create_evaluator_keys(tmp_path / "keys", 100)
    (tmp_path / "keys/final.agekey").unlink()
    keys = EvaluatorKeys(access)
    assert str(keys.acquire("train-validation").to_public()) == access["validation_recipient"]
    with pytest.raises(FileNotFoundError):
        keys.acquire("final")
