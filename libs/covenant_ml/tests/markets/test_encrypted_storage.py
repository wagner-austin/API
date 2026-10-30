"""Real age encryption, real CRSP files and failure cleanup, without fake crypto."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.stage_manifest import decode_stage_manifest, encode_stage_manifest

from covenant_ml.markets import _test_hooks_encryption as hooks
from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.encrypted_records import (
    EncryptedBundle,
    decode_encrypted_bundle,
    encode_encrypted_bundle,
)
from covenant_ml.markets.encrypted_storage import (
    decrypted_bundle,
    encrypt_bundle,
    encrypted_stage_manifest,
)

from ._crsp_fixtures import panel_protocol, source_rows, write_sources
from ._encrypted_fixtures import file_record


def _bundle(tmp_path: Path) -> tuple[Path, EncryptedBundle, hooks.AgeIdentity]:
    source = tmp_path / "source"
    source.mkdir()
    manifest, _ = write_sources(source, source_rows(), panel_protocol())
    identity = hooks.x25519.Identity.generate()
    bundle = encrypt_bundle(
        source,
        tmp_path / "encrypted",
        files=tuple(file_record(path) for path in sorted(source.iterdir())),
        manifest=file_record(manifest),
        recipient=str(identity.to_public()),
        partition="train-validation",
    )
    return tmp_path / "encrypted" / "bundle.json", bundle, identity


def _publish(path: Path, bundle: EncryptedBundle) -> str:
    path.write_text(dump_json_str(encode_encrypted_bundle(bundle)), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def test_real_crsp_round_trip_stage_identity_and_cleanup(tmp_path: Path) -> None:
    path, bundle, identity = _bundle(tmp_path)
    digest = sha256(path.read_bytes()).hexdigest()
    stage = encrypted_stage_manifest(bundle, path, "/pub/study/validation")
    assert decode_stage_manifest(encode_stage_manifest(stage)) == stage
    assert stage["provenance"] == {
        "encrypted_bundle": digest,
        "plaintext_manifest": bundle["manifest"]["sha256"],
        "partition": "train-validation",
    }
    assert {entry["name"] for entry in stage["files"]} == {
        "bundle.json",
        *(entry["ciphertext"]["name"] for entry in bundle["files"]),
    }
    assert str(identity).encode() not in path.read_bytes()
    assert decode_encrypted_bundle(load_json_str(path.read_text())) == bundle
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with decrypted_bundle(
        path,
        digest,
        partition="train-validation",
        identity=identity,
        scratch=scratch,
        max_plaintext_bytes=10_000_000,
    ) as plaintext:
        assert plaintext.parent.stat().st_mode & 0o777 == 0o700
        assert load_crsp_sources(plaintext) == load_crsp_sources(tmp_path / "source/manifest.json")
        for item in bundle["files"]:
            assert file_record(plaintext.parent / item["plaintext"]["name"]) == item["plaintext"]
    assert list(scratch.iterdir()) == []
    assert len(list(path.parent.iterdir())) == 11


@pytest.mark.parametrize("failure", ["pin", "partition", "identity", "limit"])
def test_access_and_budget_refusals_create_no_plaintext(tmp_path: Path, failure: str) -> None:
    path, _, identity = _bundle(tmp_path)
    digest = "0" * 64 if failure == "pin" else sha256(path.read_bytes()).hexdigest()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with (
        pytest.raises(ValueError, match=r"ENCRYPTED_MANIFEST|ENCRYPTED_ACCESS|ENCRYPTED_LIMIT"),
        decrypted_bundle(
            path,
            digest,
            partition="final" if failure == "partition" else "train-validation",
            identity=hooks.x25519.Identity.generate() if failure == "identity" else identity,
            scratch=scratch,
            max_plaintext_bytes=1 if failure == "limit" else 10_000_000,
        ),
    ):
        pytest.fail("A refused bundle yielded plaintext")
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("failure", ["cipher_digest", "plain_digest", "symlink", "consumer"])
def test_failure_after_decryption_starts_removes_private_files(
    tmp_path: Path, failure: str
) -> None:
    path, bundle, identity = _bundle(tmp_path)
    selected = bundle["files"][-1]
    if failure == "cipher_digest":
        (path.parent / selected["ciphertext"]["name"]).write_bytes(b"wrong bytes")
    elif failure == "plain_digest":
        selected["plaintext"]["sha256"] = "0" * 64
    elif failure == "symlink":
        cipher = path.parent / selected["ciphertext"]["name"]
        elsewhere = tmp_path / "elsewhere.age"
        cipher.rename(elsewhere)
        cipher.symlink_to(elsewhere)
    digest = _publish(path, bundle)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with (
        pytest.raises(ValueError, match=r"ENCRYPTED_DIGEST|WRDS_DATA_DRIFT|consumer failed"),
        decrypted_bundle(
            path,
            digest,
            partition="train-validation",
            identity=identity,
            scratch=scratch,
            max_plaintext_bytes=10_000_000,
        ),
    ):
        if failure == "consumer":
            raise ValueError("consumer failed")
        pytest.fail("Invalid bundle yielded plaintext")
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("mutation", ["flip", "truncate", "wrong_key", "oversize"])
def test_real_age_rejects_corruption_even_when_cipher_checksum_matches(
    tmp_path: Path, mutation: str
) -> None:
    path, bundle, identity = _bundle(tmp_path)
    item = bundle["files"][-1]
    cipher = path.parent / item["ciphertext"]["name"]
    raw = cipher.read_bytes()
    if mutation == "flip":
        cipher.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif mutation == "truncate":
        cipher.write_bytes(raw[:-10])
    elif mutation == "oversize":
        item["plaintext"]["size_bytes"] = 1
    else:
        identity = hooks.x25519.Identity.generate()
        document = encode_encrypted_bundle(bundle)
        document["recipient"] = str(identity.to_public())
        bundle = decode_encrypted_bundle(document)
    # A trusted manifest pin is essential; this deliberately changes it to
    # demonstrate that age authentication is independently enforced as well.
    if mutation in {"flip", "truncate"}:
        item["ciphertext"]["sha256"] = sha256(cipher.read_bytes()).hexdigest()
        item["ciphertext"]["size_bytes"] = cipher.stat().st_size
    digest = _publish(path, bundle)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    decrypt_error: type[Exception] = __import__("pyrage").DecryptError
    with (
        pytest.raises((decrypt_error, OSError, ValueError)),
        decrypted_bundle(
            path,
            digest,
            partition="train-validation",
            identity=identity,
            scratch=scratch,
            max_plaintext_bytes=10_000_000,
        ),
    ):
        pytest.fail("Unauthenticated plaintext was exposed")
    assert list(scratch.iterdir()) == []


def test_multichunk_file_and_distinct_ciphertext_on_reencryption(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "manifest.json"
    path.write_bytes(b"synthetic data\n" * 20_000)
    identity = hooks.x25519.Identity.generate()
    bundles = [
        encrypt_bundle(
            source,
            tmp_path / name,
            files=(file_record(path),),
            manifest=file_record(path),
            recipient=str(identity.to_public()),
            partition="final",
        )
        for name in ("one", "two")
    ]
    assert (
        bundles[0]["files"][0]["ciphertext"]["sha256"]
        != bundles[1]["files"][0]["ciphertext"]["sha256"]
    )
    cipher_manifest = tmp_path / "one/bundle.json"
    with decrypted_bundle(
        cipher_manifest,
        sha256(cipher_manifest.read_bytes()).hexdigest(),
        partition="final",
        identity=identity,
        scratch=tmp_path,
        max_plaintext_bytes=path.stat().st_size,
    ) as restored:
        assert restored.read_bytes() == path.read_bytes()


def test_changed_source_and_changed_published_metadata_are_refused(tmp_path: Path) -> None:
    path, bundle, identity = _bundle(tmp_path)
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="ENCRYPTED_MANIFEST"):
        encrypted_stage_manifest(bundle, path, "/pub/study")
    plain = tmp_path / "source/manifest.json"
    expected = file_record(plain)
    plain.write_bytes(b"changed")
    with pytest.raises(ValueError, match="ENCRYPTED_DIGEST"):
        encrypt_bundle(
            plain.parent,
            tmp_path / "failed",
            files=(expected,),
            manifest=expected,
            recipient=str(identity.to_public()),
            partition="final",
        )
    assert list((tmp_path / "failed").iterdir()) == []
