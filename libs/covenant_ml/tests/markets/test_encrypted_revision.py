"""Metadata replacement uses real age encryption while retaining data ciphertext."""

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.stage_manifest import StagedFile

from covenant_ml.markets import _test_hooks_encryption as hooks
from covenant_ml.markets.encrypted_storage import (
    decrypted_bundle,
    encrypt_bundle,
    revise_encrypted_bundle,
    verify_encrypted_bundle,
)

from ._encrypted_fixtures import file_record


def prepared(root: Path) -> tuple[Path, Path, hooks.AgeIdentity]:
    """Seal synthetic final data and prepare changed public metadata."""
    source = root / "original"
    source.mkdir()
    (source / "manifest.json").write_bytes(b'{"protocol":"old"}')
    (source / "protocol.json").write_bytes(b'{"rule":"old"}')
    (source / "observations.csv").write_bytes(b"synthetic secret observations\n" * 10000)
    identity = hooks.x25519.Identity.generate()
    encrypt_bundle(
        source,
        root / "sealed",
        files=tuple(file_record(path) for path in sorted(source.iterdir())),
        manifest=file_record(source / "manifest.json"),
        recipient=str(identity.to_public()),
        partition="final",
    )
    replacements = root / "metadata"
    replacements.mkdir()
    (replacements / "manifest.json").write_bytes(b'{"protocol":"new"}')
    (replacements / "protocol.json").write_bytes(b'{"rule":"new"}')
    return root / "sealed/bundle.json", replacements, identity


def test_revision_retains_ciphertext_and_decrypts_as_declared(tmp_path: Path) -> None:
    path, replacements, identity = prepared(tmp_path)
    original = {item.name: item.read_bytes() for item in path.parent.iterdir()}
    files = tuple(file_record(item) for item in sorted(replacements.iterdir()))
    bundle = revise_encrypted_bundle(
        path,
        sha256(path.read_bytes()).hexdigest(),
        replacements,
        tmp_path / "revised",
        files=files,
        manifest=file_record(replacements / "manifest.json"),
    )
    revised = tmp_path / "revised/bundle.json"
    digest = sha256(revised.read_bytes()).hexdigest()
    assert verify_encrypted_bundle(revised, digest) == bundle
    assert bundle["partition"] == "final"
    assert bundle["recipient"] == str(identity.to_public())
    assert (revised.parent / "observations.csv.age").read_bytes() == original[
        "observations.csv.age"
    ]
    assert revised.read_bytes() != original["bundle.json"]
    assert {item.name: item.read_bytes() for item in path.parent.iterdir()} == original
    assert {item.name for item in revised.parent.iterdir()} == set(original)
    assert not list(tmp_path.glob("encrypted-revision-*"))
    with decrypted_bundle(
        revised,
        digest,
        partition="final",
        identity=identity,
        scratch=tmp_path,
        max_plaintext_bytes=1_000_000,
    ) as opened:
        assert opened.read_bytes() == (replacements / "manifest.json").read_bytes()
        assert (opened.parent / "protocol.json").read_bytes() == (
            replacements / "protocol.json"
        ).read_bytes()
        assert (opened.parent / "observations.csv").read_bytes() == (
            tmp_path / "original/observations.csv"
        ).read_bytes()
    assert not list(tmp_path.glob("research-plaintext-*"))


@pytest.mark.parametrize(
    "fault", ["duplicate", "unknown", "missing-manifest", "renamed", "unchanged"]
)
def test_invalid_replacement_inventory_never_creates_destination(
    tmp_path: Path, fault: str
) -> None:
    path, replacements, _ = prepared(tmp_path)
    manifest = file_record(replacements / "manifest.json")
    protocol = file_record(replacements / "protocol.json")
    files: tuple[StagedFile, ...] = (manifest, protocol)
    if fault == "duplicate":
        files = (manifest, protocol, protocol)
    elif fault == "unknown":
        (replacements / "unknown").write_bytes(b"unknown")
        files = (manifest, file_record(replacements / "unknown"))
    elif fault == "missing-manifest":
        files = (protocol,)
    elif fault == "renamed":
        manifest = protocol
    else:
        manifest = file_record(tmp_path / "original/manifest.json")
        files = (manifest, protocol)
    with pytest.raises(ValueError, match="ENCRYPTED_REVISION"):
        revise_encrypted_bundle(
            path,
            sha256(path.read_bytes()).hexdigest(),
            replacements,
            tmp_path / "refused",
            files=files,
            manifest=manifest,
        )
    assert not (tmp_path / "refused").exists()


def test_failed_replacement_leaves_no_published_manifest_or_temporary_files(tmp_path: Path) -> None:
    path, replacements, _ = prepared(tmp_path)
    manifest = file_record(replacements / "manifest.json")
    files = (manifest, file_record(replacements / "protocol.json"))
    (replacements / "protocol.json").write_bytes(b"changed after pinning")
    original = {item.name: item.read_bytes() for item in path.parent.iterdir()}
    with pytest.raises(ValueError, match="ENCRYPTED_DIGEST"):
        revise_encrypted_bundle(
            path,
            sha256(path.read_bytes()).hexdigest(),
            replacements,
            tmp_path / "failed",
            files=files,
            manifest=manifest,
        )
    assert not (tmp_path / "failed/bundle.json").exists()
    assert not list(tmp_path.glob("encrypted-revision-*"))
    assert {item.name: item.read_bytes() for item in path.parent.iterdir()} == original


@pytest.mark.parametrize("fault", ["occupied", "old-pin", "old-bytes"])
def test_original_identity_and_exclusive_destination_are_required(
    tmp_path: Path, fault: str
) -> None:
    path, replacements, _ = prepared(tmp_path)
    digest = sha256(path.read_bytes()).hexdigest()
    destination = tmp_path / "refused"
    if fault == "occupied":
        destination.mkdir()
        (destination / "keep").write_bytes(b"keep")
    elif fault == "old-pin":
        digest = "0" * 64
    else:
        (path.parent / "observations.csv.age").write_bytes(b"altered")
    with pytest.raises((FileExistsError, ValueError)):
        revise_encrypted_bundle(
            path,
            digest,
            replacements,
            destination,
            files=tuple(file_record(item) for item in sorted(replacements.iterdir())),
            manifest=file_record(replacements / "manifest.json"),
        )
    assert not (destination / "bundle.json").exists()
    if fault == "occupied":
        assert (destination / "keep").read_bytes() == b"keep"
