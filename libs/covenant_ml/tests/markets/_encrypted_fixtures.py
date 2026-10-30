"""Real frozen bundle fixtures and private evaluator identities, shared by trial tests."""

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import dump_json_str
from platform_core.stage_manifest import StagedFile

from covenant_ml.markets.encrypted_storage import encrypt_bundle
from covenant_ml.markets.research_keys import (
    EvaluatorKeys,
    ResearchAccess,
    create_evaluator_keys,
    decode_research_access,
    encode_research_access,
)
from covenant_ml.markets.validation import read_document


def file_record(path: Path) -> StagedFile:
    """Record the exact small synthetic fixture bytes."""
    payload = path.read_bytes()
    return StagedFile(name=path.name, sha256=sha256(payload).hexdigest(), size_bytes=len(payload))


def create_access(root: Path) -> ResearchAccess:
    """Write distinct real identities and the private evaluator configuration."""
    access = create_evaluator_keys(root / "keys", 10_000_000)
    (root / "access.json").write_text(dump_json_str(encode_research_access(access)))
    return access


def evaluator_keys(root: Path) -> EvaluatorKeys:
    """Open the explicitly created fixture configuration without reading a key yet."""
    return EvaluatorKeys(decode_research_access(read_document(root / "access.json")))


def seal_extract(manifest: Path, destination: Path, access: ResearchAccess, partition: str) -> Path:
    """Encrypt every file of an isolated synthetic extract using its partition's public key."""
    recipient = (
        access["validation_recipient"]
        if partition == "train-validation"
        else access["final_recipient"]
    )
    encrypt_bundle(
        manifest.parent,
        destination,
        files=tuple(file_record(path) for path in sorted(manifest.parent.iterdir())),
        manifest=file_record(manifest),
        recipient=recipient,
        partition=partition,
    )
    return destination / "bundle.json"


__all__ = ["create_access", "evaluator_keys", "file_record", "seal_extract"]
