"""Qualify the encrypted train-validation partition once and publish the evidence.

The request pins the design, the encrypted bundle, the entrypoint's source
digest and the image digest before any key is acquired, the same lineage the
research worker checks before fitting. Only the train-validation identity is
acquired; the final partition is never opened. Plaintext lives only inside
the shared decryption context and is removed when it closes. The published
manifest carries counts, refused-path metadata and identities, never an
observation, and the shared run record carries the counts.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint, encode_run_fingerprint
from platform_core.json_utils import JSONObject, dump_json_str, require_dict, require_int
from platform_core.run_record import Observation
from typing_extensions import ReadOnly, TypedDict

from ..benchmarking.harness import write_observed_manifest
from .campaign_records import require_digest
from .crsp_extract import encode_crsp_extract, load_crsp_sources
from .crsp_protocol import CrspProtocol, decode_crsp_protocol, encode_crsp_protocol
from .crsp_qualification import (
    crsp_qualification_counts,
    encode_crsp_qualification,
    qualify_crsp_panel,
)
from .encrypted_records import encode_encrypted_bundle
from .encrypted_storage import decrypted_bundle, read_encrypted_bundle
from .research_keys import EvaluatorKeys
from .validation import require_nonempty

#: The counts the shared run record carries, by observation name.
OBSERVED = (
    "paths",
    "refused",
    "unfilled_training",
    "unfilled_validation",
    "dependent_decisions",
    "dependent_rows",
)


class CrspQualificationRequest(TypedDict):
    """One frozen qualification of one encrypted train-validation bundle.

    Attributes:
        design: The protocol the staged bundle must carry, byte for byte.
        manifest: Path of the encrypted bundle metadata.
        dataset_digest: SHA-256 of that metadata.
        code_digest: SHA-256 of the entrypoint source that must run.
        image_digest: Digest of the runtime image that must run it.
        label: The run record's label.
    """

    design: ReadOnly[CrspProtocol]
    manifest: ReadOnly[str]
    dataset_digest: ReadOnly[str]
    code_digest: ReadOnly[str]
    image_digest: ReadOnly[str]
    label: ReadOnly[str]


def encode_crsp_qualification_request(value: CrspQualificationRequest) -> JSONObject:
    """Encode a frozen qualification request.

    Args:
        value: Validated request.

    Returns:
        Canonical document.
    """
    return {
        "design": encode_crsp_protocol(value["design"]),
        "manifest": value["manifest"],
        "dataset_digest": value["dataset_digest"],
        "code_digest": value["code_digest"],
        "image_digest": value["image_digest"],
        "label": value["label"],
    }


def decode_crsp_qualification_request(document: JSONObject) -> CrspQualificationRequest:
    """Require a complete frozen request.

    Args:
        document: Untrusted request document.

    Returns:
        The validated request.

    Raises:
        ValueError: CRSP_QUALIFICATION_REQUEST for unknown or missing fields;
            CRSP-001 and digest validation propagate from the shared decoders.
    """
    if set(document) != set(CrspQualificationRequest.__annotations__):
        raise ValueError("CRSP_QUALIFICATION_REQUEST: exact request fields required")
    return CrspQualificationRequest(
        design=decode_crsp_protocol(require_dict(document, "design")),
        manifest=require_nonempty(document, "manifest"),
        dataset_digest=require_digest(document, "dataset_digest"),
        code_digest=require_digest(document, "code_digest"),
        image_digest=require_digest(document, "image_digest"),
        label=require_nonempty(document, "label"),
    )


def execute_crsp_qualification(
    request: CrspQualificationRequest,
    fingerprint: RunFingerprint,
    code_digest: str,
    output: Path,
    *,
    keys: EvaluatorKeys,
    scratch: Path,
) -> Path:
    """Check lineage, qualify the decrypted partition and publish the evidence.

    Args:
        request: Frozen request.
        fingerprint: Runtime measured by the entrypoint.
        code_digest: Digest of the entrypoint source, measured by the entrypoint.
        output: Manifest path in a directory created for this run.
        keys: Evaluator key custody; only train-validation is acquired.
        scratch: Private job-local temporary storage.

    Returns:
        The shared run record sidecar.

    Raises:
        ValueError: CRSP_QUALIFICATION_LINEAGE for a different entrypoint,
            image or staged protocol; ENCRYPTED_MANIFEST for a different bundle;
            calendar faults from the panel propagate.
        FileExistsError: The output directory already exists.
        OSError: Encrypted data or result storage is unavailable.
    """
    if (
        code_digest != request["code_digest"]
        or fingerprint["image_digest"] != request["image_digest"]
    ):
        raise ValueError("CRSP_QUALIFICATION_LINEAGE: entrypoint or image differs from request")
    encrypted = Path(request["manifest"])
    digest = request["dataset_digest"]
    bundle = read_encrypted_bundle(encrypted, digest)
    design = encode_crsp_protocol(request["design"])
    with decrypted_bundle(
        encrypted,
        digest,
        partition="train-validation",
        identity=keys.acquire("train-validation"),
        scratch=scratch,
        max_plaintext_bytes=keys.access["max_plaintext_bytes"],
    ) as manifest:
        sources = load_crsp_sources(manifest)
        if encode_crsp_protocol(sources.protocol) != design:
            raise ValueError("CRSP_QUALIFICATION_LINEAGE: staged protocol differs from request")
        output.parent.mkdir(parents=True, exist_ok=False)
        qualification = qualify_crsp_panel(sources)
    counts = crsp_qualification_counts(qualification)
    encoded: JSONObject = {
        "kind": "crsp-qualification-v1",
        "design_sha256": sha256(dump_json_str(design).encode()).hexdigest(),
        "code_sha256": code_digest,
        "dataset": {"bundle_sha256": digest, "bundle": encode_encrypted_bundle(bundle)},
        "source": encode_crsp_extract(sources.manifest),
        "counts": counts,
        "fingerprint": encode_run_fingerprint(fingerprint),
        **encode_crsp_qualification(qualification),
    }
    return write_observed_manifest(
        encoded,
        fingerprint,
        output,
        f"crsp-qualification-v{request['design']['version']}",
        request["label"],
        tuple(
            Observation(name=f"qualification.{name}", value=float(require_int(counts, name)))
            for name in OBSERVED
        ),
    )


__all__ = [
    "CrspQualificationRequest",
    "decode_crsp_qualification_request",
    "encode_crsp_qualification_request",
    "execute_crsp_qualification",
]
