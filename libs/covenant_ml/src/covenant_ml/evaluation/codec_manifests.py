"""Encoders and strict decoders for the two evaluation manifests.

A document of another schema version is refused before any field is read: an
old manifest decoded into new types would be wrong in ways no single field
check could see.
"""

from __future__ import annotations

from platform_core.comparability import decode_run_fingerprint, encode_run_fingerprint
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_str,
    require_dict,
    require_int,
    require_list,
    require_str,
)

from ..benchmarking.types_codec import (
    decode_benchmark_config,
    decode_dataset_info,
    encode_benchmark_config,
    encode_dataset_info,
)
from .codec_records import (
    decode_ablation_group,
    decode_alert_costs,
    decode_arm_result,
    decode_curve_point,
    decode_pooled,
    decode_resampling,
    decode_seed_comparisons,
    encode_ablation_group,
    encode_alert_costs,
    encode_arm_result,
    encode_curve_point,
    encode_pooled,
    encode_resampling,
    encode_seed_comparisons,
)
from .types import (
    ERR_SCHEMA_VERSION,
    EVALUATION_SCHEMA_VERSION,
    AblationManifest,
    ArmResult,
    EvaluationManifest,
)


def _require_schema(obj: JSONObject) -> int:
    """Refuse a document of another schema version.

    Args:
        obj: The parsed manifest.

    Returns:
        The schema version, which is :data:`EVALUATION_SCHEMA_VERSION`.

    Raises:
        JSONTypeError: If the field is missing or not an integer.
        ValueError: If it names another version.
    """
    version = require_int(obj, "schema_version")
    if version != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"[{ERR_SCHEMA_VERSION}] manifest schema {version} is not "
            f"{EVALUATION_SCHEMA_VERSION}; re-run the evaluation rather than reading it"
        )
    return version


def _decode_results(obj: JSONObject) -> list[ArmResult]:
    """Raise a manifest's per-arm results out of untrusted JSON.

    Args:
        obj: The parsed manifest.

    Returns:
        Every validated result, in document order.

    Raises:
        JSONTypeError: If the list or an entry is missing or mistyped.
        ValueError: If a quality record is malformed, from its own decoder.
    """
    return [decode_arm_result(narrow_json_to_dict(v)) for v in require_list(obj, "results")]


def encode_evaluation_manifest(manifest: EvaluationManifest) -> JSONObject:
    """Lower a paired-evaluation manifest to JSON.

    Args:
        manifest: The manifest.

    Returns:
        A JSON object mirroring :class:`EvaluationManifest`.
    """
    action_names: list[JSONValue] = list(manifest["action_names"])
    seeds: list[JSONValue] = list(manifest["seeds"])
    results: list[JSONValue] = [encode_arm_result(r) for r in manifest["results"]]
    seed_comparisons: list[JSONValue] = [
        encode_seed_comparisons(s) for s in manifest["seed_comparisons"]
    ]
    pooled: list[JSONValue] = [encode_pooled(p) for p in manifest["pooled"]]
    return {
        "schema_version": manifest["schema_version"],
        "dataset": encode_dataset_info(manifest["dataset"]),
        "config": encode_benchmark_config(manifest["config"]),
        "resampling": encode_resampling(manifest["resampling"]),
        "alert_costs": encode_alert_costs(manifest["alert_costs"]),
        "action_names": action_names,
        "reference": manifest["reference"],
        "seeds": seeds,
        "results": results,
        "seed_comparisons": seed_comparisons,
        "pooled": pooled,
        "fingerprint": encode_run_fingerprint(manifest["fingerprint"]),
    }


def decode_evaluation_manifest(value: JSONValue) -> EvaluationManifest:
    """Raise a paired-evaluation manifest out of untrusted JSON.

    Args:
        value: The parsed document.

    Returns:
        The validated manifest.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
        ValueError: If the schema version differs, or a nested benchmark
            record is malformed.
    """
    obj = narrow_json_to_dict(value)
    return {
        "schema_version": _require_schema(obj),
        "dataset": decode_dataset_info(require_dict(obj, "dataset"), "dataset"),
        "config": decode_benchmark_config(require_dict(obj, "config"), "config"),
        "resampling": decode_resampling(require_dict(obj, "resampling")),
        "alert_costs": decode_alert_costs(require_dict(obj, "alert_costs")),
        "action_names": [narrow_json_to_str(v) for v in require_list(obj, "action_names")],
        "reference": require_str(obj, "reference"),
        "seeds": [narrow_json_to_int(v) for v in require_list(obj, "seeds")],
        "results": _decode_results(obj),
        "seed_comparisons": [
            decode_seed_comparisons(narrow_json_to_dict(v))
            for v in require_list(obj, "seed_comparisons")
        ],
        "pooled": [decode_pooled(narrow_json_to_dict(v)) for v in require_list(obj, "pooled")],
        "fingerprint": decode_run_fingerprint(obj.get("fingerprint")),
    }


def encode_ablation_manifest(manifest: AblationManifest) -> JSONObject:
    """Lower an ablation manifest to JSON.

    Args:
        manifest: The manifest.

    Returns:
        A JSON object mirroring :class:`AblationManifest`.
    """
    groups: list[JSONValue] = [encode_ablation_group(g) for g in manifest["groups"]]
    seeds: list[JSONValue] = list(manifest["seeds"])
    results: list[JSONValue] = [encode_arm_result(r) for r in manifest["results"]]
    seed_comparisons: list[JSONValue] = [
        encode_seed_comparisons(s) for s in manifest["seed_comparisons"]
    ]
    pooled: list[JSONValue] = [encode_pooled(p) for p in manifest["pooled"]]
    curve: list[JSONValue] = [encode_curve_point(p) for p in manifest["learning_curve"]]
    return {
        "schema_version": manifest["schema_version"],
        "dataset": encode_dataset_info(manifest["dataset"]),
        "config": encode_benchmark_config(manifest["config"]),
        "model": manifest["model"],
        "resampling": encode_resampling(manifest["resampling"]),
        "groups": groups,
        "seeds": seeds,
        "results": results,
        "seed_comparisons": seed_comparisons,
        "pooled": pooled,
        "learning_curve": curve,
        "fingerprint": encode_run_fingerprint(manifest["fingerprint"]),
    }


def decode_ablation_manifest(value: JSONValue) -> AblationManifest:
    """Raise an ablation manifest out of untrusted JSON.

    Args:
        value: The parsed document.

    Returns:
        The validated manifest.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
        ValueError: If the schema version differs, or a nested benchmark
            record is malformed.
    """
    obj = narrow_json_to_dict(value)
    return {
        "schema_version": _require_schema(obj),
        "dataset": decode_dataset_info(require_dict(obj, "dataset"), "dataset"),
        "config": decode_benchmark_config(require_dict(obj, "config"), "config"),
        "model": require_str(obj, "model"),
        "resampling": decode_resampling(require_dict(obj, "resampling")),
        "groups": [
            decode_ablation_group(narrow_json_to_dict(v)) for v in require_list(obj, "groups")
        ],
        "seeds": [narrow_json_to_int(v) for v in require_list(obj, "seeds")],
        "results": _decode_results(obj),
        "seed_comparisons": [
            decode_seed_comparisons(narrow_json_to_dict(v))
            for v in require_list(obj, "seed_comparisons")
        ],
        "pooled": [decode_pooled(narrow_json_to_dict(v)) for v in require_list(obj, "pooled")],
        "learning_curve": [
            decode_curve_point(narrow_json_to_dict(v)) for v in require_list(obj, "learning_curve")
        ],
        "fingerprint": decode_run_fingerprint(obj.get("fingerprint")),
    }


__all__ = [
    "decode_ablation_manifest",
    "decode_evaluation_manifest",
    "encode_ablation_manifest",
    "encode_evaluation_manifest",
]
