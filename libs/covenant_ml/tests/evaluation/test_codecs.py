"""Tests for the evaluation manifests' codecs.

The manifests round-tripped here are produced by real runs, so every record
type a run emits passes through its encoder and decoder at least once.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)

from covenant_ml.benchmarking.adapters import LightGbmTrainer
from covenant_ml.benchmarking.factory import make_baseline_trainers
from covenant_ml.evaluation.ablation import run_feature_ablation
from covenant_ml.evaluation.codec_manifests import (
    decode_ablation_manifest,
    decode_evaluation_manifest,
    encode_ablation_manifest,
    encode_evaluation_manifest,
)
from covenant_ml.evaluation.codec_records import decode_comparison, decode_interval
from covenant_ml.evaluation.runner import run_paired_evaluation
from covenant_ml.evaluation.splits import CompanyDisjointGroupedSplits
from covenant_ml.evaluation.types import (
    ERR_SCHEMA_VERSION,
    AblationManifest,
    EvaluationManifest,
)

from ._fixtures import (
    COSTS,
    DATASET,
    FEATURE_NAMES,
    FINGERPRINT,
    grouped_dataset,
    settings,
    tiny_config,
)


def _evaluation() -> EvaluationManifest:
    features, labels, codes = grouped_dataset()
    return run_paired_evaluation(
        make_baseline_trainers(tiny_config()),
        CompanyDisjointGroupedSplits(features, labels, codes),
        [5, 6],
        "cleargbm",
        tiny_config(),
        DATASET,
        settings(),
        COSTS,
        FINGERPRINT,
    )


def _ablation() -> AblationManifest:
    features, labels, codes = grouped_dataset()
    return run_feature_ablation(
        LightGbmTrainer(tiny_config()),
        CompanyDisjointGroupedSplits(features, labels, codes),
        [5, 6],
        FEATURE_NAMES,
        [{"name": "noise", "features": ["noise_a", "noise_b"]}],
        [0.5, 1.0],
        tiny_config(),
        DATASET,
        settings(),
        FINGERPRINT,
    )


def _through_json(document: JSONObject) -> JSONObject:
    return narrow_json_to_dict(load_json_str(dump_json_str(document)))


def test_an_evaluation_manifest_survives_json() -> None:
    manifest = _evaluation()
    assert decode_evaluation_manifest(_through_json(encode_evaluation_manifest(manifest))) == (
        manifest
    )


def test_an_ablation_manifest_survives_json() -> None:
    manifest = _ablation()
    assert decode_ablation_manifest(_through_json(encode_ablation_manifest(manifest))) == manifest


def test_another_schema_version_is_refused_before_any_field() -> None:
    document = encode_evaluation_manifest(_evaluation())
    document["schema_version"] = 99
    with pytest.raises(ValueError, match=ERR_SCHEMA_VERSION):
        decode_evaluation_manifest(document)


def test_an_ablation_of_another_schema_version_is_refused() -> None:
    document = encode_ablation_manifest(_ablation())
    document["schema_version"] = 0
    with pytest.raises(ValueError, match=ERR_SCHEMA_VERSION):
        decode_ablation_manifest(document)


def test_a_missing_field_is_refused_by_name() -> None:
    document = encode_evaluation_manifest(_evaluation())
    del document["reference"]
    with pytest.raises(JSONTypeError, match="reference"):
        decode_evaluation_manifest(document)


def _first_comparison() -> JSONObject:
    document = encode_evaluation_manifest(_evaluation())
    first_seed = narrow_json_to_dict(require_list(document, "seed_comparisons")[0])
    return narrow_json_to_dict(require_list(first_seed, "comparisons")[0])


def test_an_unknown_metric_is_refused() -> None:
    comparison = _first_comparison()
    comparison["metric"] = "accuracy"
    with pytest.raises(JSONTypeError, match="metric"):
        decode_comparison(comparison)


def test_an_unknown_interval_method_is_refused() -> None:
    interval = require_dict(_first_comparison(), "difference")
    interval["method"] = "eyeballed"
    with pytest.raises(JSONTypeError, match="method"):
        decode_interval(interval)
