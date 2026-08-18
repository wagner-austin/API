"""Tests for the growth-policy record shapes and their codecs."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONValue

from covenant_ml.growth_policy.types import (
    ERR_NOT_FLOAT,
    ERR_NOT_INT,
    ERR_NOT_LIST,
    ERR_NOT_MAPPING,
    ERR_NOT_STR,
    ERR_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    ArmSummary,
    decode_arm_result,
    decode_arm_summary,
    decode_dataset_info,
    decode_experiment_config,
    decode_growth_policy_report,
    encode_arm_result,
    encode_arm_summary,
    encode_dataset_info,
    encode_experiment_config,
    encode_growth_policy_report,
)

from .factories import make_arm_result, make_config, make_dataset_info, make_report


class TestArmResultCodec:
    """Round trip and validation for :class:`ArmResult`."""

    def test_round_trip_is_identity(self) -> None:
        """Decoding an encoded result should return the original record."""
        result = make_arm_result()

        assert decode_arm_result(encode_arm_result(result)) == result

    def test_rejects_non_string_arm(self) -> None:
        """A numeric arm name should be refused, naming the field."""
        data = encode_arm_result(make_arm_result())
        data["arm"] = 7

        with pytest.raises(ValueError, match=ERR_NOT_STR):
            decode_arm_result(data)

    def test_rejects_boolean_seed(self) -> None:
        """A boolean seed should be refused despite bool subclassing int."""
        data = encode_arm_result(make_arm_result())
        data["seed"] = True

        with pytest.raises(ValueError, match=ERR_NOT_INT):
            decode_arm_result(data)

    def test_rejects_non_numeric_metric(self) -> None:
        """A string metric should be refused."""
        data = encode_arm_result(make_arm_result())
        data["auc_roc"] = "high"

        with pytest.raises(ValueError, match=ERR_NOT_FLOAT):
            decode_arm_result(data)

    def test_rejects_boolean_metric(self) -> None:
        """A boolean metric should be refused despite bool subclassing int."""
        data = encode_arm_result(make_arm_result())
        data["auc_pr"] = False

        with pytest.raises(ValueError, match=ERR_NOT_FLOAT):
            decode_arm_result(data)

    def test_widens_integer_metric_to_float(self) -> None:
        """An integral metric should decode as a float, since JSON writes 1.0 as 1."""
        data = encode_arm_result(make_arm_result())
        data["log_loss"] = 1

        decoded = decode_arm_result(data)

        assert decoded["log_loss"] == 1.0


class TestArmSummaryCodec:
    """Round trip and validation for :class:`ArmSummary`."""

    def test_round_trip_is_identity(self) -> None:
        """Decoding an encoded summary should return the original record."""
        summary: ArmSummary = {
            "arm": "arm-a",
            "seed_count": 3,
            "fit_seconds": 1.5,
            "auc_roc": 0.7,
            "auc_pr": 0.3,
            "log_loss": 0.2,
            "mean_leaves": 12.5,
        }

        assert decode_arm_summary(encode_arm_summary(summary)) == summary

    def test_rejects_non_integer_seed_count(self) -> None:
        """A fractional seed count should be refused."""
        original: ArmSummary = {
            "arm": "arm-a",
            "seed_count": 3,
            "fit_seconds": 1.5,
            "auc_roc": 0.7,
            "auc_pr": 0.3,
            "log_loss": 0.2,
            "mean_leaves": 12.5,
        }
        data = encode_arm_summary(original)
        data["seed_count"] = 1.5

        with pytest.raises(ValueError, match=ERR_NOT_INT):
            decode_arm_summary(data)


class TestDatasetInfoCodec:
    """Round trip and validation for :class:`DatasetInfo`."""

    def test_round_trip_is_identity(self) -> None:
        """Decoding an encoded description should return the original record."""
        info = make_dataset_info()

        assert decode_dataset_info(encode_dataset_info(info)) == info


class TestExperimentConfigCodec:
    """Round trip and validation for :class:`ExperimentConfig`."""

    def test_round_trip_is_identity(self) -> None:
        """Decoding an encoded configuration should return the original record."""
        config = make_config()

        assert decode_experiment_config(encode_experiment_config(config)) == config

    def test_rejects_non_numeric_learning_rate(self) -> None:
        """A null learning rate should be refused."""
        data = encode_experiment_config(make_config())
        data["learning_rate"] = None

        with pytest.raises(ValueError, match=ERR_NOT_FLOAT):
            decode_experiment_config(data)


class TestGrowthPolicyReportCodec:
    """Round trip and validation for the complete report."""

    def test_round_trip_is_identity(self) -> None:
        """Decoding an encoded report should return the original record."""
        report = make_report([make_arm_result()], [make_arm_result()])

        assert decode_growth_policy_report(encode_growth_policy_report(report)) == report

    def test_rejects_a_different_schema_version(self) -> None:
        """A report from another schema version should be refused, not coerced."""
        data = encode_growth_policy_report(make_report([make_arm_result()], [make_arm_result()]))
        data["schema_version"] = REPORT_SCHEMA_VERSION + 1

        with pytest.raises(ValueError, match=ERR_SCHEMA_VERSION):
            decode_growth_policy_report(data)

    def test_rejects_non_array_results(self) -> None:
        """A results field that is not an array should be refused."""
        data = encode_growth_policy_report(make_report([make_arm_result()], [make_arm_result()]))
        data["results"] = {"arm": "arm-a"}

        with pytest.raises(ValueError, match=ERR_NOT_LIST):
            decode_growth_policy_report(data)

    def test_rejects_non_object_result_entry(self) -> None:
        """A results entry that is not an object should be refused."""
        data = encode_growth_policy_report(make_report([make_arm_result()], [make_arm_result()]))
        entries: list[JSONValue] = ["arm-a"]
        data["results"] = entries

        with pytest.raises(ValueError, match=ERR_NOT_MAPPING):
            decode_growth_policy_report(data)

    def test_rejects_non_object_summary_entry(self) -> None:
        """A summaries entry that is not an object should be refused."""
        data = encode_growth_policy_report(make_report([make_arm_result()], [make_arm_result()]))
        entries: list[JSONValue] = [3]
        data["summaries"] = entries

        with pytest.raises(ValueError, match=ERR_NOT_MAPPING):
            decode_growth_policy_report(data)

    def test_rejects_non_object_dataset(self) -> None:
        """A dataset field that is not an object should be refused."""
        data = encode_growth_policy_report(make_report([make_arm_result()], [make_arm_result()]))
        data["dataset"] = "synthetic"

        with pytest.raises(ValueError, match=ERR_NOT_MAPPING):
            decode_growth_policy_report(data)

    def test_rejects_non_integer_seed(self) -> None:
        """A non-integer seed should be refused."""
        data = encode_growth_policy_report(make_report([make_arm_result()], [make_arm_result()]))
        seeds: list[JSONValue] = ["42"]
        data["seeds"] = seeds

        with pytest.raises(ValueError, match=ERR_NOT_INT):
            decode_growth_policy_report(data)
