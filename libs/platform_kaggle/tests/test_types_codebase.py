"""Tests for types: CodebaseCapability."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_kaggle.types import (
    CodebaseCapability,
    CodebaseProfile,
    decode_capability,
    decode_profile,
    encode_capability,
    encode_profile,
)
from tests._types_equality import _capabilities_equal, _profiles_equal


class TestCodebaseCapability:
    """Tests for CodebaseCapability type and encode/decode."""

    def test_capability_creation(self) -> None:
        """Test creating a CodebaseCapability instance."""
        cap = CodebaseCapability(
            name="xgboost_tabular",
            strength="strong",
            tags=("tabular", "classification"),
            description="XGBoost for tabular data",
        )
        assert cap.name == "xgboost_tabular"
        assert cap.strength == "strong"
        assert cap.tags == ("tabular", "classification")
        assert cap.description == "XGBoost for tabular data"

    def test_capability_equality(self) -> None:
        """Test CodebaseCapability equality comparison."""
        cap1 = CodebaseCapability(
            name="test",
            strength="moderate",
            tags=("test",),
            description="Test",
        )
        cap2 = CodebaseCapability(
            name="test",
            strength="moderate",
            tags=("test",),
            description="Test",
        )
        cap3 = CodebaseCapability(
            name="other",
            strength="moderate",
            tags=("test",),
            description="Test",
        )
        assert _capabilities_equal(cap1, cap2)
        assert not _capabilities_equal(cap1, cap3)

    def test_encode_decode_capability_roundtrip(self) -> None:
        """Test CodebaseCapability encode/decode roundtrip."""
        original = CodebaseCapability(
            name="xgboost_tabular",
            strength="strong",
            tags=("tabular", "classification", "regression"),
            description="XGBoost gradient boosting",
        )
        encoded = encode_capability(original)
        decoded = decode_capability(encoded)
        assert _capabilities_equal(decoded, original)

    def test_decode_capability_all_strengths(self) -> None:
        """Test decode_capability handles all valid strengths."""
        strengths = ["strong", "moderate", "basic"]
        for strength in strengths:
            data: JSONObject = {
                "name": "test",
                "strength": strength,
                "tags": ["test"],
                "description": "Test",
            }
            decoded = decode_capability(data)
            assert decoded.strength == strength

    def test_decode_capability_invalid_strength(self) -> None:
        """Test decode_capability raises on invalid strength."""
        data: JSONObject = {
            "name": "test",
            "strength": "super",
            "tags": ["test"],
            "description": "Test",
        }
        with pytest.raises(JSONTypeError, match="must be strong/moderate/basic"):
            decode_capability(data)


class TestCodebaseProfile:
    """Tests for CodebaseProfile type and encode/decode."""

    def test_profile_creation(self) -> None:
        """Test creating a CodebaseProfile instance."""
        cap = CodebaseCapability(
            name="test",
            strength="moderate",
            tags=("test",),
            description="Test",
        )
        profile = CodebaseProfile(
            capabilities=(cap,),
            ml_backends=("xgboost", "lightgbm"),
            data_formats=("csv", "parquet"),
            task_types=("binary_classification",),
        )
        assert len(profile.capabilities) == 1
        assert profile.ml_backends == ("xgboost", "lightgbm")
        assert profile.data_formats == ("csv", "parquet")
        assert profile.task_types == ("binary_classification",)

    def test_profile_equality(self) -> None:
        """Test CodebaseProfile equality comparison."""
        profile1 = CodebaseProfile(
            capabilities=(),
            ml_backends=("xgboost",),
            data_formats=("csv",),
            task_types=("classification",),
        )
        profile2 = CodebaseProfile(
            capabilities=(),
            ml_backends=("xgboost",),
            data_formats=("csv",),
            task_types=("classification",),
        )
        profile3 = CodebaseProfile(
            capabilities=(),
            ml_backends=("lightgbm",),
            data_formats=("csv",),
            task_types=("classification",),
        )
        assert _profiles_equal(profile1, profile2)
        assert not _profiles_equal(profile1, profile3)

    def test_encode_decode_profile_roundtrip(self) -> None:
        """Test CodebaseProfile encode/decode roundtrip."""
        cap = CodebaseCapability(
            name="xgboost_tabular",
            strength="strong",
            tags=("tabular",),
            description="XGBoost",
        )
        original = CodebaseProfile(
            capabilities=(cap,),
            ml_backends=("xgboost", "pytorch"),
            data_formats=("csv", "parquet"),
            task_types=("binary_classification", "regression"),
        )
        encoded = encode_profile(original)
        decoded = decode_profile(encoded)
        assert _profiles_equal(decoded, original)

    def test_decode_profile_invalid_capability(self) -> None:
        """Test decode_profile raises on invalid capability."""
        data: JSONObject = {
            "capabilities": ["not a dict"],
            "ml_backends": ["xgboost"],
            "data_formats": ["csv"],
            "task_types": ["classification"],
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_profile(data)
