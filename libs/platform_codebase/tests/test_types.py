"""Tests for platform_codebase.types module."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_codebase.types import (
    CodebaseCapability,
    CodebaseProfile,
    LibInfo,
    ServiceInfo,
    decode_capability,
    decode_profile,
    encode_capability,
    encode_lib_info,
    encode_profile,
    encode_service_info,
    require_recommendation,
    require_strength,
)


class TestCodebaseCapability:
    """Tests for CodebaseCapability class."""

    def test_init(self) -> None:
        """Test CodebaseCapability initialization."""
        cap = CodebaseCapability(
            name="test_cap",
            strength="strong",
            tags=("tag1", "tag2"),
            description="Test description",
        )

        assert cap.name == "test_cap"
        assert cap.strength == "strong"
        assert cap.tags == ("tag1", "tag2")
        assert cap.description == "Test description"

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip for CodebaseCapability."""
        cap = CodebaseCapability(
            name="ml_cap",
            strength="moderate",
            tags=("ml", "tabular"),
            description="ML capability",
        )

        encoded = encode_capability(cap)
        decoded = decode_capability(encoded)

        assert decoded.name == cap.name
        assert decoded.strength == cap.strength
        assert decoded.tags == cap.tags
        assert decoded.description == cap.description


class TestCodebaseProfile:
    """Tests for CodebaseProfile class."""

    def test_init_minimal(self) -> None:
        """Test CodebaseProfile with minimal args."""
        profile = CodebaseProfile(capabilities=())

        assert profile.capabilities == ()
        assert profile.technologies == ()
        assert profile.frameworks == ()
        assert profile.ml_backends == ()
        assert profile.data_formats == ()
        assert profile.task_types == ()

    def test_init_full(self) -> None:
        """Test CodebaseProfile with all args."""
        cap = CodebaseCapability(
            name="cap1",
            strength="strong",
            tags=("tag1",),
            description="Cap 1",
        )
        profile = CodebaseProfile(
            capabilities=(cap,),
            technologies=("python",),
            frameworks=("fastapi",),
            ml_backends=("xgboost",),
            data_formats=("csv",),
            task_types=("classification",),
        )

        assert len(profile.capabilities) == 1
        assert profile.technologies == ("python",)
        assert profile.frameworks == ("fastapi",)
        assert profile.ml_backends == ("xgboost",)
        assert profile.data_formats == ("csv",)
        assert profile.task_types == ("classification",)

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip for CodebaseProfile."""
        cap = CodebaseCapability(
            name="cap1",
            strength="basic",
            tags=("t1", "t2"),
            description="Description",
        )
        profile = CodebaseProfile(
            capabilities=(cap,),
            technologies=("python", "js"),
            frameworks=("django",),
            ml_backends=("lightgbm",),
            data_formats=("parquet",),
            task_types=("regression",),
        )

        encoded = encode_profile(profile)
        decoded = decode_profile(encoded)

        assert len(decoded.capabilities) == 1
        assert decoded.capabilities[0].name == "cap1"
        assert decoded.technologies == ("python", "js")
        assert decoded.frameworks == ("django",)
        assert decoded.ml_backends == ("lightgbm",)
        assert decoded.data_formats == ("parquet",)
        assert decoded.task_types == ("regression",)


class TestLibInfo:
    """Tests for LibInfo class."""

    def test_init(self) -> None:
        """Test LibInfo initialization."""
        lib = LibInfo(
            name="my-lib",
            path=Path("libs/my-lib"),
            dependencies=("dep1", "dep2"),
        )

        assert lib.name == "my-lib"
        assert lib.path == Path("libs/my-lib")
        assert lib.dependencies == ("dep1", "dep2")

    def test_encode(self) -> None:
        """Test encode_lib_info encodes to JSON-serializable dict."""
        lib = LibInfo(
            name="my-lib",
            path=Path("libs/my-lib"),
            dependencies=("dep1", "dep2"),
        )

        encoded = encode_lib_info(lib)

        assert encoded["name"] == "my-lib"
        assert encoded["path"] == str(Path("libs/my-lib"))
        assert encoded["dependencies"] == ["dep1", "dep2"]


class TestServiceInfo:
    """Tests for ServiceInfo class."""

    def test_init(self) -> None:
        """Test ServiceInfo initialization."""
        service = ServiceInfo(
            name="my-service",
            path=Path("services/my-service"),
            dependencies=("dep1",),
            has_rules_files=True,
        )

        assert service.name == "my-service"
        assert service.path == Path("services/my-service")
        assert service.dependencies == ("dep1",)
        assert service.has_rules_files is True

    def test_encode(self) -> None:
        """Test encode_service_info encodes to JSON-serializable dict."""
        service = ServiceInfo(
            name="my-service",
            path=Path("services/my-service"),
            dependencies=("dep1", "dep2"),
            has_rules_files=True,
        )

        encoded = encode_service_info(service)

        assert encoded["name"] == "my-service"
        assert encoded["path"] == str(Path("services/my-service"))
        assert encoded["dependencies"] == ["dep1", "dep2"]
        assert encoded["has_rules_files"] is True


class TestRequireStrength:
    """Tests for require_strength validation helper."""

    def test_strong(self) -> None:
        """Test require_strength with 'strong' value."""
        obj: JSONObject = {"strength": "strong"}
        result = require_strength(obj, "strength")
        assert result == "strong"

    def test_moderate(self) -> None:
        """Test require_strength with 'moderate' value."""
        obj: JSONObject = {"strength": "moderate"}
        result = require_strength(obj, "strength")
        assert result == "moderate"

    def test_basic(self) -> None:
        """Test require_strength with 'basic' value."""
        obj: JSONObject = {"strength": "basic"}
        result = require_strength(obj, "strength")
        assert result == "basic"

    def test_invalid_value(self) -> None:
        """Test require_strength with invalid value."""
        obj: JSONObject = {"strength": "invalid"}
        with pytest.raises(JSONTypeError) as exc_info:
            require_strength(obj, "strength")
        assert "strong/moderate/basic" in str(exc_info.value)


class TestRequireRecommendation:
    """Tests for require_recommendation validation helper."""

    def test_strong_fit(self) -> None:
        """Test require_recommendation with 'strong_fit' value."""
        obj: JSONObject = {"rec": "strong_fit"}
        result = require_recommendation(obj, "rec")
        assert result == "strong_fit"

    def test_good_fit(self) -> None:
        """Test require_recommendation with 'good_fit' value."""
        obj: JSONObject = {"rec": "good_fit"}
        result = require_recommendation(obj, "rec")
        assert result == "good_fit"

    def test_stretch(self) -> None:
        """Test require_recommendation with 'stretch' value."""
        obj: JSONObject = {"rec": "stretch"}
        result = require_recommendation(obj, "rec")
        assert result == "stretch"

    def test_new_territory(self) -> None:
        """Test require_recommendation with 'new_territory' value."""
        obj: JSONObject = {"rec": "new_territory"}
        result = require_recommendation(obj, "rec")
        assert result == "new_territory"

    def test_invalid_value(self) -> None:
        """Test require_recommendation with invalid value."""
        obj: JSONObject = {"rec": "invalid"}
        with pytest.raises(JSONTypeError) as exc_info:
            require_recommendation(obj, "rec")
        assert "valid recommendation" in str(exc_info.value)


class TestDecodeCapabilityErrors:
    """Tests for decode_capability error handling."""

    def test_missing_name(self) -> None:
        """Test decode_capability with missing name field."""
        data: JSONObject = {
            "strength": "strong",
            "tags": ["t1"],
            "description": "desc",
        }
        with pytest.raises(JSONTypeError):
            decode_capability(data)

    def test_invalid_tags_item(self) -> None:
        """Test decode_capability with non-string in tags."""
        data: JSONObject = {
            "name": "cap",
            "strength": "strong",
            "tags": ["t1", 123],
            "description": "desc",
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_capability(data)
        assert "tags[1]" in str(exc_info.value)


class TestDecodeProfileErrors:
    """Tests for decode_profile error handling."""

    def test_invalid_capability_item(self) -> None:
        """Test decode_profile with non-dict capability."""
        data: JSONObject = {
            "capabilities": ["not a dict"],
            "technologies": [],
            "frameworks": [],
            "ml_backends": [],
            "data_formats": [],
            "task_types": [],
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_profile(data)
        assert "capabilities[0]" in str(exc_info.value)
