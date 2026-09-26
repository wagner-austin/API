"""Tests for the optimize request's feature_preset reader."""

from __future__ import annotations

import pytest
from covenant_ml.features import NON_TEMPORAL_FEATURE_PRESETS, FeaturePreset
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.worker.optimize_field_decoders import parse_feature_preset


class TestParseFeaturePreset:
    """Tests for parse_feature_preset."""

    def test_omitted_field_defaults_to_none(self) -> None:
        """An omitted field reads as NONE."""
        assert parse_feature_preset(None) is FeaturePreset.NONE

    def test_every_non_temporal_word_parses_to_its_member(self) -> None:
        """Each admitted word becomes the member that carries it."""
        for preset in NON_TEMPORAL_FEATURE_PRESETS:
            assert parse_feature_preset(preset.value) is preset

    def test_temporal_is_refused(self) -> None:
        """TEMPORAL is a member but not an admitted optimize preset."""
        with pytest.raises(
            JSONTypeError,
            match=r"^feature_preset must be one of: none, log_only, ratios_only, full$",
        ):
            parse_feature_preset("temporal")

    def test_unknown_word_is_refused(self) -> None:
        """A word no member carries is refused with the admitted words named."""
        with pytest.raises(
            JSONTypeError,
            match=r"^feature_preset must be one of: none, log_only, ratios_only, full$",
        ):
            parse_feature_preset("invalid")

    def test_non_string_is_refused(self) -> None:
        """A non-string value is refused by type."""
        with pytest.raises(JSONTypeError, match=r"^feature_preset must be a string$"):
            parse_feature_preset(123)
