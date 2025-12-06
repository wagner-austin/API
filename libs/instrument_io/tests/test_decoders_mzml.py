"""Tests for mzML decoder functions."""

from __future__ import annotations

import pytest

from instrument_io._decoders.mzml import (
    _compute_spectrum_stats,
    _decode_intensity_array,
    _decode_ms_level,
    _decode_mz_array,
    _decode_polarity,
    _decode_retention_time,
    _decode_scan_number,
    _make_spectrum_data,
)
from instrument_io._exceptions import DecodingError


class TestDecodeMzArray:
    """Tests for _decode_mz_array."""

    def test_empty_array(self) -> None:
        assert _decode_mz_array([]) == []

    def test_normal_array(self) -> None:
        assert _decode_mz_array([100.5, 200.3, 300.1]) == [100.5, 200.3, 300.1]


class TestDecodeIntensityArray:
    """Tests for _decode_intensity_array."""

    def test_empty_array(self) -> None:
        assert _decode_intensity_array([]) == []

    def test_normal_array(self) -> None:
        assert _decode_intensity_array([500.0, 1000.0, 750.0]) == [500.0, 1000.0, 750.0]


class TestDecodePolarity:
    """Tests for _decode_polarity."""

    def test_polarity_none(self) -> None:
        assert _decode_polarity(None) == "unknown"

    def test_polarity_positive_word(self) -> None:
        assert _decode_polarity("positive") == "positive"

    def test_polarity_positive_symbol(self) -> None:
        assert _decode_polarity("+") == "positive"

    def test_polarity_negative_word(self) -> None:
        assert _decode_polarity("negative") == "negative"

    def test_polarity_negative_symbol(self) -> None:
        assert _decode_polarity("-") == "negative"

    def test_polarity_unknown(self) -> None:
        assert _decode_polarity("unknown_value") == "unknown"

    def test_polarity_case_insensitive(self) -> None:
        assert _decode_polarity("POSITIVE") == "positive"
        assert _decode_polarity("NEGATIVE") == "negative"


class TestDecodeMsLevel:
    """Tests for _decode_ms_level."""

    def test_level_none_defaults_to_1(self) -> None:
        assert _decode_ms_level(None) == 1

    def test_level_1(self) -> None:
        assert _decode_ms_level(1) == 1

    def test_level_2(self) -> None:
        assert _decode_ms_level(2) == 2

    def test_level_3(self) -> None:
        assert _decode_ms_level(3) == 3

    def test_level_from_string(self) -> None:
        assert _decode_ms_level("2") == 2

    def test_level_from_float(self) -> None:
        assert _decode_ms_level(2.0) == 2

    def test_level_invalid_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _decode_ms_level(4)
        assert "Invalid MS level" in str(exc_info.value)


class TestDecodeRetentionTime:
    """Tests for _decode_retention_time."""

    def test_rt_none(self) -> None:
        assert _decode_retention_time(None) == 0.0

    def test_rt_in_minutes(self) -> None:
        assert _decode_retention_time(5.5) == 5.5

    def test_rt_in_seconds_converts(self) -> None:
        # Values > 100 are assumed to be in seconds
        assert _decode_retention_time(300.0) == 5.0

    def test_rt_from_string(self) -> None:
        assert _decode_retention_time("10.5") == 10.5

    def test_rt_from_int(self) -> None:
        assert _decode_retention_time(15) == 15.0


class TestDecodeScanNumber:
    """Tests for _decode_scan_number."""

    def test_scan_none(self) -> None:
        assert _decode_scan_number(None) == 0

    def test_scan_int(self) -> None:
        assert _decode_scan_number(1234) == 1234

    def test_scan_with_scan_equals(self) -> None:
        assert _decode_scan_number("scan=1234") == 1234

    def test_scan_with_cycle_equals(self) -> None:
        assert _decode_scan_number("sample=1 period=1 cycle=22 experiment=1") == 22

    def test_scan_numeric_string(self) -> None:
        assert _decode_scan_number("5678") == 5678

    def test_scan_s_prefix(self) -> None:
        assert _decode_scan_number("S1234") == 1234

    def test_scan_negative_number(self) -> None:
        assert _decode_scan_number("-100") == -100

    def test_scan_extract_from_string(self) -> None:
        assert _decode_scan_number("spectrum_9999_data") == 9999

    def test_scan_no_number_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _decode_scan_number("no_numbers_here")
        assert "Cannot parse" in str(exc_info.value)


class TestComputeSpectrumStats:
    """Tests for _compute_spectrum_stats."""

    def test_empty_lists(self) -> None:
        stats = _compute_spectrum_stats([], [])
        assert stats["num_peaks"] == 0
        assert stats["mz_min"] == 0.0
        assert stats["mz_max"] == 0.0
        assert stats["base_peak_mz"] == 0.0
        assert stats["base_peak_intensity"] == 0.0

    def test_single_peak(self) -> None:
        stats = _compute_spectrum_stats([100.5], [500.0])
        assert stats["num_peaks"] == 1
        assert stats["mz_min"] == 100.5
        assert stats["mz_max"] == 100.5
        assert stats["base_peak_mz"] == 100.5
        assert stats["base_peak_intensity"] == 500.0

    def test_multiple_peaks(self) -> None:
        stats = _compute_spectrum_stats([100.0, 200.0, 300.0], [50.0, 1000.0, 100.0])
        assert stats["num_peaks"] == 3
        assert stats["mz_min"] == 100.0
        assert stats["mz_max"] == 300.0
        assert stats["base_peak_mz"] == 200.0
        assert stats["base_peak_intensity"] == 1000.0


class TestMakeSpectrumData:
    """Tests for _make_spectrum_data."""

    def test_make_spectrum_data(self) -> None:
        data = _make_spectrum_data([100.0, 200.0], [50.0, 100.0])
        assert data["mz_values"] == [100.0, 200.0]
        assert data["intensities"] == [50.0, 100.0]
