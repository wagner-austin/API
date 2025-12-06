"""Tests for MGF decoder functions."""

from __future__ import annotations

import pytest

from instrument_io._decoders.mgf import (
    _compute_mgf_spectrum_stats,
    _decode_charge_value,
    _decode_mgf_polarity,
    _decode_mgf_precursor,
    _decode_mgf_retention_time,
    _decode_mgf_scan_number,
    _decode_mgf_title,
    _decode_pepmass,
    _is_numeric_string,
    _make_mgf_spectrum_data,
    _make_mgf_spectrum_meta,
)
from instrument_io._exceptions import DecodingError


class TestDecodeMGFTitle:
    """Tests for _decode_mgf_title."""

    def test_title_string(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "title": "Scan 1"
        }
        assert _decode_mgf_title(params) == "Scan 1"

    def test_title_none(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {}
        assert _decode_mgf_title(params) == ""

    def test_title_non_string(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "title": 123
        }
        assert _decode_mgf_title(params) == "123"


class TestDecodePepmass:
    """Tests for _decode_pepmass."""

    def test_pepmass_none_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _decode_pepmass(None)
        assert "Missing pepmass" in str(exc_info.value)

    def test_pepmass_empty_tuple_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _decode_pepmass(())
        assert "Empty pepmass tuple" in str(exc_info.value)

    def test_pepmass_tuple_none_mz_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _decode_pepmass((None,))
        assert "Missing m/z" in str(exc_info.value)

    def test_pepmass_tuple_single_value(self) -> None:
        mz, intensity = _decode_pepmass((500.5,))
        assert mz == 500.5
        assert intensity is None

    def test_pepmass_tuple_with_none_intensity(self) -> None:
        mz, intensity = _decode_pepmass((500.5, None))
        assert mz == 500.5
        assert intensity is None

    def test_pepmass_tuple_with_intensity(self) -> None:
        mz, intensity = _decode_pepmass((500.5, 1000.0))
        assert mz == 500.5
        assert intensity == 1000.0

    def test_pepmass_int(self) -> None:
        mz, intensity = _decode_pepmass(500)
        assert mz == 500.0
        assert intensity is None

    def test_pepmass_float(self) -> None:
        mz, intensity = _decode_pepmass(500.5)
        assert mz == 500.5
        assert intensity is None

    def test_pepmass_invalid_type_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _decode_pepmass("invalid")
        assert "Invalid pepmass type" in str(exc_info.value)


class TestDecodeChargeValue:
    """Tests for _decode_charge_value."""

    def test_charge_none(self) -> None:
        assert _decode_charge_value(None) is None

    def test_charge_int(self) -> None:
        assert _decode_charge_value(2) == 2

    def test_charge_list(self) -> None:
        assert _decode_charge_value([3, 2]) == 3

    def test_charge_list_empty(self) -> None:
        assert _decode_charge_value([]) is None

    def test_charge_string_positive(self) -> None:
        assert _decode_charge_value("2+") == 2

    def test_charge_string_negative(self) -> None:
        assert _decode_charge_value("2-") == -2

    def test_charge_string_no_sign(self) -> None:
        assert _decode_charge_value("3") == 3

    def test_charge_string_non_numeric(self) -> None:
        assert _decode_charge_value("abc") is None

    def test_charge_float(self) -> None:
        # Float is not handled, returns None
        assert _decode_charge_value(2.5) is None


class TestDecodeMGFPrecursor:
    """Tests for _decode_mgf_precursor."""

    def test_precursor_basic(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "pepmass": (500.5, 1000.0),
            "charge": "2+",
        }
        precursor = _decode_mgf_precursor(params)
        assert precursor["mz"] == 500.5
        assert precursor["charge"] == 2
        assert precursor["intensity"] == 1000.0
        assert precursor["isolation_window"] is None

    def test_precursor_no_charge(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "pepmass": 500.5
        }
        precursor = _decode_mgf_precursor(params)
        assert precursor["mz"] == 500.5
        assert precursor["charge"] is None


class TestDecodeMGFPolarity:
    """Tests for _decode_mgf_polarity."""

    def test_polarity_positive(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "charge": "2+"
        }
        assert _decode_mgf_polarity(params) == "positive"

    def test_polarity_negative(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "charge": "2-"
        }
        assert _decode_mgf_polarity(params) == "negative"

    def test_polarity_unknown_no_charge(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {}
        assert _decode_mgf_polarity(params) == "unknown"

    def test_polarity_zero_charge(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "charge": 0
        }
        assert _decode_mgf_polarity(params) == "unknown"


class TestDecodeMGFScanNumber:
    """Tests for _decode_mgf_scan_number."""

    def test_scan_from_title_with_equals(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "title": "Spectrum scan=1234"
        }
        assert _decode_mgf_scan_number(params, 0) == 1234

    def test_scan_from_title_with_equals_non_digit_fallback_to_space(self) -> None:
        """Test when scan= has non-digit, falls back to 'scan ' pattern."""
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "title": "scan=abc Scan 5678 from file"
        }
        # scan= extraction fails (abc not digit), falls back to "Scan " and gets 5678
        assert _decode_mgf_scan_number(params, 0) == 5678

    def test_scan_from_title_with_space(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "title": "Scan 5678 from file"
        }
        assert _decode_mgf_scan_number(params, 0) == 5678

    def test_scan_from_scans_int(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "scans": 999
        }
        assert _decode_mgf_scan_number(params, 0) == 999

    def test_scan_from_scans_string(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "scans": "888"
        }
        assert _decode_mgf_scan_number(params, 0) == 888

    def test_scan_fallback_to_index(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {}
        assert _decode_mgf_scan_number(params, 5) == 6

    def test_scan_title_non_numeric(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "title": "Scan abc"
        }
        assert _decode_mgf_scan_number(params, 2) == 3

    def test_scan_scans_non_numeric_string(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "scans": "xyz"
        }
        assert _decode_mgf_scan_number(params, 1) == 2


class TestIsNumericString:
    """Tests for _is_numeric_string."""

    def test_integer_string(self) -> None:
        assert _is_numeric_string("123") is True

    def test_float_string(self) -> None:
        assert _is_numeric_string("123.45") is True

    def test_negative_integer(self) -> None:
        assert _is_numeric_string("-123") is True

    def test_negative_float(self) -> None:
        assert _is_numeric_string("-123.45") is True

    def test_empty_string(self) -> None:
        assert _is_numeric_string("") is False

    def test_whitespace_only(self) -> None:
        assert _is_numeric_string("   ") is False

    def test_non_numeric(self) -> None:
        assert _is_numeric_string("abc") is False

    def test_multiple_decimals(self) -> None:
        assert _is_numeric_string("12.34.56") is False

    def test_valid_float_with_leading_space(self) -> None:
        assert _is_numeric_string("  123.45") is True


class TestDecodeMGFRetentionTime:
    """Tests for _decode_mgf_retention_time."""

    def test_rtinseconds_float(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "rtinseconds": 120.0
        }
        assert _decode_mgf_retention_time(params) == 2.0

    def test_rtinseconds_int(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "rtinseconds": 180
        }
        assert _decode_mgf_retention_time(params) == 3.0

    def test_rtinseconds_string(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "rtinseconds": "240"
        }
        assert _decode_mgf_retention_time(params) == 4.0

    def test_retentiontime_float(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "retentiontime": 5.5
        }
        assert _decode_mgf_retention_time(params) == 5.5

    def test_retentiontime_string(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "retentiontime": "6.5"
        }
        assert _decode_mgf_retention_time(params) == 6.5

    def test_no_retention_time(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {}
        assert _decode_mgf_retention_time(params) == 0.0

    def test_rtinseconds_non_numeric_string(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "rtinseconds": "abc"
        }
        assert _decode_mgf_retention_time(params) == 0.0

    def test_retentiontime_non_numeric_string(self) -> None:
        params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None] = {
            "retentiontime": "xyz"
        }
        assert _decode_mgf_retention_time(params) == 0.0


class TestComputeMGFSpectrumStats:
    """Tests for _compute_mgf_spectrum_stats."""

    def test_empty_lists(self) -> None:
        stats = _compute_mgf_spectrum_stats([], [])
        assert stats["num_peaks"] == 0
        assert stats["mz_min"] == 0.0
        assert stats["mz_max"] == 0.0
        assert stats["base_peak_mz"] == 0.0
        assert stats["base_peak_intensity"] == 0.0

    def test_single_peak(self) -> None:
        stats = _compute_mgf_spectrum_stats([100.5], [500.0])
        assert stats["num_peaks"] == 1
        assert stats["mz_min"] == 100.5
        assert stats["mz_max"] == 100.5
        assert stats["base_peak_mz"] == 100.5
        assert stats["base_peak_intensity"] == 500.0

    def test_multiple_peaks(self) -> None:
        stats = _compute_mgf_spectrum_stats([100.0, 200.0, 300.0], [50.0, 1000.0, 100.0])
        assert stats["num_peaks"] == 3
        assert stats["mz_min"] == 100.0
        assert stats["mz_max"] == 300.0
        assert stats["base_peak_mz"] == 200.0
        assert stats["base_peak_intensity"] == 1000.0


class TestMakeMGFSpectrumMeta:
    """Tests for _make_mgf_spectrum_meta."""

    def test_make_spectrum_meta(self) -> None:
        meta = _make_mgf_spectrum_meta(
            source_path="/path/to/file.mgf",
            scan_number=123,
            retention_time=10.5,
            polarity="positive",
            total_ion_current=5000.0,
        )
        assert meta["source_path"] == "/path/to/file.mgf"
        assert meta["scan_number"] == 123
        assert meta["retention_time"] == 10.5
        assert meta["ms_level"] == 2
        assert meta["polarity"] == "positive"
        assert meta["total_ion_current"] == 5000.0


class TestMakeMGFSpectrumData:
    """Tests for _make_mgf_spectrum_data."""

    def test_make_spectrum_data(self) -> None:
        data = _make_mgf_spectrum_data([100.0, 200.0], [50.0, 100.0])
        assert data["mz_values"] == [100.0, 200.0]
        assert data["intensities"] == [50.0, 100.0]
