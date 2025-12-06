"""Tests for Agilent decoder functions."""

from __future__ import annotations

import pytest

from instrument_io._decoders.agilent import (
    _compute_chromatogram_stats,
    _decode_intensities_1d,
    _decode_intensities_2d,
    _decode_retention_times,
    _decode_signal_type,
    _is_1d_list,
    _is_2d_list,
    _make_chromatogram_data,
    _narrow_tolist_1d,
    _narrow_tolist_2d,
    _percentile,
    _sum_2d_to_tic,
)
from instrument_io._exceptions import DecodingError


class TestIs1dList:
    """Tests for _is_1d_list TypeGuard."""

    def test_empty_list_returns_false(self) -> None:
        result: list[float] | list[list[float]] = []
        assert _is_1d_list(result) is False

    def test_1d_float_list_returns_true(self) -> None:
        result: list[float] | list[list[float]] = [1.0, 2.0, 3.0]
        assert _is_1d_list(result) is True

    def test_1d_int_list_returns_true(self) -> None:
        result: list[int] | list[list[int]] = [1, 2, 3]
        assert _is_1d_list(result) is True

    def test_2d_list_returns_false(self) -> None:
        result: list[float] | list[list[float]] = [[1.0, 2.0], [3.0, 4.0]]
        assert _is_1d_list(result) is False


class TestIs2dList:
    """Tests for _is_2d_list TypeGuard."""

    def test_empty_list_returns_false(self) -> None:
        result: list[float] | list[list[float]] = []
        assert _is_2d_list(result) is False

    def test_1d_list_returns_false(self) -> None:
        result: list[float] | list[list[float]] = [1.0, 2.0, 3.0]
        assert _is_2d_list(result) is False

    def test_2d_list_returns_true(self) -> None:
        result: list[float] | list[list[float]] = [[1.0, 2.0], [3.0, 4.0]]
        assert _is_2d_list(result) is True


class TestNarrowTolist1d:
    """Tests for _narrow_tolist_1d."""

    def test_empty_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _narrow_tolist_1d([])
        assert "Empty array" in str(exc_info.value)

    def test_2d_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _narrow_tolist_1d([[1.0, 2.0]])
        assert "Expected 1D" in str(exc_info.value)

    def test_1d_float_list_succeeds(self) -> None:
        result = _narrow_tolist_1d([1.0, 2.0, 3.0])
        assert result == [1.0, 2.0, 3.0]

    def test_1d_int_list_converts(self) -> None:
        result = _narrow_tolist_1d([1, 2, 3])
        assert result == [1.0, 2.0, 3.0]


class TestNarrowTolist2d:
    """Tests for _narrow_tolist_2d."""

    def test_empty_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _narrow_tolist_2d([])
        assert "Empty array" in str(exc_info.value)

    def test_1d_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _narrow_tolist_2d([1.0, 2.0, 3.0])
        assert "Expected 2D" in str(exc_info.value)

    def test_2d_float_list_succeeds(self) -> None:
        result = _narrow_tolist_2d([[1.0, 2.0], [3.0, 4.0]])
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_2d_int_list_converts(self) -> None:
        result = _narrow_tolist_2d([[1, 2], [3, 4]])
        assert result == [[1.0, 2.0], [3.0, 4.0]]


class TestDecodeRetentionTimes:
    """Tests for _decode_retention_times."""

    def test_empty_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _decode_retention_times([])
        assert "Empty retention times" in str(exc_info.value)

    def test_valid_list_succeeds(self) -> None:
        result = _decode_retention_times([0.0, 1.0, 2.0])
        assert result == [0.0, 1.0, 2.0]


class TestDecodeIntensities1d:
    """Tests for _decode_intensities_1d."""

    def test_empty_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _decode_intensities_1d([])
        assert "Empty intensities" in str(exc_info.value)

    def test_valid_list_succeeds(self) -> None:
        result = _decode_intensities_1d([100.0, 200.0, 150.0])
        assert result == [100.0, 200.0, 150.0]


class TestDecodeIntensities2d:
    """Tests for _decode_intensities_2d."""

    def test_empty_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _decode_intensities_2d([])
        assert "Empty 2D intensities" in str(exc_info.value)

    def test_row_out_of_range_raises(self) -> None:
        data = [[100.0, 200.0]]
        with pytest.raises(DecodingError) as exc_info:
            _decode_intensities_2d(data, row_index=5)
        assert "out of range" in str(exc_info.value)

    def test_valid_extraction(self) -> None:
        data = [[100.0, 200.0], [300.0, 400.0]]
        result = _decode_intensities_2d(data, row_index=1)
        assert result == [300.0, 400.0]

    def test_default_row_index(self) -> None:
        data = [[100.0, 200.0], [300.0, 400.0]]
        result = _decode_intensities_2d(data)
        assert result == [100.0, 200.0]


class TestSum2dToTic:
    """Tests for _sum_2d_to_tic."""

    def test_empty_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _sum_2d_to_tic([])
        assert "Empty 2D array" in str(exc_info.value)

    def test_empty_first_row_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _sum_2d_to_tic([[]])
        assert "Empty first row" in str(exc_info.value)

    def test_mismatched_row_lengths_raises(self) -> None:
        data = [[100.0, 200.0], [300.0]]  # Row 1 has 1 element, not 2
        with pytest.raises(DecodingError) as exc_info:
            _sum_2d_to_tic(data)
        assert "Row 1 has 1 points" in str(exc_info.value)

    def test_valid_sum(self) -> None:
        data = [[100.0, 200.0], [300.0, 400.0]]
        result = _sum_2d_to_tic(data)
        assert result == [400.0, 600.0]


class TestDecodeSignalType:
    """Tests for _decode_signal_type."""

    def test_tic_detection(self) -> None:
        assert _decode_signal_type("TIC") == "TIC"
        assert _decode_signal_type("Total Ion Current") == "TIC"

    def test_eic_detection(self) -> None:
        assert _decode_signal_type("EIC") == "EIC"
        assert _decode_signal_type("Extracted Ion") == "EIC"

    def test_dad_detection(self) -> None:
        assert _decode_signal_type("DAD") == "DAD"
        assert _decode_signal_type("Diode Array") == "DAD"

    def test_uv_detection(self) -> None:
        assert _decode_signal_type("UV") == "UV"

    def test_fid_detection(self) -> None:
        assert _decode_signal_type("FID") == "FID"

    def test_ms_detection(self) -> None:
        assert _decode_signal_type("MS") == "MS"

    def test_unknown_defaults_to_ms(self) -> None:
        assert _decode_signal_type("Unknown Detector") == "MS"


class TestComputeChromatogramStats:
    """Tests for _compute_chromatogram_stats."""

    def test_empty_data_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _compute_chromatogram_stats([], [])
        assert "Empty data" in str(exc_info.value)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(DecodingError) as exc_info:
            _compute_chromatogram_stats([1.0, 2.0], [100.0])
        assert "Mismatched lengths" in str(exc_info.value)

    def test_single_point(self) -> None:
        stats = _compute_chromatogram_stats([1.0], [100.0])
        assert stats["num_points"] == 1
        assert stats["rt_min"] == 1.0
        assert stats["rt_max"] == 1.0
        assert stats["rt_step_mean"] == 0.0
        assert stats["intensity_min"] == 100.0
        assert stats["intensity_max"] == 100.0

    def test_multiple_points(self) -> None:
        rt = [0.0, 1.0, 2.0]
        intensities = [100.0, 200.0, 150.0]
        stats = _compute_chromatogram_stats(rt, intensities)
        assert stats["num_points"] == 3
        assert stats["rt_min"] == 0.0
        assert stats["rt_max"] == 2.0
        assert stats["rt_step_mean"] == 1.0
        assert stats["intensity_min"] == 100.0
        assert stats["intensity_max"] == 200.0


class TestPercentile:
    """Tests for _percentile."""

    def test_single_value(self) -> None:
        assert _percentile([100.0], 0.99) == 100.0

    def test_percentile_99(self) -> None:
        values = [i * 1.0 for i in range(100)]
        result = _percentile(values, 0.99)
        assert 98.0 <= result <= 99.0


class TestMakeChromatogramData:
    """Tests for _make_chromatogram_data."""

    def test_creates_typeddict(self) -> None:
        data = _make_chromatogram_data([0.0, 1.0], [100.0, 200.0])
        assert data["retention_times"] == [0.0, 1.0]
        assert data["intensities"] == [100.0, 200.0]
