"""Tests for imzML decoder functions."""

from __future__ import annotations

from instrument_io._decoders.imzml import (
    _compute_imzml_spectrum_stats,
    _decode_coordinate,
    _decode_imzml_polarity,
    _decode_spectrum_mode,
    _make_imzml_spectrum_data,
    _make_imzml_spectrum_meta,
)


class TestDecodeCoordinate:
    """Tests for _decode_coordinate."""

    def test_basic_coordinate(self) -> None:
        coord = _decode_coordinate((10, 20, 1))
        assert coord["x"] == 10
        assert coord["y"] == 20
        assert coord["z"] is None  # z=1 treated as None for 2D imaging

    def test_3d_coordinate(self) -> None:
        coord = _decode_coordinate((10, 20, 5))
        assert coord["x"] == 10
        assert coord["y"] == 20
        assert coord["z"] == 5


class TestDecodeImzmlPolarity:
    """Tests for _decode_imzml_polarity."""

    def test_positive(self) -> None:
        assert _decode_imzml_polarity("positive") == "positive"

    def test_positive_uppercase(self) -> None:
        assert _decode_imzml_polarity("POSITIVE") == "positive"

    def test_negative(self) -> None:
        assert _decode_imzml_polarity("negative") == "negative"

    def test_unknown(self) -> None:
        assert _decode_imzml_polarity("mixed") == "unknown"

    def test_empty_string(self) -> None:
        assert _decode_imzml_polarity("") == "unknown"


class TestDecodeSpectrumMode:
    """Tests for _decode_spectrum_mode."""

    def test_centroid(self) -> None:
        assert _decode_spectrum_mode("centroid") == "centroid"

    def test_profile(self) -> None:
        assert _decode_spectrum_mode("profile") == "profile"

    def test_unknown(self) -> None:
        assert _decode_spectrum_mode("other") == "unknown"

    def test_empty_string(self) -> None:
        assert _decode_spectrum_mode("") == "unknown"


class TestComputeImzmlSpectrumStats:
    """Tests for _compute_imzml_spectrum_stats."""

    def test_empty_data(self) -> None:
        stats = _compute_imzml_spectrum_stats([], [])
        assert stats["num_peaks"] == 0
        assert stats["mz_min"] == 0.0
        assert stats["mz_max"] == 0.0
        assert stats["base_peak_mz"] == 0.0
        assert stats["base_peak_intensity"] == 0.0

    def test_empty_mz_values(self) -> None:
        stats = _compute_imzml_spectrum_stats([], [100.0, 200.0])
        assert stats["num_peaks"] == 0

    def test_empty_intensities(self) -> None:
        stats = _compute_imzml_spectrum_stats([100.0, 200.0], [])
        assert stats["num_peaks"] == 0

    def test_single_peak(self) -> None:
        stats = _compute_imzml_spectrum_stats([500.0], [1000.0])
        assert stats["num_peaks"] == 1
        assert stats["mz_min"] == 500.0
        assert stats["mz_max"] == 500.0
        assert stats["base_peak_mz"] == 500.0
        assert stats["base_peak_intensity"] == 1000.0

    def test_multiple_peaks(self) -> None:
        mz = [100.0, 200.0, 300.0]
        intensities = [500.0, 1500.0, 1000.0]
        stats = _compute_imzml_spectrum_stats(mz, intensities)
        assert stats["num_peaks"] == 3
        assert stats["mz_min"] == 100.0
        assert stats["mz_max"] == 300.0
        assert stats["base_peak_mz"] == 200.0
        assert stats["base_peak_intensity"] == 1500.0


class TestMakeImzmlSpectrumData:
    """Tests for _make_imzml_spectrum_data."""

    def test_creates_typeddict(self) -> None:
        data = _make_imzml_spectrum_data([100.0, 200.0], [500.0, 1000.0])
        assert data["mz_values"] == [100.0, 200.0]
        assert data["intensities"] == [500.0, 1000.0]


class TestMakeImzmlSpectrumMeta:
    """Tests for _make_imzml_spectrum_meta."""

    def test_creates_typeddict(self) -> None:
        coord = _decode_coordinate((10, 20, 1))
        meta = _make_imzml_spectrum_meta(
            source_path="/path/to/file.imzML",
            index=5,
            coordinate=coord,
            polarity="positive",
            total_ion_current=123456.0,
        )
        assert meta["source_path"] == "/path/to/file.imzML"
        assert meta["index"] == 5
        assert meta["coordinate"]["x"] == 10
        assert meta["ms_level"] == 1
        assert meta["polarity"] == "positive"
        assert meta["total_ion_current"] == 123456.0
