"""Tests for ImzMLReader class."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import ImzMLReadError
from instrument_io._protocols.imzml import ImzMLParserProtocol
from instrument_io.readers.imzml import (
    ImzMLReader,
    _compute_image_dimensions,
    _get_spectrum_arrays,
    _is_imzml_file,
    _spectrum_to_imaging_spectrum,
)
from instrument_io.testing import FakeImzMLParser, hooks

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestIsImzmlFile:
    """Tests for _is_imzml_file function."""

    def test_returns_true_for_imzml(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzML"
        file.write_text("")
        assert _is_imzml_file(file) is True

    def test_returns_true_for_lowercase_imzml(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzml"
        file.write_text("")
        assert _is_imzml_file(file) is True

    def test_returns_false_for_non_imzml(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.mzML"
        file.write_text("")
        assert _is_imzml_file(file) is False

    def test_returns_false_for_directory(self, tmp_path: Path) -> None:
        assert _is_imzml_file(tmp_path) is False

    def test_returns_false_for_nonexistent(self, tmp_path: Path) -> None:
        file = tmp_path / "nonexistent.imzML"
        assert _is_imzml_file(file) is False


class TestGetSpectrumArrays:
    """Tests for _get_spectrum_arrays function."""

    def test_extracts_arrays(self) -> None:
        parser: ImzMLParserProtocol = FakeImzMLParser(
            coordinates=[(1, 1, 1)],
            spectra=[([100.0, 200.0, 300.0], [1000.0, 2000.0, 500.0])],
        )
        mz_values, intensities = _get_spectrum_arrays(parser, 0)

        assert mz_values == [100.0, 200.0, 300.0]
        assert intensities == [1000.0, 2000.0, 500.0]

    def test_extracts_arrays_at_index(self) -> None:
        parser: ImzMLParserProtocol = FakeImzMLParser(
            coordinates=[(1, 1, 1), (2, 1, 1)],
            spectra=[
                ([100.0, 200.0, 300.0], [1000.0, 2000.0, 500.0]),
                ([101.0, 201.0, 301.0], [1000.0, 2000.0, 500.0]),
            ],
        )
        mz_values, intensities = _get_spectrum_arrays(parser, 1)

        assert mz_values == [101.0, 201.0, 301.0]
        assert intensities == [1000.0, 2000.0, 500.0]


class TestSpectrumToImagingSpectrum:
    """Tests for _spectrum_to_imaging_spectrum function."""

    def test_converts_spectrum(self) -> None:
        parser: ImzMLParserProtocol = FakeImzMLParser(
            coordinates=[(1, 2, 1), (2, 2, 1)],
            spectra=[
                ([100.0, 200.0, 300.0], [1000.0, 2000.0, 500.0]),
                ([100.0, 200.0, 300.0], [1000.0, 2000.0, 500.0]),
            ],
        )
        result = _spectrum_to_imaging_spectrum(parser, "/path/file.imzML", 0, "positive")

        assert result["meta"]["source_path"] == "/path/file.imzML"
        assert result["meta"]["index"] == 0
        assert result["meta"]["coordinate"]["x"] == 1
        assert result["meta"]["coordinate"]["y"] == 2
        assert result["meta"]["coordinate"]["z"] is None  # z=1 becomes None
        assert result["meta"]["polarity"] == "positive"
        assert result["meta"]["ms_level"] == 1
        assert result["data"]["mz_values"] == [100.0, 200.0, 300.0]
        assert result["data"]["intensities"] == [1000.0, 2000.0, 500.0]
        assert result["stats"]["num_peaks"] == 3

    def test_negative_polarity(self) -> None:
        parser: ImzMLParserProtocol = FakeImzMLParser(
            coordinates=[(1, 1, 1)],
            spectra=[([100.0], [1000.0])],
        )
        result = _spectrum_to_imaging_spectrum(parser, "/test.imzML", 0, "negative")

        assert result["meta"]["polarity"] == "negative"

    def test_with_z_coordinate(self) -> None:
        parser: ImzMLParserProtocol = FakeImzMLParser(
            coordinates=[(1, 1, 3)],
            spectra=[([100.0], [1000.0])],
        )
        result = _spectrum_to_imaging_spectrum(parser, "/test.imzML", 0, "positive")

        assert result["meta"]["coordinate"]["z"] == 3


class TestComputeImageDimensions:
    """Tests for _compute_image_dimensions function."""

    def test_computes_dimensions(self) -> None:
        coords = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)]
        x, y = _compute_image_dimensions(coords)
        assert x == 2
        assert y == 2

    def test_empty_coordinates(self) -> None:
        x, y = _compute_image_dimensions([])
        assert x == 0
        assert y == 0

    def test_single_coordinate(self) -> None:
        coords = [(5, 3, 1)]
        x, y = _compute_image_dimensions(coords)
        assert x == 5
        assert y == 3


class TestImzMLReaderSupportsFormat:
    """Tests for ImzMLReader.supports_format method."""

    def test_supports_imzml_file(self, tmp_path: Path) -> None:
        reader = ImzMLReader()
        file = tmp_path / "sample.imzML"
        file.write_text("")
        assert reader.supports_format(file) is True

    def test_rejects_non_imzml_file(self, tmp_path: Path) -> None:
        reader = ImzMLReader()
        file = tmp_path / "sample.mzML"
        file.write_text("")
        assert reader.supports_format(file) is False


class TestImzMLReaderGetFileInfo:
    """Tests for ImzMLReader.get_file_info method."""

    def test_raises_for_non_imzml(self, tmp_path: Path) -> None:
        reader = ImzMLReader()
        file = tmp_path / "sample.mzML"
        file.write_text("")

        with pytest.raises(ImzMLReadError, match="Not an imzML file"):
            reader.get_file_info(file)

    def test_returns_file_info(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzML"
        file.write_text("")

        fake_parser = FakeImzMLParser(
            coordinates=[(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)],
            spectra=[([100.0], [1000.0])] * 4,
            polarity="positive",
            spectrum_mode="centroid",
        )
        hooks.open_imzml = lambda p: fake_parser

        reader = ImzMLReader()
        info = reader.get_file_info(file)

        assert info["source_path"] == str(file)
        assert info["num_spectra"] == 4
        assert info["polarity"] == "positive"
        assert info["spectrum_mode"] == "centroid"
        assert info["x_pixels"] == 2
        assert info["y_pixels"] == 2

    def test_returns_negative_polarity(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzML"
        file.write_text("")

        fake_parser = FakeImzMLParser(
            coordinates=[(1, 1, 1)],
            spectra=[([100.0], [1000.0])],
            polarity="negative",
            spectrum_mode="profile",
        )
        hooks.open_imzml = lambda p: fake_parser

        reader = ImzMLReader()
        info = reader.get_file_info(file)

        assert info["polarity"] == "negative"
        assert info["spectrum_mode"] == "profile"


class TestImzMLReaderGetCoordinates:
    """Tests for ImzMLReader.get_coordinates method."""

    def test_raises_for_non_imzml(self, tmp_path: Path) -> None:
        reader = ImzMLReader()
        file = tmp_path / "sample.mzML"
        file.write_text("")

        with pytest.raises(ImzMLReadError, match="Not an imzML file"):
            reader.get_coordinates(file)

    def test_returns_coordinates(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzML"
        file.write_text("")

        fake_parser = FakeImzMLParser(
            coordinates=[(1, 1, 1), (2, 1, 1), (1, 2, 1)],
            spectra=[([100.0], [1000.0])] * 3,
        )
        hooks.open_imzml = lambda p: fake_parser

        reader = ImzMLReader()
        coords = reader.get_coordinates(file)

        assert len(coords) == 3
        assert coords[0]["x"] == 1
        assert coords[0]["y"] == 1
        assert coords[1]["x"] == 2
        assert coords[2]["y"] == 2


class TestImzMLReaderIterSpectra:
    """Tests for ImzMLReader.iter_spectra method."""

    def test_raises_for_non_imzml(self, tmp_path: Path) -> None:
        reader = ImzMLReader()
        file = tmp_path / "sample.mzML"
        file.write_text("")

        with pytest.raises(ImzMLReadError, match="Not an imzML file"):
            list(reader.iter_spectra(file))

    def test_iterates_all_spectra(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzML"
        file.write_text("")

        fake_parser = FakeImzMLParser(
            coordinates=[(1, 1, 1), (2, 1, 1)],
            spectra=[
                ([100.0, 200.0, 300.0], [1000.0, 2000.0, 500.0]),
                ([101.0, 201.0, 301.0], [1000.0, 2000.0, 500.0]),
            ],
            polarity="negative",
        )
        hooks.open_imzml = lambda p: fake_parser

        reader = ImzMLReader()
        spectra = list(reader.iter_spectra(file))

        assert len(spectra) == 2
        assert spectra[0]["meta"]["index"] == 0
        assert spectra[1]["meta"]["index"] == 1
        assert spectra[0]["meta"]["polarity"] == "negative"


class TestImzMLReaderReadSpectrum:
    """Tests for ImzMLReader.read_spectrum method."""

    def test_raises_for_non_imzml(self, tmp_path: Path) -> None:
        reader = ImzMLReader()
        file = tmp_path / "sample.mzML"
        file.write_text("")

        with pytest.raises(ImzMLReadError, match="Not an imzML file"):
            reader.read_spectrum(file, 0)

    def test_raises_for_negative_index(self, tmp_path: Path) -> None:
        reader = ImzMLReader()
        file = tmp_path / "sample.imzML"
        file.write_text("")

        with pytest.raises(ImzMLReadError, match="Invalid index"):
            reader.read_spectrum(file, -1)

    def test_reads_spectrum_by_index(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzML"
        file.write_text("")

        fake_parser = FakeImzMLParser(
            coordinates=[(1, 1, 1), (2, 1, 1), (3, 1, 1)],
            spectra=[
                ([100.0], [1000.0]),
                ([101.0], [1001.0]),
                ([102.0], [1002.0]),
            ],
        )
        hooks.open_imzml = lambda p: fake_parser

        reader = ImzMLReader()
        spectrum = reader.read_spectrum(file, 1)

        assert spectrum["meta"]["index"] == 1
        assert spectrum["meta"]["coordinate"]["x"] == 2

    def test_raises_for_index_out_of_range(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzML"
        file.write_text("")

        fake_parser = FakeImzMLParser(
            coordinates=[(1, 1, 1)],
            spectra=[([100.0], [1000.0])],
        )
        hooks.open_imzml = lambda p: fake_parser

        reader = ImzMLReader()
        with pytest.raises(ImzMLReadError, match="Spectrum index 5 not found"):
            reader.read_spectrum(file, 5)


class TestImzMLReaderReadSpectrumAtCoordinate:
    """Tests for ImzMLReader.read_spectrum_at_coordinate method."""

    def test_raises_for_non_imzml(self, tmp_path: Path) -> None:
        reader = ImzMLReader()
        file = tmp_path / "sample.mzML"
        file.write_text("")

        with pytest.raises(ImzMLReadError, match="Not an imzML file"):
            reader.read_spectrum_at_coordinate(file, 1, 1)

    def test_reads_spectrum_at_coordinate(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzML"
        file.write_text("")

        fake_parser = FakeImzMLParser(
            coordinates=[(1, 1, 1), (2, 1, 1), (1, 2, 1)],
            spectra=[
                ([100.0], [1000.0]),
                ([101.0], [1001.0]),
                ([102.0], [1002.0]),
            ],
        )
        hooks.open_imzml = lambda p: fake_parser

        reader = ImzMLReader()
        spectrum = reader.read_spectrum_at_coordinate(file, x=2, y=1, z=1)

        assert spectrum["meta"]["coordinate"]["x"] == 2
        assert spectrum["meta"]["coordinate"]["y"] == 1

    def test_raises_for_coordinate_not_found(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzML"
        file.write_text("")

        fake_parser = FakeImzMLParser(
            coordinates=[(1, 1, 1), (2, 1, 1)],
            spectra=[
                ([100.0], [1000.0]),
                ([101.0], [1001.0]),
            ],
        )
        hooks.open_imzml = lambda p: fake_parser

        reader = ImzMLReader()
        with pytest.raises(ImzMLReadError, match=r"Coordinate \(99, 99, 1\) not found"):
            reader.read_spectrum_at_coordinate(file, x=99, y=99, z=1)


class TestImzMLReaderCountSpectra:
    """Tests for ImzMLReader.count_spectra method."""

    def test_raises_for_non_imzml(self, tmp_path: Path) -> None:
        reader = ImzMLReader()
        file = tmp_path / "sample.mzML"
        file.write_text("")

        with pytest.raises(ImzMLReadError, match="Not an imzML file"):
            reader.count_spectra(file)

    def test_returns_count(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.imzML"
        file.write_text("")

        fake_parser = FakeImzMLParser(
            coordinates=[(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (3, 1, 1)],
            spectra=[([100.0], [1000.0])] * 5,
        )
        hooks.open_imzml = lambda p: fake_parser

        reader = ImzMLReader()
        count = reader.count_spectra(file)

        assert count == 5
