"""Integration tests for ImzMLReader using real imzML files.

Uses sample imzML file from fixtures to verify that ImzMLReader
correctly reads imaging mass spectrometry data with spatial coordinates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io.readers.imzml import ImzMLReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_IMZML = FIXTURES_DIR / "sample.imzML"


def _get_sample_imzml() -> Path:
    """Get path to sample imzML fixture, skip if not found."""
    if not SAMPLE_IMZML.exists():
        pytest.skip("sample.imzML test fixture not found")
    return SAMPLE_IMZML


class TestImzMLReaderIntegration:
    """Integration tests using real imzML fixture."""

    def test_supports_format_real_file(self) -> None:
        """Test format detection for real imzML file."""
        sample_imzml = _get_sample_imzml()
        reader = ImzMLReader()
        assert reader.supports_format(sample_imzml) is True

    def test_get_file_info_real_file(self) -> None:
        """Test reading file info from real imzML file."""
        sample_imzml = _get_sample_imzml()
        reader = ImzMLReader()
        info = reader.get_file_info(sample_imzml)

        # Verify structure
        assert info["source_path"] == str(sample_imzml)
        assert info["num_spectra"] == 4  # 2x2 image
        assert info["x_pixels"] == 2
        assert info["y_pixels"] == 2
        assert info["polarity"] in ("positive", "negative", "unknown")
        assert info["spectrum_mode"] in ("centroid", "profile", "unknown")

    def test_get_coordinates_real_file(self) -> None:
        """Test reading coordinates from real imzML file."""
        sample_imzml = _get_sample_imzml()
        reader = ImzMLReader()
        coords = reader.get_coordinates(sample_imzml)

        # 2x2 image has 4 coordinates
        assert len(coords) == 4

        # Verify all coordinates have valid structure
        for coord in coords:
            assert coord["x"] >= 1
            assert coord["y"] >= 1

    def test_iter_spectra_real_file(self) -> None:
        """Test iterating over spectra from real imzML file."""
        sample_imzml = _get_sample_imzml()
        reader = ImzMLReader()
        spectra = list(reader.iter_spectra(sample_imzml))

        # 2x2 image has 4 spectra
        assert len(spectra) == 4

        # Verify each spectrum has valid structure
        for spectrum in spectra:
            assert spectrum["meta"]["source_path"] == str(sample_imzml)
            assert spectrum["meta"]["ms_level"] == 1
            assert spectrum["meta"]["coordinate"]["x"] >= 1
            assert spectrum["meta"]["coordinate"]["y"] >= 1

            # Verify data arrays are present and match length
            mz_len = len(spectrum["data"]["mz_values"])
            int_len = len(spectrum["data"]["intensities"])
            assert mz_len == int_len
            assert spectrum["data"]["mz_values"], "Expected at least one peak"

            # Verify stats match data
            assert spectrum["stats"]["num_peaks"] == mz_len

    def test_read_spectrum_by_index_real_file(self) -> None:
        """Test reading specific spectrum by index from real imzML file."""
        sample_imzml = _get_sample_imzml()
        reader = ImzMLReader()
        spectrum = reader.read_spectrum(sample_imzml, 0)

        assert spectrum["meta"]["index"] == 0
        assert spectrum["meta"]["source_path"] == str(sample_imzml)
        assert spectrum["data"]["mz_values"], "Expected at least one peak"

    def test_read_spectrum_at_coordinate_real_file(self) -> None:
        """Test reading spectrum at specific coordinate from real imzML file."""
        sample_imzml = _get_sample_imzml()
        reader = ImzMLReader()

        # First get the coordinates to find a valid one
        coords = reader.get_coordinates(sample_imzml)
        first_coord = coords[0]

        # Read spectrum at that coordinate
        spectrum = reader.read_spectrum_at_coordinate(
            sample_imzml,
            x=first_coord["x"],
            y=first_coord["y"],
            z=1,  # Default z for 2D imaging
        )

        assert spectrum["meta"]["coordinate"]["x"] == first_coord["x"]
        assert spectrum["meta"]["coordinate"]["y"] == first_coord["y"]

    def test_count_spectra_real_file(self) -> None:
        """Test counting spectra from real imzML file."""
        sample_imzml = _get_sample_imzml()
        reader = ImzMLReader()
        count = reader.count_spectra(sample_imzml)

        assert count == 4  # 2x2 image

    def test_spectrum_data_values_valid(self) -> None:
        """Test that spectrum data values are valid (non-negative)."""
        sample_imzml = _get_sample_imzml()
        reader = ImzMLReader()
        spectra = list(reader.iter_spectra(sample_imzml))

        for spectrum in spectra:
            # All m/z values should be positive
            for mz in spectrum["data"]["mz_values"]:
                assert mz > 0.0

            # All intensities should be non-negative
            for intensity in spectrum["data"]["intensities"]:
                assert intensity >= 0.0

            # TIC should be sum of intensities
            expected_tic = sum(spectrum["data"]["intensities"])
            assert spectrum["meta"]["total_ion_current"] == expected_tic

    def test_spectrum_stats_consistent(self) -> None:
        """Test that spectrum stats are consistent with data."""
        sample_imzml = _get_sample_imzml()
        reader = ImzMLReader()
        spectra = list(reader.iter_spectra(sample_imzml))

        for spectrum in spectra:
            mz_values = spectrum["data"]["mz_values"]
            intensities = spectrum["data"]["intensities"]

            if mz_values:
                # min/max should match actual data
                assert spectrum["stats"]["mz_min"] == min(mz_values)
                assert spectrum["stats"]["mz_max"] == max(mz_values)

                # Base peak should be at max intensity
                max_intensity = max(intensities)
                max_idx = intensities.index(max_intensity)
                assert spectrum["stats"]["base_peak_intensity"] == max_intensity
                assert spectrum["stats"]["base_peak_mz"] == mz_values[max_idx]
