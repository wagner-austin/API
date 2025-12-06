"""Integration tests for WatersReader using real Waters .raw directories.

Tests against multiple Waters fixtures with different detector configurations:
- blue.raw: MS, UV, CAD (full featured)
- indigo.raw: MS only
- violet.raw: MS, UV, ELSD
- white.raw: UV only (no MS)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import WatersReadError
from instrument_io.readers.waters import WatersReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _get_waters_fixture(name: str) -> Path:
    """Get path to Waters fixture, skip if not found."""
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"{name} test fixture not found")
    return path


class TestWatersReaderSupportsFormat:
    """Test supports_format method."""

    def test_supports_format_blue_raw(self) -> None:
        """Test format detection for blue.raw (MS, UV, CAD)."""
        path = _get_waters_fixture("blue.raw")
        reader = WatersReader()
        assert reader.supports_format(path) is True

    def test_supports_format_indigo_raw(self) -> None:
        """Test format detection for indigo.raw (MS only)."""
        path = _get_waters_fixture("indigo.raw")
        reader = WatersReader()
        assert reader.supports_format(path) is True

    def test_supports_format_violet_raw(self) -> None:
        """Test format detection for violet.raw (MS, UV, ELSD)."""
        path = _get_waters_fixture("violet.raw")
        reader = WatersReader()
        assert reader.supports_format(path) is True

    def test_supports_format_white_raw(self) -> None:
        """Test format detection for white.raw (UV only)."""
        path = _get_waters_fixture("white.raw")
        reader = WatersReader()
        assert reader.supports_format(path) is True

    def test_does_not_support_agilent_d(self) -> None:
        """Test that WatersReader does not support Agilent .D directories."""
        path = _get_waters_fixture("sample.D")
        reader = WatersReader()
        assert reader.supports_format(path) is False


class TestWatersReaderReadTIC:
    """Test read_tic method."""

    def test_read_tic_blue_raw(self) -> None:
        """Test reading TIC from blue.raw (has MS)."""
        path = _get_waters_fixture("blue.raw")
        reader = WatersReader()
        tic = reader.read_tic(path)

        assert tic["meta"]["source_path"] == str(path)
        assert tic["stats"]["num_points"] > 0
        assert len(tic["data"]["retention_times"]) == tic["stats"]["num_points"]
        assert len(tic["data"]["intensities"]) == tic["stats"]["num_points"]

    def test_read_tic_indigo_raw(self) -> None:
        """Test reading TIC from indigo.raw (MS only)."""
        path = _get_waters_fixture("indigo.raw")
        reader = WatersReader()
        tic = reader.read_tic(path)

        assert tic["stats"]["num_points"] > 0
        # indigo has 3865 points
        assert tic["stats"]["num_points"] == 3865

    def test_read_tic_violet_raw(self) -> None:
        """Test reading TIC from violet.raw (MS, UV, ELSD)."""
        path = _get_waters_fixture("violet.raw")
        reader = WatersReader()
        tic = reader.read_tic(path)

        assert tic["stats"]["num_points"] > 0

    def test_read_tic_white_raw_fails(self) -> None:
        """Test that read_tic fails for white.raw (UV only, no MS)."""
        path = _get_waters_fixture("white.raw")
        reader = WatersReader()

        with pytest.raises(WatersReadError) as exc_info:
            reader.read_tic(path)
        assert "No TIC or MS data available" in str(exc_info.value)

    def test_tic_data_values_valid(self) -> None:
        """Test that TIC data values are valid."""
        path = _get_waters_fixture("blue.raw")
        reader = WatersReader()
        tic = reader.read_tic(path)

        # Retention times should be increasing
        rts = tic["data"]["retention_times"]
        for i in range(1, len(rts)):
            assert rts[i] >= rts[i - 1], "Retention times should be monotonic"

        # Intensities should be non-negative
        for intensity in tic["data"]["intensities"]:
            assert intensity >= 0.0


class TestWatersReaderIterSpectra:
    """Test iter_spectra method."""

    def test_iter_spectra_blue_raw(self) -> None:
        """Test iterating spectra from blue.raw."""
        path = _get_waters_fixture("blue.raw")
        reader = WatersReader()
        spectra = list(reader.iter_spectra(path))

        assert len(spectra) == 725
        for spectrum in spectra:
            assert spectrum["meta"]["source_path"] == str(path)
            assert spectrum["meta"]["ms_level"] == 1

    def test_iter_spectra_indigo_raw(self) -> None:
        """Test iterating spectra from indigo.raw."""
        path = _get_waters_fixture("indigo.raw")
        reader = WatersReader()
        spectra = list(reader.iter_spectra(path))

        assert len(spectra) == 3865

    def test_iter_spectra_white_raw_fails(self) -> None:
        """Test that iter_spectra fails for white.raw (UV only)."""
        path = _get_waters_fixture("white.raw")
        reader = WatersReader()

        with pytest.raises(WatersReadError) as exc_info:
            list(reader.iter_spectra(path))
        assert "No MS data file found" in str(exc_info.value)

    def test_spectra_have_valid_structure(self) -> None:
        """Test that spectra have valid structure."""
        path = _get_waters_fixture("blue.raw")
        reader = WatersReader()

        # Just check first few spectra
        for count, spectrum in enumerate(reader.iter_spectra(path)):
            assert spectrum["meta"]["scan_number"] >= 1
            assert spectrum["meta"]["retention_time"] >= 0.0

            mz_len = len(spectrum["data"]["mz_values"])
            int_len = len(spectrum["data"]["intensities"])
            assert mz_len == int_len

            assert spectrum["stats"]["num_peaks"] == mz_len

            if count >= 9:
                break


class TestWatersReaderReadUV:
    """Test read_uv method."""

    def test_read_uv_blue_raw(self) -> None:
        """Test reading UV from blue.raw."""
        path = _get_waters_fixture("blue.raw")
        reader = WatersReader()
        uv = reader.read_uv(path)

        assert uv["meta"]["source_path"] == str(path)
        assert uv["retention_times"], "Expected retention times"
        assert uv["wavelengths"], "Expected wavelengths"
        assert len(uv["intensity_matrix"]) == len(uv["retention_times"])

    def test_read_uv_white_raw(self) -> None:
        """Test reading UV from white.raw (UV only file)."""
        path = _get_waters_fixture("white.raw")
        reader = WatersReader()
        uv = reader.read_uv(path)

        # white.raw has 259 timepoints x 3 wavelengths
        assert len(uv["retention_times"]) == 259
        assert len(uv["wavelengths"]) == 3

    def test_read_uv_violet_raw(self) -> None:
        """Test reading UV from violet.raw."""
        path = _get_waters_fixture("violet.raw")
        reader = WatersReader()
        uv = reader.read_uv(path)

        # violet.raw has 24001 timepoints x 190 wavelengths
        assert len(uv["retention_times"]) == 24001
        assert len(uv["wavelengths"]) == 190

    def test_read_uv_indigo_raw_fails(self) -> None:
        """Test that read_uv fails for indigo.raw (MS only, no UV)."""
        path = _get_waters_fixture("indigo.raw")
        reader = WatersReader()

        with pytest.raises(WatersReadError) as exc_info:
            reader.read_uv(path)
        assert "No UV data file found" in str(exc_info.value)


class TestWatersReaderReadEIC:
    """Test read_eic method."""

    def test_read_eic_blue_raw(self) -> None:
        """Test reading EIC from blue.raw."""
        path = _get_waters_fixture("blue.raw")
        reader = WatersReader()

        # Use a common m/z value
        eic = reader.read_eic(path, target_mz=500.0, mz_tolerance=1.0)

        assert eic["meta"]["source_path"] == str(path)
        assert eic["meta"]["signal_type"] == "EIC"
        assert eic["params"]["target_mz"] == 500.0
        assert eic["params"]["mz_tolerance"] == 1.0
        assert eic["data"]["retention_times"], "Expected retention times"

    def test_read_eic_no_match_raises(self) -> None:
        """Test that read_eic raises when no m/z values match."""
        path = _get_waters_fixture("blue.raw")
        reader = WatersReader()

        # Use an extremely high m/z that won't exist
        with pytest.raises(WatersReadError) as exc_info:
            reader.read_eic(path, target_mz=99999.0, mz_tolerance=0.1)
        assert "No m/z values within" in str(exc_info.value)


class TestWatersReaderFindRuns:
    """Test find_runs method."""

    def test_find_runs_in_fixtures(self) -> None:
        """Test finding Waters runs in fixtures directory."""
        reader = WatersReader()
        runs = reader.find_runs(FIXTURES_DIR)

        # Should find at least the 4 Waters fixtures
        raw_runs = [r for r in runs if r["path"].endswith(".raw")]
        assert len(raw_runs) >= 4

        # Verify structure
        for run in raw_runs:
            assert "run_id" in run
            assert "path" in run
            assert "has_ms" in run
            assert "has_dad" in run

    def test_find_runs_not_directory_raises(self) -> None:
        """Test that find_runs raises for non-directory path."""
        reader = WatersReader()

        with pytest.raises(WatersReadError) as exc_info:
            reader.find_runs(Path("/nonexistent/path"))
        assert "Not a directory" in str(exc_info.value)


class TestRainbowProtocolGetFile:
    """Test rainbow protocol get_file method."""

    def test_get_file_by_name(self) -> None:
        """Test getting data file by name from rainbow protocol."""
        from instrument_io._protocols.rainbow import _load_data_directory

        raw_path = _get_waters_fixture("blue.raw")
        raw_dir = _load_data_directory(raw_path)

        # Get MS data file by name (should exist)
        ms_file = raw_dir.get_file("_FUNC001.DAT")
        if ms_file is None:
            raise AssertionError("Expected _FUNC001.DAT to exist")
        assert ms_file.name == "_FUNC001.DAT"

        # Get non-existent file (should return None)
        missing_file = raw_dir.get_file("nonexistent.dat")
        assert missing_file is None
