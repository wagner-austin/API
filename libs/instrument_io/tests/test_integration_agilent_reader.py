"""Integration tests for AgilentReader using real Agilent .D directories.

Tests against multiple Agilent fixtures with different detector configurations:
- sample.D: MS only (original fixture)
- brown.D: UV only
- green.D: UV, MS
- orange.D: ELSD, MS
- pink.D: UV, FID
- red.D: CAD, UV
- yellow.D: MS, FID
"""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import AgilentReadError
from instrument_io.readers.agilent import AgilentReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _get_agilent_fixture(name: str) -> Path:
    """Get path to Agilent fixture, skip if not found."""
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"{name} test fixture not found")
    return path


class TestAgilentReaderSupportsFormat:
    """Test supports_format method."""

    def test_supports_format_sample_d(self) -> None:
        """Test format detection for sample.D."""
        path = _get_agilent_fixture("sample.D")
        reader = AgilentReader()
        assert reader.supports_format(path) is True

    def test_supports_format_all_fixtures(self) -> None:
        """Test format detection for all Agilent fixtures."""
        reader = AgilentReader()
        fixtures = ["sample.D", "brown.D", "green.D", "orange.D", "pink.D", "red.D", "yellow.D"]
        for name in fixtures:
            path = _get_agilent_fixture(name)
            assert reader.supports_format(path) is True, f"{name} should be supported"

    def test_does_not_support_waters_raw(self) -> None:
        """Test that AgilentReader does not support Waters .raw directories."""
        path = _get_agilent_fixture("blue.raw")
        reader = AgilentReader()
        assert reader.supports_format(path) is False


class TestAgilentReaderReadTIC:
    """Test read_tic method across different fixtures."""

    def test_read_tic_sample_d(self) -> None:
        """Test reading TIC from sample.D (MS only)."""
        path = _get_agilent_fixture("sample.D")
        reader = AgilentReader()
        tic = reader.read_tic(path)

        assert tic["meta"]["source_path"] == str(path)
        assert tic["stats"]["num_points"] > 0

    def test_read_tic_green_d(self) -> None:
        """Test reading TIC from green.D (UV, MS)."""
        path = _get_agilent_fixture("green.D")
        reader = AgilentReader()
        tic = reader.read_tic(path)

        assert tic["stats"]["num_points"] > 0

    def test_read_tic_orange_d(self) -> None:
        """Test reading TIC from orange.D (ELSD, MS)."""
        path = _get_agilent_fixture("orange.D")
        reader = AgilentReader()
        tic = reader.read_tic(path)

        assert tic["stats"]["num_points"] > 0

    def test_read_tic_yellow_d(self) -> None:
        """Test reading TIC from yellow.D (MS, FID)."""
        path = _get_agilent_fixture("yellow.D")
        reader = AgilentReader()
        tic = reader.read_tic(path)

        assert tic["stats"]["num_points"] > 0

    def test_read_tic_brown_d_fails(self) -> None:
        """Test that read_tic fails for brown.D (UV only, no MS)."""
        path = _get_agilent_fixture("brown.D")
        reader = AgilentReader()

        with pytest.raises(AgilentReadError) as exc_info:
            reader.read_tic(path)
        assert "No MS data available" in str(exc_info.value)

    def test_tic_data_consistency(self) -> None:
        """Test that TIC data is internally consistent."""
        path = _get_agilent_fixture("sample.D")
        reader = AgilentReader()
        tic = reader.read_tic(path)

        # Arrays should match in length
        assert len(tic["data"]["retention_times"]) == len(tic["data"]["intensities"])
        assert len(tic["data"]["retention_times"]) == tic["stats"]["num_points"]

        # Retention times should be monotonic
        rts = tic["data"]["retention_times"]
        for i in range(1, len(rts)):
            assert rts[i] >= rts[i - 1]


class TestAgilentReaderIterSpectra:
    """Test iter_spectra method."""

    def test_iter_spectra_sample_d(self) -> None:
        """Test iterating spectra from sample.D."""
        path = _get_agilent_fixture("sample.D")
        reader = AgilentReader()
        spectra = list(reader.iter_spectra(path))

        assert spectra, "Expected at least one spectrum"
        for spectrum in spectra:
            assert spectrum["meta"]["source_path"] == str(path)
            assert spectrum["meta"]["ms_level"] == 1

    def test_iter_spectra_green_d(self) -> None:
        """Test iterating spectra from green.D (UV, MS)."""
        path = _get_agilent_fixture("green.D")
        reader = AgilentReader()
        spectra = list(reader.iter_spectra(path))

        assert spectra, "Expected at least one spectrum"

    def test_iter_spectra_brown_d_fails(self) -> None:
        """Test that iter_spectra fails for brown.D (UV only)."""
        path = _get_agilent_fixture("brown.D")
        reader = AgilentReader()

        with pytest.raises(AgilentReadError) as exc_info:
            list(reader.iter_spectra(path))
        assert "No MS data file found" in str(exc_info.value)

    def test_spectra_structure_valid(self) -> None:
        """Test that spectra have valid structure."""
        path = _get_agilent_fixture("sample.D")
        reader = AgilentReader()

        for count, spectrum in enumerate(reader.iter_spectra(path)):
            assert spectrum["meta"]["scan_number"] >= 1
            assert spectrum["meta"]["retention_time"] >= 0.0
            assert len(spectrum["data"]["mz_values"]) == len(spectrum["data"]["intensities"])
            assert spectrum["stats"]["num_peaks"] == len(spectrum["data"]["mz_values"])

            if count >= 9:
                break


class TestAgilentReaderReadDAD:
    """Test read_dad method.

    Note: rainbow-api returns UV detector data as "UV", not "DAD".
    The read_dad method always raises since DAD detector type is not produced.
    """

    def test_read_dad_always_raises(self) -> None:
        """Test that read_dad always raises (rainbow-api uses UV, not DAD)."""
        path = _get_agilent_fixture("sample.D")
        reader = AgilentReader()

        with pytest.raises(AgilentReadError) as exc_info:
            reader.read_dad(path)
        assert "DAD data not available" in str(exc_info.value)


class TestAgilentReaderReadEIC:
    """Test read_eic method."""

    def test_read_eic_sample_d(self) -> None:
        """Test reading EIC from sample.D."""
        path = _get_agilent_fixture("sample.D")
        reader = AgilentReader()

        eic = reader.read_eic(path, target_mz=100.0, mz_tolerance=1.0)

        assert eic["meta"]["source_path"] == str(path)
        assert eic["meta"]["signal_type"] == "EIC"
        assert eic["params"]["target_mz"] == 100.0
        assert eic["params"]["mz_tolerance"] == 1.0

    def test_read_eic_no_match_raises(self) -> None:
        """Test that read_eic raises when no m/z values match."""
        path = _get_agilent_fixture("sample.D")
        reader = AgilentReader()

        with pytest.raises(AgilentReadError) as exc_info:
            reader.read_eic(path, target_mz=99999.0, mz_tolerance=0.1)
        assert "No m/z values within" in str(exc_info.value)


class TestAgilentReaderFindRuns:
    """Test find_runs method."""

    def test_find_runs_in_fixtures(self) -> None:
        """Test finding Agilent runs in fixtures directory."""
        reader = AgilentReader()
        runs = reader.find_runs(FIXTURES_DIR)

        # Should find at least the 7 Agilent fixtures
        d_runs = [r for r in runs if r["path"].endswith(".D")]
        assert len(d_runs) >= 7

    def test_find_runs_not_directory_raises(self) -> None:
        """Test that find_runs raises for non-directory path."""
        reader = AgilentReader()

        with pytest.raises(AgilentReadError) as exc_info:
            reader.find_runs(Path("/nonexistent/path"))
        assert "Not a directory" in str(exc_info.value)
