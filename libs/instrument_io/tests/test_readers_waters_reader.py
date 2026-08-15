"""Tests for readers waters: WatersReaderWithFakes."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import WatersReadError
from instrument_io._protocols.rainbow import DataDirectoryProtocol
from instrument_io.readers.waters import (
    WatersReader,
)
from instrument_io.testing import FakeDataDirectory, FakeDataFile, FakeDataFile3D, hooks


class TestWatersReaderWithFakes:
    """Test WatersReader methods with fake data via hooks."""

    def test_read_tic_1d_data(self, tmp_path: Path) -> None:
        """Test reading TIC with 1D data array."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        tic_file = FakeDataFile([1.0, 2.0, 3.0], [], [100.0, 200.0, 300.0], "TIC", "tic_data.dat")
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([tic_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        tic = reader.read_tic(raw_dir)

        assert tic["meta"]["source_path"] == str(raw_dir)
        assert tic["stats"]["num_points"] == 3

    def test_read_tic_2d_data(self, tmp_path: Path) -> None:
        """Test reading TIC with 2D data array (summed)."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        tic_file = FakeDataFile(
            [1.0, 2.0],
            [],
            [[100.0, 200.0], [150.0, 250.0]],
            "TIC",
            "tic_data.dat",
        )
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([tic_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        tic = reader.read_tic(raw_dir)

        assert tic["stats"]["num_points"] == 2
        # Summed: [300.0, 400.0]
        assert tic["data"]["intensities"] == [300.0, 400.0]

    def test_read_tic_3d_data_raises(self, tmp_path: Path) -> None:
        """Test reading TIC fails when data has 3+ dimensions.

        Covers waters.py line 299: Unexpected data shape error.
        """
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        # FakeDataFile3D has 3D shape which triggers the error branch
        tic_file = FakeDataFile3D(
            [1.0, 2.0],
            [],
            (2, 3, 4),  # 3D shape
            "TIC",
            "tic_data.dat",
        )
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([tic_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.read_tic(raw_dir)
        assert "Unexpected data shape" in str(exc_info.value)
        assert "(2, 3, 4)" in str(exc_info.value)

    def test_read_tic_no_data_raises(self, tmp_path: Path) -> None:
        """Test reading TIC when no TIC or MS data available."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        # Only UV data, no TIC or MS
        uv_file = FakeDataFile([1.0], [200.0], [[50.0]], "UV", "uv_data.dat")
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([uv_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.read_tic(raw_dir)
        assert "No TIC or MS data available" in str(exc_info.value)

    def test_compute_tic_from_ms(self, tmp_path: Path) -> None:
        """Test computing TIC from MS data when no TIC file."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = FakeDataFile(
            [1.0, 2.0],
            [100.0, 200.0],
            [[50.0, 150.0], [75.0, 225.0]],
            "MS",
            "ms_data.dat",
        )
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        tic = reader.read_tic(raw_dir)

        assert tic["stats"]["num_points"] == 2
        # Summed across m/z: [200.0, 300.0]
        assert tic["data"]["intensities"] == [200.0, 300.0]
        assert "(computed)" in tic["meta"]["detector"]

    def test_compute_tic_from_ms_1d_raises(self, tmp_path: Path) -> None:
        """Test computing TIC fails when MS data is 1D."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = FakeDataFile([1.0], [100.0], [50.0], "MS", "ms_data.dat")
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.read_tic(raw_dir)
        assert "MS data must be 2D to compute TIC" in str(exc_info.value)

    def test_read_eic(self, tmp_path: Path) -> None:
        """Test reading EIC."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = FakeDataFile(
            [1.0, 2.0],
            [100.0, 200.0, 300.0],
            [[50.0, 150.0, 250.0], [75.0, 175.0, 275.0]],
            "MS",
            "ms_data.dat",
        )
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        eic = reader.read_eic(raw_dir, target_mz=200.0, mz_tolerance=1.0)

        assert eic["params"]["target_mz"] == 200.0
        assert eic["data"]["intensities"] == [150.0, 175.0]

    def test_read_eic_1d_raises(self, tmp_path: Path) -> None:
        """Test reading EIC fails when MS data is 1D."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = FakeDataFile([1.0], [100.0], [50.0], "MS", "ms_data.dat")
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.read_eic(raw_dir, target_mz=100.0, mz_tolerance=1.0)
        assert "MS data must be 2D for EIC" in str(exc_info.value)

    def test_read_uv(self, tmp_path: Path) -> None:
        """Test reading UV data."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        uv_file = FakeDataFile(
            [1.0, 2.0],
            [200.0, 300.0],
            [[50.0, 150.0], [75.0, 175.0]],
            "UV",
            "uv_data.dat",
        )
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([uv_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        uv = reader.read_uv(raw_dir)

        assert uv["wavelengths"] == [200.0, 300.0]
        assert len(uv["retention_times"]) == 2

    def test_read_uv_1d_raises(self, tmp_path: Path) -> None:
        """Test reading UV fails when data is 1D."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        uv_file = FakeDataFile([1.0], [200.0], [50.0], "UV", "uv_data.dat")
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([uv_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.read_uv(raw_dir)
        assert "UV data must be 2D" in str(exc_info.value)

    def test_iter_spectra(self, tmp_path: Path) -> None:
        """Test iterating spectra."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = FakeDataFile(
            [1.0, 2.0],
            [100.0, 200.0],
            [[50.0, 150.0], [0.0, 200.0]],  # Second row has zero at first mz
            "MS",
            "ms_data.dat",
        )
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        spectra = list(reader.iter_spectra(raw_dir))

        assert len(spectra) == 2
        assert spectra[0]["meta"]["scan_number"] == 1
        assert spectra[1]["meta"]["scan_number"] == 2
        # First spectrum has 2 peaks
        assert spectra[0]["stats"]["num_peaks"] == 2
        # Second spectrum has 1 peak (zero filtered out)
        assert spectra[1]["stats"]["num_peaks"] == 1

    def test_iter_spectra_empty_row(self, tmp_path: Path) -> None:
        """Test iterating spectra with all-zero row."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = FakeDataFile(
            [1.0],
            [100.0, 200.0],
            [[0.0, 0.0]],  # All zeros
            "MS",
            "ms_data.dat",
        )
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        spectra = list(reader.iter_spectra(raw_dir))

        assert len(spectra) == 1
        assert spectra[0]["stats"]["num_peaks"] == 0
        assert spectra[0]["stats"]["mz_min"] == 0.0
        assert spectra[0]["stats"]["base_peak_mz"] == 0.0

    def test_iter_spectra_1d_raises(self, tmp_path: Path) -> None:
        """Test iter_spectra fails when MS data is 1D."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = FakeDataFile([1.0], [100.0], [50.0], "MS", "ms_data.dat")
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            list(reader.iter_spectra(raw_dir))
        assert "MS data must be 2D for spectra" in str(exc_info.value)

    def test_iter_spectra_row_length_mismatch(self, tmp_path: Path) -> None:
        """Test iter_spectra fails when row length doesn't match mz axis."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        # Create fake where row length != mz axis length
        ms_file = FakeDataFile(
            [1.0],
            [100.0, 200.0, 300.0],  # 3 m/z values
            [[50.0, 150.0]],  # But only 2 intensities
            "MS",
            "ms_data.dat",
        )
        fake_datadir: DataDirectoryProtocol = FakeDataDirectory([ms_file], str(raw_dir))

        hooks.load_data_directory = lambda p: fake_datadir

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            list(reader.iter_spectra(raw_dir))
        assert "row length" in str(exc_info.value)
        assert "mz axis length" in str(exc_info.value)
