"""Unit tests for WatersReader with mocked dependencies.

Tests error paths and edge cases using monkeypatch.
All mock classes properly implement the required protocols.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from instrument_io._exceptions import WatersReadError
from instrument_io._protocols.numpy import NdArrayProtocol
from instrument_io._protocols.rainbow import DataDirectoryProtocol, DataFileProtocol
from instrument_io.readers.waters import (
    WatersReader,
    _build_chromatogram_meta,
    _extract_eic_intensities,
    _find_ms_file,
    _find_ms_file_optional,
    _find_tic_file_optional,
    _find_uv_file,
    _is_waters_raw_directory,
)


def _make_loader(datadir: DataDirectoryProtocol) -> Callable[[Path], DataDirectoryProtocol]:
    """Create a typed mock loader function that returns the given datadir."""

    def loader(path: Path) -> DataDirectoryProtocol:
        del path  # unused
        return datadir

    return loader


class MockDType:
    """Mock dtype that satisfies DTypeProtocol."""

    @property
    def name(self) -> str:
        return "float64"


class MockNdArray:
    """Mock ndarray that satisfies NdArrayProtocol.

    Supports both 1D and 2D data representations.
    """

    def __init__(self, data: list[float] | list[list[float]]) -> None:
        self._data = data
        # Detect dimensionality by checking first element
        is_2d = bool(data) and isinstance(data[0], list)
        if is_2d:
            # 2D data: count rows and cols
            rows = len(data)
            first_row = data[0]
            cols = len(first_row) if isinstance(first_row, list) else 0
            self._shape: tuple[int, ...] = (rows, cols)
            self._ndim = 2
            self._size = rows * cols
        else:
            # 1D data
            self._shape = (len(data),)
            self._ndim = 1
            self._size = len(data)
        self._dtype = MockDType()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> MockDType:
        return self._dtype

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def size(self) -> int:
        return self._size

    def tolist(self) -> list[float] | list[list[float]] | list[int] | list[list[int]]:
        return self._data

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, idx: int) -> float:
        item = self._data[idx]
        if isinstance(item, list):
            # 2D case: return first element of row
            return item[0]
        # 1D case: item is already a float
        return item


class MockNdArray3D:
    """Mock 3D ndarray for testing error case.

    Note: This intentionally returns a shape that will trigger
    the "unexpected data shape" error in the reader. The tolist()
    return type matches NdArrayProtocol even though the actual
    shape is 3D - this allows us to test the shape validation.
    """

    def __init__(self) -> None:
        self._dtype = MockDType()

    @property
    def shape(self) -> tuple[int, ...]:
        # Returns 3D shape to trigger error
        return (2, 2, 2)

    @property
    def dtype(self) -> MockDType:
        return self._dtype

    @property
    def ndim(self) -> int:
        return 3

    @property
    def size(self) -> int:
        return 8

    def tolist(self) -> list[float] | list[list[float]] | list[int] | list[list[int]]:
        # Return empty list - the shape check happens before tolist() is called
        return []

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> float:
        return 1.0


class MockDataFile:
    """Mock DataFile that satisfies DataFileProtocol."""

    def __init__(
        self,
        detector: str,
        xlabels_data: list[float],
        ylabels_data: list[float],
        data: list[float] | list[list[float]],
    ) -> None:
        self._detector = detector
        self._xlabels: NdArrayProtocol = MockNdArray(xlabels_data)
        self._ylabels: NdArrayProtocol = MockNdArray(ylabels_data)
        self._data: NdArrayProtocol = MockNdArray(data)
        self._name = f"{detector.lower()}_data.dat"

    @property
    def xlabels(self) -> NdArrayProtocol:
        return self._xlabels

    @property
    def ylabels(self) -> NdArrayProtocol:
        return self._ylabels

    @property
    def data(self) -> NdArrayProtocol:
        return self._data

    @property
    def detector(self) -> str:
        return self._detector

    @property
    def name(self) -> str:
        return self._name

    def get_info(self) -> str:
        return f"MockDataFile({self._detector})"

    def set_data(self, new_data: NdArrayProtocol) -> None:
        """Set data for testing edge cases."""
        self._data = new_data


class MockDataDirectory:
    """Mock DataDirectory that satisfies DataDirectoryProtocol."""

    def __init__(
        self,
        directory: str,
        files: list[MockDataFile],
        by_detector: dict[str, list[MockDataFile]],
    ) -> None:
        self._directory = directory
        # Convert to list[DataFileProtocol] to satisfy type checker (list is invariant)
        self._files: list[DataFileProtocol] = list(files)
        self._by_detector = by_detector
        self._by_name: dict[str, DataFileProtocol] = {f.name: f for f in files}

    @property
    def datafiles(self) -> list[DataFileProtocol]:
        return self._files

    @property
    def directory(self) -> str:
        return self._directory

    def get_file(self, name: str) -> DataFileProtocol | None:
        return self._by_name.get(name.upper())

    def get_detector(self, detector: str) -> list[DataFileProtocol]:
        result = self._by_detector.get(detector.upper(), [])
        # Return as list[DataFileProtocol]
        return list(result)


class TestIsWatersRawDirectory:
    """Test _is_waters_raw_directory helper."""

    def test_valid_raw_directory(self, tmp_path: Path) -> None:
        """Test detection of valid .raw directory."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()
        assert _is_waters_raw_directory(raw_dir) is True

    def test_lowercase_raw_extension(self, tmp_path: Path) -> None:
        """Test lowercase .raw extension."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()
        assert _is_waters_raw_directory(raw_dir) is True

    def test_uppercase_raw_extension(self, tmp_path: Path) -> None:
        """Test uppercase .RAW extension."""
        raw_dir = tmp_path / "test.RAW"
        raw_dir.mkdir()
        assert _is_waters_raw_directory(raw_dir) is True

    def test_not_directory(self, tmp_path: Path) -> None:
        """Test file with .raw extension is rejected."""
        raw_file = tmp_path / "test.raw"
        raw_file.write_text("not a directory")
        assert _is_waters_raw_directory(raw_file) is False

    def test_wrong_extension(self, tmp_path: Path) -> None:
        """Test directory with wrong extension."""
        d_dir = tmp_path / "test.D"
        d_dir.mkdir()
        assert _is_waters_raw_directory(d_dir) is False


class TestFindTicFileOptional:
    """Test _find_tic_file_optional helper."""

    def test_finds_tic_by_detector(self) -> None:
        """Test finding TIC via get_detector."""
        tic_file = MockDataFile("TIC", [1.0], [], [100.0])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [tic_file],
            {"TIC": [tic_file]},
        )
        result = _find_tic_file_optional(datadir)
        assert result is tic_file

    def test_finds_tic_in_datafiles(self) -> None:
        """Test finding TIC by searching datafiles."""
        tic_file = MockDataFile("tic_scan", [1.0], [], [100.0])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [tic_file],
            {},
        )
        result = _find_tic_file_optional(datadir)
        assert result is tic_file

    def test_finds_total_in_datafiles(self) -> None:
        """Test finding TIC via 'total' in detector name."""
        total_file = MockDataFile("Total Ion", [1.0], [], [100.0])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [total_file],
            {},
        )
        result = _find_tic_file_optional(datadir)
        assert result is total_file

    def test_returns_none_when_not_found(self) -> None:
        """Test returns None when no TIC found."""
        ms_file = MockDataFile("MS", [1.0], [], [100.0])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [ms_file],
            {},
        )
        result = _find_tic_file_optional(datadir)
        assert result is None


class TestFindMsFileOptional:
    """Test _find_ms_file_optional helper."""

    def test_finds_ms_by_detector(self) -> None:
        """Test finding MS via get_detector."""
        ms_file = MockDataFile("MS", [1.0], [100.0], [[50.0]])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [ms_file],
            {"MS": [ms_file]},
        )
        result = _find_ms_file_optional(datadir)
        assert result is ms_file

    def test_finds_ms_in_datafiles(self) -> None:
        """Test finding MS by searching datafiles."""
        # Use "ms_scan" which contains "ms" substring (unlike "mass_spec" which doesn't)
        ms_file = MockDataFile("ms_scan", [1.0], [100.0], [[50.0]])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [ms_file],
            {},
        )
        result = _find_ms_file_optional(datadir)
        assert result is ms_file

    def test_returns_none_when_not_found(self) -> None:
        """Test returns None when no MS found."""
        uv_file = MockDataFile("UV", [1.0], [], [100.0])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [uv_file],
            {},
        )
        result = _find_ms_file_optional(datadir)
        assert result is None


class TestFindMsFile:
    """Test _find_ms_file helper."""

    def test_raises_when_not_found(self) -> None:
        """Test raises WatersReadError when MS not found."""
        uv_file = MockDataFile("UV", [1.0], [], [100.0])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [uv_file],
            {},
        )
        with pytest.raises(WatersReadError) as exc_info:
            _find_ms_file(datadir, "/test")
        assert "No MS data file found" in str(exc_info.value)


class TestFindUvFile:
    """Test _find_uv_file helper."""

    def test_finds_uv_by_detector(self) -> None:
        """Test finding UV via get_detector."""
        uv_file = MockDataFile("UV", [1.0], [200.0], [[50.0]])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [uv_file],
            {"UV": [uv_file]},
        )
        result = _find_uv_file(datadir, "/test")
        assert result is uv_file

    def test_finds_pda_in_datafiles(self) -> None:
        """Test finding UV via 'pda' in detector name."""
        pda_file = MockDataFile("pda_scan", [1.0], [200.0], [[50.0]])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [pda_file],
            {},
        )
        result = _find_uv_file(datadir, "/test")
        assert result is pda_file

    def test_raises_when_not_found(self) -> None:
        """Test raises WatersReadError when UV not found."""
        ms_file = MockDataFile("MS", [1.0], [100.0], [[50.0]])
        datadir: DataDirectoryProtocol = MockDataDirectory(
            "/test",
            [ms_file],
            {},
        )
        with pytest.raises(WatersReadError) as exc_info:
            _find_uv_file(datadir, "/test")
        assert "No UV data file found" in str(exc_info.value)


class TestBuildChromatogramMeta:
    """Test _build_chromatogram_meta helper."""

    def test_builds_meta_correctly(self) -> None:
        """Test building ChromatogramMeta."""
        meta = _build_chromatogram_meta("/path/to/file", "TIC", "MS")
        assert meta["source_path"] == "/path/to/file"
        assert meta["signal_type"] == "TIC"
        assert meta["detector"] == "MS"
        assert meta["instrument"] == ""
        assert meta["method_name"] == ""


class TestExtractEicIntensities:
    """Test _extract_eic_intensities helper."""

    def test_extracts_single_mz(self) -> None:
        """Test extracting EIC for single m/z match."""
        ms_data = [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]]
        mz_axis = [100.0, 200.0, 300.0]
        result = _extract_eic_intensities(ms_data, mz_axis, 200.0, 0.5, "/test")
        assert result == [200.0, 250.0]

    def test_extracts_multiple_mz(self) -> None:
        """Test extracting EIC summing multiple m/z channels."""
        ms_data = [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]]
        mz_axis = [199.0, 200.0, 201.0]
        result = _extract_eic_intensities(ms_data, mz_axis, 200.0, 1.5, "/test")
        # Should sum all three channels
        assert result == [600.0, 750.0]

    def test_raises_on_empty_data(self) -> None:
        """Test raises on empty MS data."""
        with pytest.raises(WatersReadError) as exc_info:
            _extract_eic_intensities([], [100.0], 100.0, 1.0, "/test")
        assert "Empty MS data" in str(exc_info.value)

    def test_raises_on_empty_mz_axis(self) -> None:
        """Test raises on empty m/z axis."""
        with pytest.raises(WatersReadError) as exc_info:
            _extract_eic_intensities([[100.0]], [], 100.0, 1.0, "/test")
        assert "Empty MS data" in str(exc_info.value)

    def test_raises_on_no_match(self) -> None:
        """Test raises when no m/z values match."""
        ms_data = [[100.0, 200.0]]
        mz_axis = [100.0, 200.0]
        with pytest.raises(WatersReadError) as exc_info:
            _extract_eic_intensities(ms_data, mz_axis, 500.0, 0.1, "/test")
        assert "No m/z values within" in str(exc_info.value)


class TestWatersReaderSupportsFormat:
    """Test WatersReader.supports_format method."""

    def test_supports_raw_directory(self, tmp_path: Path) -> None:
        """Test supports .raw directory."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()
        reader = WatersReader()
        assert reader.supports_format(raw_dir) is True

    def test_rejects_d_directory(self, tmp_path: Path) -> None:
        """Test rejects .D directory."""
        d_dir = tmp_path / "test.D"
        d_dir.mkdir()
        reader = WatersReader()
        assert reader.supports_format(d_dir) is False


class TestWatersReaderWithMocks:
    """Test WatersReader methods with mocked data."""

    def test_read_tic_1d_data(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test reading TIC with 1D data array."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        tic_file = MockDataFile("TIC", [1.0, 2.0, 3.0], [], [100.0, 200.0, 300.0])
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [tic_file],
            {"TIC": [tic_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        tic = reader.read_tic(raw_dir)

        assert tic["meta"]["source_path"] == str(raw_dir)
        assert tic["stats"]["num_points"] == 3

    def test_read_tic_2d_data(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test reading TIC with 2D data array (summed)."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        tic_file = MockDataFile(
            "TIC",
            [1.0, 2.0],
            [],
            [[100.0, 200.0], [150.0, 250.0]],
        )
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [tic_file],
            {"TIC": [tic_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        tic = reader.read_tic(raw_dir)

        assert tic["stats"]["num_points"] == 2
        # Summed: [300.0, 400.0]
        assert tic["data"]["intensities"] == [300.0, 400.0]

    def test_read_tic_3d_data_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test reading TIC with unexpected data shape raises."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        tic_file = MockDataFile("TIC", [1.0], [], [100.0])
        # Replace data with 3D array
        tic_file.set_data(MockNdArray3D())

        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [tic_file],
            {"TIC": [tic_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.read_tic(raw_dir)
        assert "Unexpected data shape" in str(exc_info.value)

    def test_compute_tic_from_ms(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test computing TIC from MS data when no TIC file."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = MockDataFile(
            "MS",
            [1.0, 2.0],
            [100.0, 200.0],
            [[50.0, 150.0], [75.0, 225.0]],
        )
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [ms_file],
            {"MS": [ms_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        tic = reader.read_tic(raw_dir)

        assert tic["stats"]["num_points"] == 2
        # Summed across m/z: [200.0, 300.0]
        assert tic["data"]["intensities"] == [200.0, 300.0]
        assert "(computed)" in tic["meta"]["detector"]

    def test_compute_tic_from_ms_1d_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test computing TIC fails when MS data is 1D."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = MockDataFile("MS", [1.0], [100.0], [50.0])
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [ms_file],
            {"MS": [ms_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.read_tic(raw_dir)
        assert "MS data must be 2D to compute TIC" in str(exc_info.value)

    def test_read_eic(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test reading EIC."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = MockDataFile(
            "MS",
            [1.0, 2.0],
            [100.0, 200.0, 300.0],
            [[50.0, 150.0, 250.0], [75.0, 175.0, 275.0]],
        )
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [ms_file],
            {"MS": [ms_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        eic = reader.read_eic(raw_dir, target_mz=200.0, mz_tolerance=1.0)

        assert eic["params"]["target_mz"] == 200.0
        assert eic["data"]["intensities"] == [150.0, 175.0]

    def test_read_eic_1d_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test reading EIC fails when MS data is 1D."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = MockDataFile("MS", [1.0], [100.0], [50.0])
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [ms_file],
            {"MS": [ms_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.read_eic(raw_dir, target_mz=100.0, mz_tolerance=1.0)
        assert "MS data must be 2D for EIC" in str(exc_info.value)

    def test_read_uv(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test reading UV data."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        uv_file = MockDataFile(
            "UV",
            [1.0, 2.0],
            [200.0, 300.0],
            [[50.0, 150.0], [75.0, 175.0]],
        )
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [uv_file],
            {"UV": [uv_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        uv = reader.read_uv(raw_dir)

        assert uv["wavelengths"] == [200.0, 300.0]
        assert len(uv["retention_times"]) == 2

    def test_read_uv_1d_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test reading UV fails when data is 1D."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        uv_file = MockDataFile("UV", [1.0], [200.0], [50.0])
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [uv_file],
            {"UV": [uv_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.read_uv(raw_dir)
        assert "UV data must be 2D" in str(exc_info.value)

    def test_iter_spectra(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test iterating spectra."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = MockDataFile(
            "MS",
            [1.0, 2.0],
            [100.0, 200.0],
            [[50.0, 150.0], [0.0, 200.0]],  # Second row has zero at first mz
        )
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [ms_file],
            {"MS": [ms_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        spectra = list(reader.iter_spectra(raw_dir))

        assert len(spectra) == 2
        assert spectra[0]["meta"]["scan_number"] == 1
        assert spectra[1]["meta"]["scan_number"] == 2
        # First spectrum has 2 peaks
        assert spectra[0]["stats"]["num_peaks"] == 2
        # Second spectrum has 1 peak (zero filtered out)
        assert spectra[1]["stats"]["num_peaks"] == 1

    def test_iter_spectra_empty_row(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test iterating spectra with all-zero row."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = MockDataFile(
            "MS",
            [1.0],
            [100.0, 200.0],
            [[0.0, 0.0]],  # All zeros
        )
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [ms_file],
            {"MS": [ms_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        spectra = list(reader.iter_spectra(raw_dir))

        assert len(spectra) == 1
        assert spectra[0]["stats"]["num_peaks"] == 0
        assert spectra[0]["stats"]["mz_min"] == 0.0
        assert spectra[0]["stats"]["base_peak_mz"] == 0.0

    def test_iter_spectra_1d_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test iter_spectra fails when MS data is 1D."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        ms_file = MockDataFile("MS", [1.0], [100.0], [50.0])
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [ms_file],
            {"MS": [ms_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            list(reader.iter_spectra(raw_dir))
        assert "MS data must be 2D for spectra" in str(exc_info.value)

    def test_iter_spectra_row_length_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test iter_spectra fails when row length doesn't match mz axis."""
        raw_dir = tmp_path / "test.raw"
        raw_dir.mkdir()

        # Create mock where row length != mz axis length
        ms_file = MockDataFile(
            "MS",
            [1.0],
            [100.0, 200.0, 300.0],  # 3 m/z values
            [[50.0, 150.0]],  # But only 2 intensities
        )
        mock_datadir: DataDirectoryProtocol = MockDataDirectory(
            str(raw_dir),
            [ms_file],
            {"MS": [ms_file]},
        )

        monkeypatch.setattr(
            "instrument_io.readers.waters._load_data_directory",
            _make_loader(mock_datadir),
        )

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            list(reader.iter_spectra(raw_dir))
        assert "row length" in str(exc_info.value)
        assert "mz axis length" in str(exc_info.value)


class TestWatersReaderFindRuns:
    """Test WatersReader.find_runs method."""

    def test_find_runs_not_directory_raises(self, tmp_path: Path) -> None:
        """Test find_runs raises when path is not a directory."""
        not_dir = tmp_path / "not_a_directory.raw"
        not_dir.write_text("not a directory")

        reader = WatersReader()
        with pytest.raises(WatersReadError) as exc_info:
            reader.find_runs(not_dir)
        assert "Not a directory" in str(exc_info.value)

    def test_find_runs_empty_directory(self, tmp_path: Path) -> None:
        """Test find_runs returns empty list for empty directory."""
        reader = WatersReader()
        runs = reader.find_runs(tmp_path)
        assert runs == []

    def test_find_runs_with_raw_directories(self, tmp_path: Path) -> None:
        """Test find_runs finds .raw directories."""
        # Create .raw directories
        raw1 = tmp_path / "sample1.raw"
        raw1.mkdir()
        raw2 = tmp_path / "sample2.raw"
        raw2.mkdir()

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert len(runs) == 2
        run_ids = [r["run_id"] for r in runs]
        assert "sample1" in run_ids
        assert "sample2" in run_ids

    def test_find_runs_with_data_files(self, tmp_path: Path) -> None:
        """Test find_runs detects TIC, MS, and DAD files."""
        raw_dir = tmp_path / "sample.raw"
        raw_dir.mkdir()

        # Create files that indicate different data types
        (raw_dir / "_FUNC001.DAT").write_text("tic data")  # TIC
        (raw_dir / "ms_data.idx").write_text("ms data")  # MS
        (raw_dir / "PDA_spectrum.dat").write_text("pda data")  # DAD

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert len(runs) == 1
        run = runs[0]
        assert run["has_tic"] is True
        assert run["has_ms"] is True
        assert run["has_dad"] is True
        assert run["file_count"] == 3

    def test_find_runs_detects_uv_as_dad(self, tmp_path: Path) -> None:
        """Test find_runs detects UV files as DAD."""
        raw_dir = tmp_path / "sample.raw"
        raw_dir.mkdir()

        (raw_dir / "uv_data.dat").write_text("uv data")

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert len(runs) == 1
        assert runs[0]["has_dad"] is True

    def test_find_runs_skips_raw_file(self, tmp_path: Path) -> None:
        """Test find_runs skips .raw files (not directories)."""
        # Create a .raw file (not directory)
        raw_file = tmp_path / "sample.raw"
        raw_file.write_text("not a directory")

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert runs == []

    def test_find_runs_extracts_site_from_parent(self, tmp_path: Path) -> None:
        """Test find_runs extracts site from parent directory name."""
        site_dir = tmp_path / "site_A"
        site_dir.mkdir()
        raw_dir = site_dir / "sample.raw"
        raw_dir.mkdir()

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert len(runs) == 1
        assert runs[0]["site"] == "site_A"

    def test_find_runs_ignores_subdirectories(self, tmp_path: Path) -> None:
        """Test find_runs only counts files, not subdirectories."""
        raw_dir = tmp_path / "sample.raw"
        raw_dir.mkdir()

        # Create a file and a subdirectory
        (raw_dir / "data.dat").write_text("data")
        (raw_dir / "subdir").mkdir()

        reader = WatersReader()
        runs = reader.find_runs(tmp_path)

        assert len(runs) == 1
        # Only the file should be counted, not the subdirectory
        assert runs[0]["file_count"] == 1
