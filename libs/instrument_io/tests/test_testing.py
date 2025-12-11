"""Tests for testing.py module to achieve 100% coverage.

Tests production implementations and fake class methods.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import SMPSReadError, TXTReadError
from instrument_io.testing import (
    FakeDataDirectory,
    FakeDataFile,
    FakeDataFile3D,
    FakeImzMLParser,
    FakeMzMLReader,
    FakePDF,
    FakePDFPage,
    _FakeNdArray,
    _FakeNdArray1D,
    _FakeNdArray2D,
    _FakeNdArray3D,
    _prod_find_monorepo_root,
    _prod_shutil_which,
    _prod_smps_read_lines,
    _prod_txt_detect_encoding,
    _prod_txt_read_lines,
    _prod_txt_read_text,
    reset_hooks,
)
from instrument_io.types.spectrum import MSSpectrum, SpectrumData, SpectrumMeta, SpectrumStats


class TestResetHooks:
    """Tests for reset_hooks function."""

    def test_reset_hooks_restores_defaults(self) -> None:
        """Test that reset_hooks restores production implementations."""
        reset_hooks()
        # Just verify it runs without error


class TestProdShutilWhich:
    """Tests for _prod_shutil_which function."""

    def test_returns_path_for_python(self) -> None:
        """Test that shutil.which finds python."""
        result = _prod_shutil_which("python")
        # Result should be a path containing 'python'
        # Use str() for type narrowing since result can be str | None
        assert "python" in str(result).lower()

    def test_returns_none_for_nonexistent(self) -> None:
        """Test that shutil.which returns None for nonexistent command."""
        result = _prod_shutil_which("nonexistent_command_12345")
        assert result is None


class TestProdFindMonorepoRoot:
    """Tests for _prod_find_monorepo_root function."""

    def test_finds_monorepo_root(self) -> None:
        """Test finding monorepo root from current directory."""
        start = Path(__file__).resolve().parent
        result = _prod_find_monorepo_root(start)
        assert (result / "libs").is_dir()

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        """Test raises RuntimeError when no libs directory found."""
        isolated = tmp_path / "isolated"
        isolated.mkdir()
        with pytest.raises(RuntimeError) as exc_info:
            _prod_find_monorepo_root(isolated)
        assert "libs" in str(exc_info.value)


class TestProdSmpsReadLines:
    """Tests for _prod_smps_read_lines function."""

    def test_reads_utf8_file(self, tmp_path: Path) -> None:
        """Test reading UTF-8 encoded file."""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline2\n", encoding="utf-8")
        result = _prod_smps_read_lines(file)
        assert result == ["line1", "line2"]

    def test_reads_cp1252_file(self, tmp_path: Path) -> None:
        """Test reading CP1252 encoded file (fallback)."""
        file = tmp_path / "test.txt"
        # Write bytes that are valid CP1252 but not UTF-8
        file.write_bytes(b"\xe9data\n")  # é in CP1252
        result = _prod_smps_read_lines(file)
        assert "data" in result[0]

    def test_raises_on_unreadable_file(self, tmp_path: Path) -> None:
        """Test raises SMPSReadError on file that can't be read."""
        file = tmp_path / "nonexistent.txt"
        with pytest.raises(SMPSReadError):
            _prod_smps_read_lines(file)


class TestProdTxtDetectEncoding:
    """Tests for _prod_txt_detect_encoding function."""

    def test_detects_utf8(self, tmp_path: Path) -> None:
        """Test detecting UTF-8 encoding."""
        file = tmp_path / "test.txt"
        file.write_text("hello", encoding="utf-8")
        result = _prod_txt_detect_encoding(file)
        assert result == "utf-8"

    def test_detects_utf16(self, tmp_path: Path) -> None:
        """Test detecting UTF-16 encoding."""
        file = tmp_path / "test.txt"
        file.write_text("hello", encoding="utf-16")
        result = _prod_txt_detect_encoding(file)
        assert "utf-16" in result.lower()

    def test_falls_back_to_latin1(self, tmp_path: Path) -> None:
        """Test fallback to latin-1 for unrecognized encoding."""
        file = tmp_path / "test.txt"
        # Write raw bytes that don't match any preferred encoding BOM
        file.write_bytes(bytes(range(128, 256)))
        result = _prod_txt_detect_encoding(file)
        assert result == "latin-1"


class TestProdTxtReadText:
    """Tests for _prod_txt_read_text function."""

    def test_reads_text(self, tmp_path: Path) -> None:
        """Test reading text file content."""
        file = tmp_path / "test.txt"
        file.write_text("hello world", encoding="utf-8")
        result = _prod_txt_read_text(file, "utf-8")
        assert result == "hello world"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Test raises TXTReadError on missing file."""
        file = tmp_path / "nonexistent.txt"
        with pytest.raises(TXTReadError):
            _prod_txt_read_text(file, "utf-8")


class TestProdTxtReadLines:
    """Tests for _prod_txt_read_lines function."""

    def test_reads_lines(self, tmp_path: Path) -> None:
        """Test reading text file lines."""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline2", encoding="utf-8")
        result = _prod_txt_read_lines(file, "utf-8")
        assert result == ["line1", "line2"]

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Test raises TXTReadError on missing file."""
        file = tmp_path / "nonexistent.txt"
        with pytest.raises(TXTReadError):
            _prod_txt_read_lines(file, "utf-8")


class TestFakeNdArray:
    """Tests for _FakeNdArray class."""

    def test_properties(self) -> None:
        """Test all properties."""
        arr = _FakeNdArray([1.0, 2.0, 3.0])
        assert arr.shape == (3,)
        assert arr.dtype.name == "float64"
        assert arr.ndim == 1
        assert arr.size == 3
        assert len(arr) == 3
        assert arr[0] == 1.0
        assert arr.tolist() == [1.0, 2.0, 3.0]


class TestFakeNdArray2D:
    """Tests for _FakeNdArray2D class."""

    def test_properties(self) -> None:
        """Test all properties."""
        arr = _FakeNdArray2D([[1.0, 2.0], [3.0, 4.0]])
        assert arr.shape == (2, 2)
        assert arr.dtype.name == "float64"
        assert arr.ndim == 2
        assert arr.size == 4
        assert len(arr) == 2
        assert arr[0] == 1.0  # Flattened access
        assert arr[3] == 4.0
        assert arr.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_empty_shape(self) -> None:
        """Test empty array shape."""
        arr = _FakeNdArray2D([])
        assert arr.shape == (0, 0)
        assert arr.size == 0

    def test_getitem_raises_on_empty(self) -> None:
        """Test __getitem__ raises IndexError on empty array."""
        arr = _FakeNdArray2D([])
        with pytest.raises(IndexError):
            _ = arr[0]


class TestFakeNdArray1D:
    """Tests for _FakeNdArray1D class."""

    def test_properties(self) -> None:
        """Test all properties.

        Covers lines 657, 662, 670, 674: _FakeNdArray1D methods.
        """
        arr = _FakeNdArray1D([1.0, 2.0, 3.0])
        assert arr.shape == (3,)
        assert arr.dtype.name == "float64"
        assert len(arr) == 3
        assert arr[0] == 1.0
        assert arr[2] == 3.0
        assert arr.tolist() == [1.0, 2.0, 3.0]


class TestFakeNdArray3D:
    """Tests for _FakeNdArray3D class."""

    def test_properties(self) -> None:
        """Test all properties."""
        arr = _FakeNdArray3D((2, 3, 4))
        assert arr.shape == (2, 3, 4)
        assert arr.dtype.name == "float64"
        assert arr.ndim == 3
        assert arr.size == 24
        assert len(arr) == 2
        assert arr[0] == 0.0
        assert arr.tolist() == []


class TestFakeDataFile:
    """Tests for FakeDataFile class."""

    def test_properties_1d(self) -> None:
        """Test 1D data file properties."""
        df = FakeDataFile([1.0, 2.0], [100.0], [10.0, 20.0], "TIC", "test.dat")
        assert df.detector == "TIC"
        assert df.name == "test.dat"
        assert df.get_info() == "FakeDataFile(test.dat)"
        assert df.xlabels.shape == (2,)
        assert df.ylabels.shape == (1,)
        assert df.data.shape == (2,)

    def test_properties_2d(self) -> None:
        """Test 2D data file properties."""
        df = FakeDataFile([1.0], [100.0], [[10.0, 20.0]], "MS", "test.dat")
        assert df.data.shape == (1, 2)

    def test_empty_data(self) -> None:
        """Test empty data handling."""
        df = FakeDataFile([1.0], [], [], "TIC", "empty.dat")
        assert df.data.shape == (0,)


class TestFakeDataFile3D:
    """Tests for FakeDataFile3D class."""

    def test_properties(self) -> None:
        """Test 3D data file properties."""
        df = FakeDataFile3D([1.0, 2.0], [100.0], (2, 3, 4), "MS", "test.dat")
        assert df.detector == "MS"
        assert df.name == "test.dat"
        assert df.get_info() == "FakeDataFile3D(test.dat)"
        assert df.xlabels.shape == (2,)
        assert df.ylabels.shape == (1,)
        assert df.data.shape == (2, 3, 4)


class TestFakeDataDirectory:
    """Tests for FakeDataDirectory class."""

    def test_properties(self) -> None:
        """Test directory properties."""
        df = FakeDataFile([1.0], [], [10.0], "TIC", "test.dat")
        dd = FakeDataDirectory([df], "/test/path")
        assert dd.directory == "/test/path"
        assert len(dd.datafiles) == 1
        assert dd.get_file("TEST.DAT") == df
        assert dd.get_file("nonexistent") is None
        assert dd.get_detector("TIC") == [df]
        assert dd.get_detector("MS") == []

    def test_multiple_files_same_detector(self) -> None:
        """Test directory with multiple files of same detector.

        Covers line 624->626: when adding second file with same detector.
        """
        df1 = FakeDataFile([1.0], [], [10.0], "MS", "ms1.dat")
        df2 = FakeDataFile([2.0], [], [20.0], "MS", "ms2.dat")
        dd = FakeDataDirectory([df1, df2], "/test/path")
        assert len(dd.get_detector("MS")) == 2
        assert df1 in dd.get_detector("MS")
        assert df2 in dd.get_detector("MS")


class TestFakeImzMLParser:
    """Tests for FakeImzMLParser class."""

    def test_properties(self) -> None:
        """Test parser properties."""
        parser = FakeImzMLParser(
            [(1, 1, 1)],
            [([100.0, 200.0], [1000.0, 2000.0])],
            polarity="positive",
            spectrum_mode="centroid",
        )
        assert parser.coordinates == [(1, 1, 1)]
        assert parser.polarity == "positive"
        assert parser.spectrum_mode == "centroid"

    def test_getspectrum(self) -> None:
        """Test getspectrum method."""
        parser = FakeImzMLParser(
            [(1, 1, 1)],
            [([100.0, 200.0], [1000.0, 2000.0])],
        )
        mz, intensities = parser.getspectrum(0)
        assert mz.tolist() == [100.0, 200.0]
        assert intensities.tolist() == [1000.0, 2000.0]

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        parser = FakeImzMLParser([(1, 1, 1)], [([100.0], [1000.0])])
        with parser as p:
            assert p.coordinates == [(1, 1, 1)]


class TestFakeMzMLReader:
    """Tests for FakeMzMLReader class."""

    def test_iter_spectra(self) -> None:
        """Test iterating spectra."""
        spectrum = MSSpectrum(
            meta=SpectrumMeta(
                source_path="/test",
                scan_number=1,
                retention_time=1.0,
                ms_level=1,
                polarity="positive",
                total_ion_current=1000.0,
            ),
            data=SpectrumData(mz_values=[100.0], intensities=[1000.0]),
            stats=SpectrumStats(
                num_peaks=1,
                mz_min=100.0,
                mz_max=100.0,
                base_peak_mz=100.0,
                base_peak_intensity=1000.0,
            ),
        )
        reader = FakeMzMLReader([spectrum])
        spectra = list(reader.iter_spectra(Path("/test")))
        assert len(spectra) == 1

    def test_read_spectrum(self) -> None:
        """Test reading specific spectrum."""
        spectrum = MSSpectrum(
            meta=SpectrumMeta(
                source_path="/test",
                scan_number=5,
                retention_time=1.0,
                ms_level=1,
                polarity="positive",
                total_ion_current=1000.0,
            ),
            data=SpectrumData(mz_values=[100.0], intensities=[1000.0]),
            stats=SpectrumStats(
                num_peaks=1,
                mz_min=100.0,
                mz_max=100.0,
                base_peak_mz=100.0,
                base_peak_intensity=1000.0,
            ),
        )
        reader = FakeMzMLReader([spectrum])
        result = reader.read_spectrum(Path("/test"), 5)
        assert result["meta"]["scan_number"] == 5

    def test_read_spectrum_not_found(self) -> None:
        """Test reading nonexistent spectrum raises."""
        reader = FakeMzMLReader([])
        with pytest.raises(ValueError) as exc_info:
            reader.read_spectrum(Path("/test"), 1)
        assert "not found" in str(exc_info.value)

    def test_count_spectra(self) -> None:
        """Test counting spectra."""
        spectrum = MSSpectrum(
            meta=SpectrumMeta(
                source_path="/test",
                scan_number=1,
                retention_time=1.0,
                ms_level=1,
                polarity="positive",
                total_ion_current=1000.0,
            ),
            data=SpectrumData(mz_values=[100.0], intensities=[1000.0]),
            stats=SpectrumStats(
                num_peaks=1,
                mz_min=100.0,
                mz_max=100.0,
                base_peak_mz=100.0,
                base_peak_intensity=1000.0,
            ),
        )
        reader = FakeMzMLReader([spectrum, spectrum])
        assert reader.count_spectra(Path("/test")) == 2


class TestFakePDFPage:
    """Tests for FakePDFPage class."""

    def test_properties(self) -> None:
        """Test page properties."""
        page = FakePDFPage(text="hello", tables=[[["a", "b"]]], page_number=1)
        assert page.page_number == 1
        assert page.width == 612.0
        assert page.height == 792.0
        assert page.extract_text() == "hello"
        assert page.extract_tables() == [[["a", "b"]]]

    def test_default_text(self) -> None:
        """Test default text is empty string."""
        page = FakePDFPage()
        assert page.extract_text() == ""


class TestFakePDF:
    """Tests for FakePDF class."""

    def test_properties(self) -> None:
        """Test PDF properties."""
        page = FakePDFPage(text="hello")
        pdf = FakePDF([page])
        assert len(pdf.pages) == 1
        assert pdf.metadata == {}

    def test_close(self) -> None:
        """Test close method."""
        pdf = FakePDF([])
        pdf.close()  # Should not raise

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        page = FakePDFPage()
        pdf = FakePDF([page])
        with pdf as p:
            assert len(p.pages) == 1
