"""Shared test doubles for instrument_io readers and writers.

The DI hooks themselves live in :mod:`instrument_io.testing`.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import TracebackType

from instrument_io._protocols.imzml import ImzMLParserProtocol
from instrument_io._protocols.numpy import NdArray1DProtocol, NdArrayProtocol
from instrument_io._protocols.pdfplumber import PageProtocol, PDFProtocol
from instrument_io._protocols.rainbow import DataFileProtocol
from instrument_io.types.spectrum import MSSpectrum


class FakeDataFile:
    """Fake DataFile for Waters tests."""

    _xlabels: _FakeNdArray
    _ylabels: _FakeNdArray
    _data: _FakeNdArray | _FakeNdArray2D

    def __init__(
        self,
        xlabels: list[float],
        ylabels: list[float],
        data: list[list[float]] | list[float],
        detector: str,
        name: str,
    ) -> None:
        self._xlabels = _FakeNdArray(xlabels)
        self._ylabels = _FakeNdArray(ylabels)
        # Check if this is 2D data (list of lists)
        if data and len(data) > 0:
            first_element = data[0]
            if isinstance(first_element, list):
                # 2D data - already typed as list[list[float]]
                data_2d = [row for row in data if isinstance(row, list)]
                self._data = _FakeNdArray2D(data_2d)
            else:
                # 1D data - convert all numeric values
                data_1d = [float(val) for val in data if isinstance(val, (int, float))]
                self._data = _FakeNdArray(data_1d)
        else:
            self._data = _FakeNdArray([])
        self._detector = detector
        self._name = name

    @property
    def xlabels(self) -> NdArrayProtocol:
        """Return retention time array."""
        return self._xlabels

    @property
    def ylabels(self) -> NdArrayProtocol:
        """Return wavelength/m/z array."""
        return self._ylabels

    @property
    def data(self) -> NdArrayProtocol:
        """Return intensity data array."""
        return self._data

    @property
    def detector(self) -> str:
        """Return detector name."""
        return self._detector

    @property
    def name(self) -> str:
        """Return filename."""
        return self._name

    def get_info(self) -> str:
        """Return human-readable info string."""
        return f"FakeDataFile({self._name})"


class _FakeDType:
    """Fake dtype for tests."""

    @property
    def name(self) -> str:
        """Return dtype name."""
        return "float64"


class _FakeNdArray:
    """Fake 1D ndarray for tests."""

    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._dtype = _FakeDType()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape tuple."""
        return (len(self._values),)

    @property
    def dtype(self) -> _FakeDType:
        """Return dtype."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return 1

    @property
    def size(self) -> int:
        """Return total number of elements."""
        return len(self._values)

    def tolist(self) -> list[float] | list[list[float]] | list[int] | list[list[int]]:
        """Return values as list."""
        return self._values

    def __len__(self) -> int:
        """Return length."""
        return len(self._values)

    def __getitem__(self, idx: int) -> float:
        """Get element at index."""
        return self._values[idx]


class _FakeNdArray2D:
    """Fake 2D ndarray for tests."""

    def __init__(self, values: list[list[float]]) -> None:
        self._values = values
        self._dtype = _FakeDType()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape tuple."""
        if not self._values:
            return (0, 0)
        return (len(self._values), len(self._values[0]))

    @property
    def dtype(self) -> _FakeDType:
        """Return dtype."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return 2

    @property
    def size(self) -> int:
        """Return total number of elements."""
        if not self._values:
            return 0
        return len(self._values) * len(self._values[0])

    def tolist(self) -> list[float] | list[list[float]] | list[int] | list[list[int]]:
        """Return values as list."""
        return self._values

    def __len__(self) -> int:
        """Return length of first dimension."""
        return len(self._values)

    def __getitem__(self, idx: int) -> float:
        """Get element at index (flattened)."""
        if not self._values:
            raise IndexError("index out of range")
        cols = len(self._values[0])
        row = idx // cols
        col = idx % cols
        return self._values[row][col]


class _FakeNdArray3D:
    """Fake 3D ndarray for tests (to test error branches)."""

    def __init__(self, shape: tuple[int, int, int]) -> None:
        self._shape = shape
        self._dtype = _FakeDType()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape tuple."""
        return self._shape

    @property
    def dtype(self) -> _FakeDType:
        """Return dtype."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return 3

    @property
    def size(self) -> int:
        """Return total number of elements."""
        return self._shape[0] * self._shape[1] * self._shape[2]

    def tolist(self) -> list[float] | list[list[float]] | list[int] | list[list[int]]:
        """Return values as list."""
        return []

    def __len__(self) -> int:
        """Return length of first dimension."""
        return self._shape[0]

    def __getitem__(self, idx: int) -> float:
        """Get element at index."""
        return 0.0


class FakeDataFile3D:
    """Fake DataFile with 3D data for testing error branches."""

    _xlabels: _FakeNdArray
    _ylabels: _FakeNdArray
    _data: _FakeNdArray3D

    def __init__(
        self,
        xlabels: list[float],
        ylabels: list[float],
        data_shape: tuple[int, int, int],
        detector: str,
        name: str,
    ) -> None:
        self._xlabels = _FakeNdArray(xlabels)
        self._ylabels = _FakeNdArray(ylabels)
        self._data = _FakeNdArray3D(data_shape)
        self._detector = detector
        self._name = name

    @property
    def xlabels(self) -> NdArrayProtocol:
        """Return retention time array."""
        return self._xlabels

    @property
    def ylabels(self) -> NdArrayProtocol:
        """Return wavelength/m/z array."""
        return self._ylabels

    @property
    def data(self) -> NdArrayProtocol:
        """Return intensity data array."""
        return self._data

    @property
    def detector(self) -> str:
        """Return detector name."""
        return self._detector

    @property
    def name(self) -> str:
        """Return filename."""
        return self._name

    def get_info(self) -> str:
        """Return human-readable info string."""
        return f"FakeDataFile3D({self._name})"


class FakeDataDirectory:
    """Fake DataDirectory for Waters tests."""

    _datafiles: list[DataFileProtocol]
    _by_name: dict[str, DataFileProtocol]
    _by_detector: dict[str, list[DataFileProtocol]]

    def __init__(
        self,
        datafiles: list[FakeDataFile] | list[FakeDataFile3D] | list[DataFileProtocol],
        directory: str,
    ) -> None:
        # Store as protocol types for covariance
        files_list: list[DataFileProtocol] = list(datafiles)
        self._datafiles = files_list
        self._directory = directory
        self._by_name = {}
        self._by_detector = {}
        for df in datafiles:
            self._by_name[df.name.upper()] = df
            det_upper = df.detector.upper()
            if det_upper not in self._by_detector:
                self._by_detector[det_upper] = []
            self._by_detector[det_upper].append(df)

    @property
    def datafiles(self) -> list[DataFileProtocol]:
        """Return list of data files."""
        return self._datafiles

    @property
    def directory(self) -> str:
        """Return directory path."""
        return self._directory

    def get_file(self, name: str) -> DataFileProtocol | None:
        """Get data file by name."""
        return self._by_name.get(name.upper())

    def get_detector(self, detector: str) -> list[DataFileProtocol]:
        """Get files for detector."""
        return self._by_detector.get(detector.upper(), [])


class _FakeNdArray1D:
    """Fake 1D ndarray that matches NdArray1DProtocol exactly."""

    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._dtype = _FakeDType()

    @property
    def shape(self) -> tuple[int]:
        """Return shape tuple."""
        return (len(self._values),)

    @property
    def dtype(self) -> _FakeDType:
        """Return dtype."""
        return self._dtype

    def tolist(self) -> list[float]:
        """Return values as list."""
        return self._values

    def __len__(self) -> int:
        """Return length."""
        return len(self._values)

    def __getitem__(self, idx: int) -> float:
        """Get element at index."""
        return self._values[idx]


class FakeImzMLParser:
    """Fake ImzML parser for tests."""

    def __init__(
        self,
        coordinates: list[tuple[int, int, int]],
        spectra: list[tuple[list[float], list[float]]],
        polarity: str = "positive",
        spectrum_mode: str = "centroid",
    ) -> None:
        self._coordinates = coordinates
        self._spectra = spectra
        self._polarity = polarity
        self._spectrum_mode = spectrum_mode

    @property
    def coordinates(self) -> list[tuple[int, int, int]]:
        """Return list of coordinates."""
        return self._coordinates

    @property
    def polarity(self) -> str:
        """Return polarity string."""
        return self._polarity

    @property
    def spectrum_mode(self) -> str:
        """Return spectrum mode."""
        return self._spectrum_mode

    def getspectrum(self, index: int) -> tuple[NdArray1DProtocol, NdArray1DProtocol]:
        """Get spectrum at index."""
        mz, intensities = self._spectra[index]
        return _FakeNdArray1D(mz), _FakeNdArray1D(intensities)

    def __enter__(self) -> ImzMLParserProtocol:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        pass


class FakeMzMLReader:
    """Fake MzMLReader for Thermo tests."""

    def __init__(
        self,
        spectra: list[MSSpectrum],
    ) -> None:
        self._spectra = spectra

    def iter_spectra(self, path: Path) -> Generator[MSSpectrum, None, None]:
        """Iterate over spectra."""
        yield from self._spectra

    def read_spectrum(self, path: Path, scan_number: int) -> MSSpectrum:
        """Read spectrum by scan number."""
        matching = [s for s in self._spectra if s["meta"]["scan_number"] == scan_number]
        if matching:
            return matching[0]
        raise ValueError(f"Spectrum {scan_number} not found")

    def count_spectra(self, path: Path) -> int:
        """Count spectra."""
        return len(self._spectra)


class FakePDFPage:
    """Fake PDF page for tests."""

    def __init__(
        self,
        text: str | None = None,
        tables: list[list[list[str | None]]] | None = None,
        page_number: int = 1,
        width: float = 612.0,
        height: float = 792.0,
    ) -> None:
        self._text = text
        self._tables = tables if tables is not None else []
        self._page_number = page_number
        self._width = width
        self._height = height

    @property
    def page_number(self) -> int:
        """Return 1-based page number."""
        return self._page_number

    @property
    def width(self) -> float:
        """Return page width in points."""
        return self._width

    @property
    def height(self) -> float:
        """Return page height in points."""
        return self._height

    def extract_text(self) -> str:
        """Extract text from page."""
        return self._text if self._text is not None else ""

    def extract_tables(self) -> list[list[list[str | None]]]:
        """Extract tables from page."""
        return self._tables


class FakePDF:
    """Fake PDF for tests."""

    _pages: list[PageProtocol]

    def __init__(
        self,
        pages_list: list[FakePDFPage],
    ) -> None:
        # Store as protocol types for covariance
        self._pages = list(pages_list)
        self._metadata: dict[str, str | int | float | bool | None] = {}

    @property
    def pages(self) -> list[PageProtocol]:
        """Return list of pages."""
        return self._pages

    @property
    def metadata(self) -> dict[str, str | int | float | bool | None]:
        """Return PDF metadata."""
        return self._metadata

    def close(self) -> None:
        """Close the PDF file."""
        pass

    def __enter__(self) -> PDFProtocol:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        pass


__all__ = [
    "FakeDataDirectory",
    "FakeDataFile",
    "FakeDataFile3D",
    "FakeImzMLParser",
    "FakeMzMLReader",
    "FakePDF",
    "FakePDFPage",
    "_FakeNdArray",
    "_FakeNdArray1D",
    "_FakeNdArray2D",
    "_FakeNdArray3D",
]
