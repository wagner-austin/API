"""Strictly typed IO for analytical chemistry instrument data formats.

This library provides readers and writers for:
- Agilent .D directories (via rainbow-api)
- mzML/mzXML files (via pyteomics)
- Excel files (via openpyxl/polars)
- CSV/TSV chromatogram exports
- PDF files (via pdfplumber)
- Thermo .raw files (via pythonnet/ThermoFisher.CommonCore)
- Waters .raw directories (via rainbow-api)

All data structures use TypedDicts for strict typing.
No Any, cast, or type: ignore used anywhere.
"""

from __future__ import annotations

# Errors
from instrument_io._exceptions import (
    AgilentReadError,
    CSVReadError,
    DOCXReadError,
    DecodingError,
    ExcelReadError,
    ImzMLReadError,
    InstrumentIOError,
    MATReadError,
    MGFReadError,
    MzMLReadError,
    PDFReadError,
    PPTXReadError,
    SMPSReadError,
    TXTReadError,
    ThermoReadError,
    UnsupportedFormatError,
    WatersReadError,
    WriterError,
)

# Readers
from instrument_io.readers import (
    AgilentReader,
    ChromatogramReaderProtocol,
    CSVChromatogramReader,
    DOCXReader,
    ExcelReader,
    ExcelReaderProtocol,
    ImzMLReader,
    MATReader,
    MGFReader,
    MzMLReader,
    PDFReader,
    PPTXReader,
    SMPSReader,
    SpectrumReaderProtocol,
    ThermoReader,
    TXTReader,
    WatersReader,
)

# Types - Chromatogram
from instrument_io.types.chromatogram import (
    ChromatogramData,
    ChromatogramMeta,
    ChromatogramStats,
    DADData,
    DADSlice,
    EICData,
    EICParams,
    TICData,
)

# Types - Common
from instrument_io.types.common import (
    CellValue,
    ErrorResult,
    JSONValue,
    MSLevel,
    OperationResult,
    Polarity,
    SignalType,
    SuccessResult,
    make_error,
    make_success,
)

# Types - Metadata
from instrument_io.types.metadata import (
    AcquisitionInfo,
    BatchInfo,
    FileInfo,
    InstrumentInfo,
    MethodInfo,
    RunInfo,
    SampleInfo,
)

# Types - Peak List
from instrument_io.types.peaklist import (
    AnnotatedMassPeak,
    AnnotatedMassPeakList,
    ChromatogramPeak,
    ChromatogramPeakList,
    MassPeak,
    MassPeakList,
    PeakListMeta,
)

# Types - Spectrum
from instrument_io.types.spectrum import (
    MS2Spectrum,
    MS3Spectrum,
    MSSpectrum,
    PrecursorInfo,
    SpectrumData,
    SpectrumMeta,
    SpectrumStats,
)

# Writers
from instrument_io.writers import (
    ExcelWriter,
    ExcelWriterProtocol,
)

__all__ = [
    "AcquisitionInfo",
    "AgilentReadError",
    "AgilentReader",
    "AnnotatedMassPeak",
    "AnnotatedMassPeakList",
    "BatchInfo",
    "CSVChromatogramReader",
    "CSVReadError",
    "CellValue",
    "ChromatogramData",
    "ChromatogramMeta",
    "ChromatogramPeak",
    "ChromatogramPeakList",
    "ChromatogramReaderProtocol",
    "ChromatogramStats",
    "DADData",
    "DADSlice",
    "DOCXReadError",
    "DOCXReader",
    "DecodingError",
    "EICData",
    "EICParams",
    "ErrorResult",
    "ExcelReadError",
    "ExcelReader",
    "ExcelReaderProtocol",
    "ExcelWriter",
    "ExcelWriterProtocol",
    "FileInfo",
    "ImzMLReadError",
    "ImzMLReader",
    "InstrumentIOError",
    "InstrumentInfo",
    "JSONValue",
    "MATReadError",
    "MATReader",
    "MGFReadError",
    "MGFReader",
    "MS2Spectrum",
    "MS3Spectrum",
    "MSLevel",
    "MSSpectrum",
    "MassPeak",
    "MassPeakList",
    "MethodInfo",
    "MzMLReadError",
    "MzMLReader",
    "OperationResult",
    "PDFReadError",
    "PDFReader",
    "PPTXReadError",
    "PPTXReader",
    "PeakListMeta",
    "Polarity",
    "PrecursorInfo",
    "RunInfo",
    "SMPSReadError",
    "SMPSReader",
    "SampleInfo",
    "SignalType",
    "SpectrumData",
    "SpectrumMeta",
    "SpectrumReaderProtocol",
    "SpectrumStats",
    "SuccessResult",
    "TICData",
    "TXTReadError",
    "TXTReader",
    "ThermoReadError",
    "ThermoReader",
    "UnsupportedFormatError",
    "WatersReadError",
    "WatersReader",
    "WriterError",
    "make_error",
    "make_success",
]
