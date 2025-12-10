"""TypedDict definitions for instrument data structures.

All types are immutable TypedDicts with strict typing.
"""

from __future__ import annotations

# Chromatogram types
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

# Common types
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

# Document types (for Word/PDF writers)
from instrument_io.types.document import (
    PAGE_SIZES,
    DocumentContent,
    DocumentSection,
    FigureContent,
    HeadingContent,
    ListContent,
    PageBreakContent,
    PageSize,
    ParagraphContent,
    TableContent,
    is_figure,
    is_heading,
    is_list,
    is_page_break,
    is_paragraph,
    is_table,
)

# Excel types
from instrument_io.types.excel import (
    ExcelRow,
    ExcelRows,
    ExcelSheets,
)

# Imaging types
from instrument_io.types.imaging import (
    ImagingSpectrum,
    ImagingSpectrumMeta,
    ImzMLFileInfo,
    SpatialCoordinate,
    SpectrumMode,
)

# Metadata types
from instrument_io.types.metadata import (
    AcquisitionInfo,
    BatchInfo,
    FileInfo,
    InstrumentInfo,
    MethodInfo,
    RunInfo,
    SampleInfo,
)

# Peak list types
from instrument_io.types.peaklist import (
    AnnotatedMassPeak,
    AnnotatedMassPeakList,
    ChromatogramPeak,
    ChromatogramPeakList,
    MassPeak,
    MassPeakList,
    PeakListMeta,
)

# Spectrum types
from instrument_io.types.spectrum import (
    MS2Spectrum,
    MS3Spectrum,
    MSSpectrum,
    PrecursorInfo,
    SpectrumData,
    SpectrumMeta,
    SpectrumStats,
)

__all__ = [
    "PAGE_SIZES",
    "AcquisitionInfo",
    "AnnotatedMassPeak",
    "AnnotatedMassPeakList",
    "BatchInfo",
    "CellValue",
    "ChromatogramData",
    "ChromatogramMeta",
    "ChromatogramPeak",
    "ChromatogramPeakList",
    "ChromatogramStats",
    "DADData",
    "DADSlice",
    "DocumentContent",
    "DocumentSection",
    "EICData",
    "EICParams",
    "ErrorResult",
    "ExcelRow",
    "ExcelRows",
    "ExcelSheets",
    "FigureContent",
    "FileInfo",
    "HeadingContent",
    "ImagingSpectrum",
    "ImagingSpectrumMeta",
    "ImzMLFileInfo",
    "InstrumentInfo",
    "JSONValue",
    "ListContent",
    "MS2Spectrum",
    "MS3Spectrum",
    "MSLevel",
    "MSSpectrum",
    "MassPeak",
    "MassPeakList",
    "MethodInfo",
    "OperationResult",
    "PageBreakContent",
    "PageSize",
    "ParagraphContent",
    "PeakListMeta",
    "Polarity",
    "PrecursorInfo",
    "RunInfo",
    "SampleInfo",
    "SignalType",
    "SpatialCoordinate",
    "SpectrumData",
    "SpectrumMeta",
    "SpectrumMode",
    "SpectrumStats",
    "SuccessResult",
    "TICData",
    "TableContent",
    "is_figure",
    "is_heading",
    "is_list",
    "is_page_break",
    "is_paragraph",
    "is_table",
    "make_error",
    "make_success",
]
