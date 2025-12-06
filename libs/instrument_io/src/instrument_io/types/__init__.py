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
    "EICData",
    "EICParams",
    "ErrorResult",
    "FileInfo",
    "ImagingSpectrum",
    "ImagingSpectrumMeta",
    "ImzMLFileInfo",
    "InstrumentInfo",
    "JSONValue",
    "MS2Spectrum",
    "MS3Spectrum",
    "MSLevel",
    "MSSpectrum",
    "MassPeak",
    "MassPeakList",
    "MethodInfo",
    "OperationResult",
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
    "make_error",
    "make_success",
]
