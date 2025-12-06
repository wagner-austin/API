"""Exception hierarchy for instrument_io library.

All exceptions propagate without recovery. Callers handle failures explicitly.
"""

from __future__ import annotations


class InstrumentIOError(Exception):
    """Base exception for instrument_io library.

    All library exceptions inherit from this base class.
    """


class UnsupportedFormatError(InstrumentIOError):
    """Raised when file format is not supported by any reader.

    Attributes:
        path: The path that could not be read.
        message: Description of why the format is unsupported.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class AgilentReadError(InstrumentIOError):
    """Raised when Agilent .D directory reading fails.

    Attributes:
        path: The .D directory path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class MzMLReadError(InstrumentIOError):
    """Raised when mzML/mzXML file reading fails.

    Attributes:
        path: The mzML file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class ExcelReadError(InstrumentIOError):
    """Raised when Excel file reading fails.

    Attributes:
        path: The Excel file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class DecodingError(InstrumentIOError):
    """Raised when data decoding or validation fails.

    This indicates malformed or unexpected data in an otherwise valid file.

    Attributes:
        context: Description of what was being decoded.
        message: Description of the decoding failure.
    """

    def __init__(self, context: str, message: str) -> None:
        self.context = context
        self.message = message
        super().__init__(f"{context}: {message}")


class WriterError(InstrumentIOError):
    """Raised when writing output fails.

    Attributes:
        path: The output path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class CSVReadError(InstrumentIOError):
    """Raised when CSV chromatogram file reading fails.

    Attributes:
        path: The CSV file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class ThermoReadError(InstrumentIOError):
    """Raised when Thermo .raw file reading fails.

    Attributes:
        path: The .raw file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class MGFReadError(InstrumentIOError):
    """Raised when MGF peak list file reading fails.

    Attributes:
        path: The .mgf file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class PDFReadError(InstrumentIOError):
    """Raised when PDF file reading fails.

    Attributes:
        path: The .pdf file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class ImzMLReadError(InstrumentIOError):
    """Raised when imzML imaging mass spectrometry file reading fails.

    Attributes:
        path: The .imzML file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class WatersReadError(InstrumentIOError):
    """Raised when Waters .raw directory reading fails.

    Note: Waters .raw is a DIRECTORY format (unlike Thermo .raw which is a file).

    Attributes:
        path: The .raw directory path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class TXTReadError(InstrumentIOError):
    """Raised when plain text file reading fails.

    Attributes:
        path: The .txt file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class SMPSReadError(InstrumentIOError):
    """Raised when SMPS .rps file reading fails.

    Attributes:
        path: The .rps file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class DOCXReadError(InstrumentIOError):
    """Raised when Word document (.docx) reading fails.

    Attributes:
        path: The .docx file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class PPTXReadError(InstrumentIOError):
    """Raised when PowerPoint presentation (.pptx) reading fails.

    Attributes:
        path: The .pptx file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


class MATReadError(InstrumentIOError):
    """Raised when MATLAB data file (.mat) reading fails.

    Attributes:
        path: The .mat file path.
        message: Description of the failure.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")


__all__ = [
    "AgilentReadError",
    "CSVReadError",
    "DOCXReadError",
    "DecodingError",
    "ExcelReadError",
    "ImzMLReadError",
    "InstrumentIOError",
    "MATReadError",
    "MGFReadError",
    "MzMLReadError",
    "PDFReadError",
    "PPTXReadError",
    "SMPSReadError",
    "TXTReadError",
    "ThermoReadError",
    "UnsupportedFormatError",
    "WatersReadError",
    "WriterError",
]
