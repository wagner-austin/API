"""Protocol definitions for external library abstraction.

Internal protocols used to provide type-safe interfaces to external
libraries (rainbow-api, pyteomics, openpyxl, numpy) without importing
them directly at module load time.
"""

from __future__ import annotations

# imzML protocols
from instrument_io._protocols.imzml import (
    ImzMLParserProtocol,
    _open_imzml,
)

# Numpy protocols
from instrument_io._protocols.numpy import (
    DTypeProtocol,
    NdArray1DProtocol,
    NdArray2DProtocol,
    NdArrayProtocol,
)

# Openpyxl protocols
from instrument_io._protocols.openpyxl import (
    AlignmentProtocol,
    CellProtocol,
    ColumnDimensionProtocol,
    FontProtocol,
    WorkbookProtocol,
    WorksheetProtocol,
    _create_alignment,
    _create_font,
    _create_styled_table,
    _create_workbook,
    _get_column_letter,
    _load_workbook,
)

# PDFPlumber protocols
from instrument_io._protocols.pdfplumber import (
    PageProtocol,
    PDFProtocol,
    _open_pdf,
)

# Pyteomics protocols
from instrument_io._protocols.pyteomics import (
    MzMLReaderProtocol,
    MzXMLReaderProtocol,
    SpectrumDictProtocol,
    _open_mzml,
    _open_mzxml,
)

# Rainbow protocols
from instrument_io._protocols.rainbow import (
    DataDirectoryProtocol,
    DataFileProtocol,
    _load_data_directory,
)

# Thermo protocols
from instrument_io._protocols.thermo import (
    _cleanup_temp_dir,
    _convert_raw_to_mzml,
    _create_temp_dir,
    _find_thermorawfileparser,
)

__all__ = [
    "AlignmentProtocol",
    "CellProtocol",
    "ColumnDimensionProtocol",
    "DTypeProtocol",
    "DataDirectoryProtocol",
    "DataFileProtocol",
    "FontProtocol",
    "ImzMLParserProtocol",
    "MzMLReaderProtocol",
    "MzXMLReaderProtocol",
    "NdArray1DProtocol",
    "NdArray2DProtocol",
    "NdArrayProtocol",
    "PDFProtocol",
    "PageProtocol",
    "SpectrumDictProtocol",
    "WorkbookProtocol",
    "WorksheetProtocol",
    "_cleanup_temp_dir",
    "_convert_raw_to_mzml",
    "_create_alignment",
    "_create_font",
    "_create_styled_table",
    "_create_temp_dir",
    "_create_workbook",
    "_find_thermorawfileparser",
    "_get_column_letter",
    "_load_data_directory",
    "_load_workbook",
    "_open_imzml",
    "_open_mzml",
    "_open_mzxml",
    "_open_pdf",
]
