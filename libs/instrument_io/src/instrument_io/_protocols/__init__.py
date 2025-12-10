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
    _create_table,
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

# Polars protocols
from instrument_io._protocols.polars import (
    PolarsDataFrameProtocol,
    _get_polars_read_excel,
)

# Pyteomics protocols
from instrument_io._protocols.pyteomics import (
    MzMLReaderProtocol,
    MzXMLReaderProtocol,
    SpectrumDictProtocol,
    _open_mzml,
    _open_mzxml,
)

# Python-docx protocols
from instrument_io._protocols.python_docx import (
    CellProtocol as DocxCellProtocol,
)
from instrument_io._protocols.python_docx import (
    DocumentProtocol,
    InlineShapeProtocol,
    LengthProtocol,
    ParagraphProtocol,
    RunProtocol,
    WdAlignParagraphProtocol,
    _create_document,
    _get_inches,
    _get_pt,
    _get_wd_align_center,
    _open_docx,
)
from instrument_io._protocols.python_docx import (
    RowProtocol as DocxRowProtocol,
)
from instrument_io._protocols.python_docx import (
    StyleProtocol as DocxStyleProtocol,
)
from instrument_io._protocols.python_docx import (
    TableProtocol as DocxTableProtocol,
)

# Rainbow protocols
from instrument_io._protocols.rainbow import (
    DataDirectoryProtocol,
    DataFileProtocol,
    _load_data_directory,
)

# Reportlab protocols
from instrument_io._protocols.reportlab import (
    CanvasProtocol,
    FlowableProtocol,
    ParagraphStyleProtocol,
    SimpleDocTemplateProtocol,
    StyleSheetProtocol,
    TableStyleCommand4,
    TableStyleCommand5,
    TableStyleProtocol,
    _command4_to_tuple,
    _command5_to_tuple,
    _create_image,
    _create_list_flowable,
    _create_list_item,
    _create_page_break,
    _create_paragraph,
    _create_simple_doc_template,
    _create_spacer,
    _create_table_style_from_commands4,
    _create_table_style_from_commands5,
    _create_table_style_mixed,
    _get_sample_stylesheet,
)
from instrument_io._protocols.reportlab import (
    StyleProtocol as ReportlabStyleProtocol,
)
from instrument_io._protocols.reportlab import (
    _create_table as _create_reportlab_table,
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
    "CanvasProtocol",
    "CellProtocol",
    "ColumnDimensionProtocol",
    "DTypeProtocol",
    "DataDirectoryProtocol",
    "DataFileProtocol",
    "DocumentProtocol",
    "DocxCellProtocol",
    "DocxRowProtocol",
    "DocxStyleProtocol",
    "DocxTableProtocol",
    "FlowableProtocol",
    "FontProtocol",
    "ImzMLParserProtocol",
    "InlineShapeProtocol",
    "LengthProtocol",
    "MzMLReaderProtocol",
    "MzXMLReaderProtocol",
    "NdArray1DProtocol",
    "NdArray2DProtocol",
    "NdArrayProtocol",
    "PDFProtocol",
    "PageProtocol",
    "ParagraphProtocol",
    "ParagraphStyleProtocol",
    "PolarsDataFrameProtocol",
    "ReportlabStyleProtocol",
    "RunProtocol",
    "SimpleDocTemplateProtocol",
    "SpectrumDictProtocol",
    "StyleSheetProtocol",
    "TableStyleCommand4",
    "TableStyleCommand5",
    "TableStyleProtocol",
    "WdAlignParagraphProtocol",
    "WorkbookProtocol",
    "WorksheetProtocol",
    "_cleanup_temp_dir",
    "_command4_to_tuple",
    "_command5_to_tuple",
    "_convert_raw_to_mzml",
    "_create_alignment",
    "_create_document",
    "_create_font",
    "_create_image",
    "_create_list_flowable",
    "_create_list_item",
    "_create_page_break",
    "_create_paragraph",
    "_create_reportlab_table",
    "_create_simple_doc_template",
    "_create_spacer",
    "_create_table",
    "_create_table_style_from_commands4",
    "_create_table_style_from_commands5",
    "_create_table_style_mixed",
    "_create_temp_dir",
    "_create_workbook",
    "_find_thermorawfileparser",
    "_get_column_letter",
    "_get_inches",
    "_get_polars_read_excel",
    "_get_pt",
    "_get_sample_stylesheet",
    "_get_wd_align_center",
    "_load_data_directory",
    "_load_workbook",
    "_open_docx",
    "_open_imzml",
    "_open_mzml",
    "_open_mzxml",
    "_open_pdf",
]
