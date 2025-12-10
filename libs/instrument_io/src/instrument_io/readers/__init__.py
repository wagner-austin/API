"""Reader classes for instrument data formats.

Provides typed reader interfaces for Agilent, mzML, MGF, imzML, Excel, CSV, Waters and Thermo files.
"""

from __future__ import annotations

from instrument_io.readers.agilent import AgilentReader
from instrument_io.readers.base import (
    ChromatogramReaderProtocol,
    ExcelReaderProtocol,
    SpectrumReaderProtocol,
)
from instrument_io.readers.csv import CSVChromatogramReader
from instrument_io.readers.docx import DOCXReader
from instrument_io.readers.excel import ExcelReader
from instrument_io.readers.imzml import ImzMLReader
from instrument_io.readers.mat import MATReader
from instrument_io.readers.mgf import MGFReader
from instrument_io.readers.mzml import MzMLReader
from instrument_io.readers.pdf import PDFReader
from instrument_io.readers.pptx import PPTXReader
from instrument_io.readers.smps import SMPSReader
from instrument_io.readers.thermo import ThermoReader
from instrument_io.readers.txt import TXTReader
from instrument_io.readers.waters import WatersReader

__all__ = [
    "AgilentReader",
    "CSVChromatogramReader",
    "ChromatogramReaderProtocol",
    "DOCXReader",
    "ExcelReader",
    "ExcelReaderProtocol",
    "ImzMLReader",
    "MATReader",
    "MGFReader",
    "MzMLReader",
    "PDFReader",
    "PPTXReader",
    "SMPSReader",
    "SpectrumReaderProtocol",
    "TXTReader",
    "ThermoReader",
    "WatersReader",
]
