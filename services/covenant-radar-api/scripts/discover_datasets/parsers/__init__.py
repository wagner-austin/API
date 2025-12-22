"""Parsers for various dataset file formats.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from scripts.discover_datasets.parsers.arff import read_arff_header_and_sample
from scripts.discover_datasets.parsers.csv import (
    MAX_SAMPLE_ROWS,
    detect_csv_delimiter,
    read_csv_header_and_sample,
    read_data_header_and_sample,
    strip_quotes,
)
from scripts.discover_datasets.parsers.excel import (
    read_excel_header_and_sample,
    read_xls_header_and_sample,
)

__all__ = [
    "MAX_SAMPLE_ROWS",
    "detect_csv_delimiter",
    "read_arff_header_and_sample",
    "read_csv_header_and_sample",
    "read_data_header_and_sample",
    "read_excel_header_and_sample",
    "read_xls_header_and_sample",
    "strip_quotes",
]
