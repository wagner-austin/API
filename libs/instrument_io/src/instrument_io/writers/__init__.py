"""Writer classes for instrument data output.

Provides typed writer interfaces for Excel files.
"""

from __future__ import annotations

from instrument_io.writers.base import ExcelWriterProtocol
from instrument_io.writers.excel import ExcelWriter

__all__ = [
    "ExcelWriter",
    "ExcelWriterProtocol",
]
