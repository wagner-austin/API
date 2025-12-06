"""Decoder functions for converting external data to typed structures.

Internal module providing type-safe conversion from external library
outputs (numpy arrays, pyteomics dicts) to our TypedDict structures.
"""

from __future__ import annotations

from instrument_io._decoders.agilent import (
    _compute_chromatogram_stats,
    _decode_intensities_1d,
    _decode_intensities_2d,
    _decode_retention_times,
    _decode_signal_type,
    _make_chromatogram_data,
    _percentile,
    _sum_2d_to_tic,
)
from instrument_io._decoders.csv import (
    _compute_chromatogram_stats_from_data,
    _detect_delimiter,
    _find_column_index,
    _parse_csv_line,
    _parse_float_column,
    _parse_float_value,
)
from instrument_io._decoders.excel import (
    _decode_cell_value,
    _decode_row_dict,
    _decode_rows,
    _extract_bool,
    _extract_float,
    _extract_float_or_none,
    _extract_int,
    _extract_int_or_none,
    _extract_string,
    _extract_string_or_none,
)
from instrument_io._decoders.imzml import (
    _compute_imzml_spectrum_stats,
    _decode_coordinate,
    _decode_imzml_polarity,
    _decode_spectrum_mode,
    _make_imzml_spectrum_data,
    _make_imzml_spectrum_meta,
)
from instrument_io._decoders.mzml import (
    _compute_spectrum_stats,
    _decode_intensity_array,
    _decode_ms_level,
    _decode_mz_array,
    _decode_polarity,
    _decode_retention_time,
    _decode_scan_number,
    _make_spectrum_data,
)
from instrument_io._decoders.pdf import (
    _decode_pdf_cell,
    _decode_pdf_row,
    _decode_pdf_table,
)

__all__ = [
    "_compute_chromatogram_stats",
    "_compute_chromatogram_stats_from_data",
    "_compute_imzml_spectrum_stats",
    "_compute_spectrum_stats",
    "_decode_cell_value",
    "_decode_coordinate",
    "_decode_imzml_polarity",
    "_decode_intensities_1d",
    "_decode_intensities_2d",
    "_decode_intensity_array",
    "_decode_ms_level",
    "_decode_mz_array",
    "_decode_pdf_cell",
    "_decode_pdf_row",
    "_decode_pdf_table",
    "_decode_polarity",
    "_decode_retention_time",
    "_decode_retention_times",
    "_decode_row_dict",
    "_decode_rows",
    "_decode_scan_number",
    "_decode_signal_type",
    "_decode_spectrum_mode",
    "_detect_delimiter",
    "_extract_bool",
    "_extract_float",
    "_extract_float_or_none",
    "_extract_int",
    "_extract_int_or_none",
    "_extract_string",
    "_extract_string_or_none",
    "_find_column_index",
    "_make_chromatogram_data",
    "_make_imzml_spectrum_data",
    "_make_imzml_spectrum_meta",
    "_make_spectrum_data",
    "_parse_csv_line",
    "_parse_float_column",
    "_parse_float_value",
    "_percentile",
    "_sum_2d_to_tic",
]
