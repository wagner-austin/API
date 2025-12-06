"""Decoder functions for SMPS .rps data files.

Converts tab-delimited SMPS data to typed structures.
"""

from __future__ import annotations

from instrument_io._exceptions import DecodingError
from instrument_io.types.common import CellValue
from instrument_io.types.smps import SMPSData, SMPSMetadata


def _parse_cell_value(value: str) -> CellValue:
    """Parse a cell value to appropriate type.

    Args:
        value: String cell value.

    Returns:
        CellValue (str, int, float, bool, or None).
    """
    stripped = value.strip()
    if not stripped:
        return None

    # Try boolean
    lower = stripped.lower()
    if lower in ("true", "yes", "y"):
        return True
    if lower in ("false", "no", "n"):
        return False

    # Try integer
    if stripped.lstrip("-").isdigit():
        return int(stripped)

    # Try float
    try:
        return float(stripped)
    except ValueError:
        # Return as string
        return stripped


def _decode_smps_metadata(lines: list[str]) -> SMPSMetadata:
    """Decode SMPS metadata from header lines.

    SMPS .rps files have this structure:
    - Line 0: timestamp, time, instrument name (tab-separated)
    - Line 1: parameter names (tab-separated)
    - Line 2: parameter values (tab-separated)

    Args:
        lines: List of all lines from .rps file.

    Returns:
        SMPSMetadata TypedDict.

    Raises:
        DecodingError: If metadata parsing fails.
    """
    if len(lines) < 3:
        raise DecodingError("SMPS metadata", "File has fewer than 3 lines")

    # Parse line 0: date, time, instrument
    header_parts = lines[0].split("\t")
    if len(header_parts) < 3:
        raise DecodingError("SMPS header", "Expected at least 3 tab-separated values")

    date_str = header_parts[0].strip()
    time_str = header_parts[1].strip()
    instrument = header_parts[2].strip()
    timestamp = f"{date_str} {time_str}"

    # Parse lines 1-2: parameter names and values
    param_names = [name.strip() for name in lines[1].split("\t")]
    param_values = [val.strip() for val in lines[2].split("\t")]

    if len(param_names) != len(param_values):
        raise DecodingError(
            "SMPS parameters",
            f"Mismatch: {len(param_names)} names vs {len(param_values)} values",
        )

    # Build parameter dictionary
    params: dict[str, str] = {}
    for name, value in zip(param_names, param_values, strict=True):
        if name:
            params[name] = value

    # Extract required fields
    lower_voltage_str = params.get("Lower Voltage Limit [V]", "0")
    lower_voltage = float(lower_voltage_str)

    upper_voltage_str = params.get("Upper Voltage Limit [V]", "0")
    upper_voltage = float(upper_voltage_str)

    sample_duration_str = params.get("Sample Duration [s]", "0")
    sample_duration = float(sample_duration_str)

    return SMPSMetadata(
        timestamp=timestamp,
        instrument=instrument,
        lower_voltage_limit=lower_voltage,
        upper_voltage_limit=upper_voltage,
        sample_duration=sample_duration,
    )


def _decode_smps_data(lines: list[str]) -> list[dict[str, CellValue]]:
    """Decode SMPS data rows.

    Data starts at line 3 (0-indexed).
    Line 3 contains column headers, lines 4+ contain data.

    Args:
        lines: List of all lines from .rps file.

    Returns:
        List of row dictionaries with typed cell values.

    Raises:
        DecodingError: If data parsing fails.
    """
    if len(lines) < 4:
        raise DecodingError("SMPS data", "File has fewer than 4 lines (no data rows)")

    # Line 3: column headers
    headers = [h.strip() for h in lines[3].split("\t")]

    # Lines 4+: data rows
    result: list[dict[str, CellValue]] = []
    for line in lines[4:]:
        values = line.split("\t")
        if len(values) != len(headers):
            # Skip rows with mismatched column counts
            continue

        row_dict: dict[str, CellValue] = {}
        for header, value in zip(headers, values, strict=True):
            if header:
                row_dict[header] = _parse_cell_value(value)

        if row_dict:
            result.append(row_dict)

    return result


def _decode_smps_full(lines: list[str]) -> SMPSData:
    """Decode full SMPS file (metadata + data).

    Args:
        lines: List of all lines from .rps file.

    Returns:
        SMPSData TypedDict with metadata and data.

    Raises:
        DecodingError: If decoding fails.
    """
    metadata = _decode_smps_metadata(lines)
    data = _decode_smps_data(lines)

    return SMPSData(
        metadata=metadata,
        data=data,
    )


__all__ = [
    "_decode_smps_data",
    "_decode_smps_full",
    "_decode_smps_metadata",
    "_parse_cell_value",
]
