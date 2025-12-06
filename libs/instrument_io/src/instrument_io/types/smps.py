"""Type definitions for SMPS (Scanning Mobility Particle Sizer) data.

Provides TypedDict definitions for SMPS .rps file structure.
"""

from __future__ import annotations

from typing import TypedDict

from instrument_io.types.common import CellValue


class SMPSMetadata(TypedDict):
    """Metadata from SMPS .rps file header.

    Attributes:
        timestamp: Date and time of measurement.
        instrument: Instrument name from header.
        lower_voltage_limit: Lower voltage limit in Volts.
        upper_voltage_limit: Upper voltage limit in Volts.
        sample_duration: Sample duration in seconds.
    """

    timestamp: str
    instrument: str
    lower_voltage_limit: float
    upper_voltage_limit: float
    sample_duration: float


class SMPSData(TypedDict):
    """Complete SMPS data with metadata and measurements.

    Attributes:
        metadata: SMPS measurement metadata.
        data: List of measurement row dictionaries.
    """

    metadata: SMPSMetadata
    data: list[dict[str, CellValue]]


__all__ = [
    "SMPSData",
    "SMPSMetadata",
]
