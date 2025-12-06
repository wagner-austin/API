"""Thermo Fisher .raw file reader implementation.

Converts .raw files to mzML via ThermoRawFileParser CLI,
then reads with MzMLReader for fully typed output.

ThermoRawFileParser is cross-platform:
- Windows: Native .NET
- Linux/Mac: Via Mono runtime
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from instrument_io._exceptions import ThermoReadError
from instrument_io._protocols.thermo import (
    _cleanup_temp_dir,
    _convert_raw_to_mzml,
    _create_temp_dir,
)
from instrument_io.readers.mzml import MzMLReader
from instrument_io.types.chromatogram import (
    ChromatogramData,
    ChromatogramMeta,
    ChromatogramStats,
    EICData,
    EICParams,
    TICData,
)
from instrument_io.types.spectrum import MSSpectrum


def _is_raw_file(path: Path) -> bool:
    """Check if path is a Thermo .raw file."""
    return path.is_file() and path.suffix.lower() == ".raw"


def _compute_chromatogram_stats(
    retention_times: list[float],
    intensities: list[float],
) -> ChromatogramStats:
    """Compute statistics from chromatogram data.

    Args:
        retention_times: List of time points.
        intensities: List of intensity values.

    Returns:
        ChromatogramStats TypedDict.
    """
    num_points = len(retention_times)
    rt_min = min(retention_times)
    rt_max = max(retention_times)
    rt_step_mean = (rt_max - rt_min) / (num_points - 1) if num_points > 1 else 0.0

    intensity_min = min(intensities)
    intensity_max = max(intensities)
    intensity_mean = sum(intensities) / num_points

    # Percentile calculation
    sorted_intensities = sorted(intensities)
    p99_idx = int(0.99 * (num_points - 1))
    intensity_p99 = sorted_intensities[p99_idx]

    return ChromatogramStats(
        num_points=num_points,
        rt_min=rt_min,
        rt_max=rt_max,
        rt_step_mean=rt_step_mean,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        intensity_mean=intensity_mean,
        intensity_p99=intensity_p99,
    )


class ThermoReader:
    """Reader for Thermo Fisher .raw files.

    Converts .raw files to mzML using ThermoRawFileParser CLI,
    then reads the mzML with MzMLReader for fully typed output.

    Requires ThermoRawFileParser to be installed:
    - Windows: dotnet tool install -g ThermoRawFileParser
    - Linux/Mac: Install Mono, then download ThermoRawFileParser

    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def __init__(self) -> None:
        """Initialize ThermoReader with internal MzMLReader."""
        self._mzml_reader = MzMLReader()

    def supports_format(self, path: Path) -> bool:
        """Check if path is a Thermo .raw file.

        Args:
            path: Path to check.

        Returns:
            True if path is a .raw file.
        """
        return _is_raw_file(path)

    def read_tic(self, path: Path) -> TICData:
        """Read Total Ion Chromatogram from Thermo .raw file.

        Converts to mzML, extracts TIC from scan metadata.

        Args:
            path: Path to .raw file.

        Returns:
            TICData TypedDict with complete chromatogram.

        Raises:
            ThermoReadError: If conversion or reading fails.
        """
        source_path = str(path)

        if not _is_raw_file(path):
            raise ThermoReadError(source_path, "Not a .raw file")

        # Convert to mzML in temp directory
        temp_dir = _create_temp_dir()

        mzml_path = _convert_raw_to_mzml(path, temp_dir)

        # Extract TIC from spectra
        retention_times: list[float] = []
        intensities: list[float] = []

        for spectrum in self._mzml_reader.iter_spectra(mzml_path):
            retention_times.append(spectrum["meta"]["retention_time"])
            intensities.append(spectrum["meta"]["total_ion_current"])

        # Cleanup temp files
        _cleanup_temp_dir(temp_dir)

        if not retention_times:
            raise ThermoReadError(source_path, "No spectra found in file")

        # Build typed structures
        meta = ChromatogramMeta(
            source_path=source_path,
            instrument="",
            method_name="",
            sample_name="",
            acquisition_date="",
            signal_type="TIC",
            detector="MS",
        )
        data = ChromatogramData(
            retention_times=retention_times,
            intensities=intensities,
        )
        stats = _compute_chromatogram_stats(retention_times, intensities)

        return TICData(meta=meta, data=data, stats=stats)

    def read_eic(
        self,
        path: Path,
        target_mz: float,
        mz_tolerance: float,
    ) -> EICData:
        """Read Extracted Ion Chromatogram for target m/z.

        Converts to mzML, sums intensities within m/z window per scan.

        Args:
            path: Path to .raw file.
            target_mz: Target m/z value.
            mz_tolerance: Tolerance window in Daltons.

        Returns:
            EICData TypedDict with extracted chromatogram.

        Raises:
            ThermoReadError: If conversion or reading fails.
        """
        source_path = str(path)

        if not _is_raw_file(path):
            raise ThermoReadError(source_path, "Not a .raw file")

        # Convert to mzML
        temp_dir = _create_temp_dir()

        mzml_path = _convert_raw_to_mzml(path, temp_dir)

        # Calculate m/z window
        mz_low = target_mz - mz_tolerance
        mz_high = target_mz + mz_tolerance

        # Extract EIC from spectra
        retention_times: list[float] = []
        intensities: list[float] = []

        for spectrum in self._mzml_reader.iter_spectra(mzml_path):
            rt = spectrum["meta"]["retention_time"]
            mz_values = spectrum["data"]["mz_values"]
            int_values = spectrum["data"]["intensities"]

            # Sum intensities within m/z window
            total_intensity = 0.0
            for mz, intensity in zip(mz_values, int_values, strict=True):
                if mz_low <= mz <= mz_high:
                    total_intensity += intensity

            retention_times.append(rt)
            intensities.append(total_intensity)

        # Cleanup
        _cleanup_temp_dir(temp_dir)

        if not retention_times:
            raise ThermoReadError(source_path, "No spectra found in file")

        # Build typed structures
        meta = ChromatogramMeta(
            source_path=source_path,
            instrument="",
            method_name="",
            sample_name="",
            acquisition_date="",
            signal_type="EIC",
            detector="MS",
        )
        params = EICParams(target_mz=target_mz, mz_tolerance=mz_tolerance)
        data = ChromatogramData(
            retention_times=retention_times,
            intensities=intensities,
        )
        stats = _compute_chromatogram_stats(retention_times, intensities)

        return EICData(meta=meta, params=params, data=data, stats=stats)

    def read_spectrum(self, path: Path, scan_number: int) -> MSSpectrum:
        """Read a single spectrum by scan number.

        Args:
            path: Path to .raw file.
            scan_number: 1-based scan number.

        Returns:
            MSSpectrum TypedDict.

        Raises:
            ThermoReadError: If scan not found or reading fails.
        """
        source_path = str(path)

        if not _is_raw_file(path):
            raise ThermoReadError(source_path, "Not a .raw file")

        # Convert to mzML
        temp_dir = _create_temp_dir()

        mzml_path = _convert_raw_to_mzml(path, temp_dir)

        # Read specific spectrum
        spectrum = self._mzml_reader.read_spectrum(mzml_path, scan_number)

        # Cleanup
        _cleanup_temp_dir(temp_dir)

        # Update source path to original .raw file
        updated_meta = spectrum["meta"].copy()
        updated_meta["source_path"] = source_path

        return MSSpectrum(
            meta=updated_meta,
            data=spectrum["data"],
            stats=spectrum["stats"],
        )

    def iter_spectra(self, path: Path) -> Generator[MSSpectrum, None, None]:
        """Iterate over all spectra in .raw file.

        Args:
            path: Path to .raw file.

        Yields:
            MSSpectrum TypedDict for each scan.

        Raises:
            ThermoReadError: If conversion or reading fails.
        """
        source_path = str(path)

        if not _is_raw_file(path):
            raise ThermoReadError(source_path, "Not a .raw file")

        # Convert to mzML
        temp_dir = _create_temp_dir()

        mzml_path = _convert_raw_to_mzml(path, temp_dir)

        # Iterate and update source paths
        for spectrum in self._mzml_reader.iter_spectra(mzml_path):
            updated_meta = spectrum["meta"].copy()
            updated_meta["source_path"] = source_path

            yield MSSpectrum(
                meta=updated_meta,
                data=spectrum["data"],
                stats=spectrum["stats"],
            )

        # Cleanup after iteration complete
        _cleanup_temp_dir(temp_dir)

    def count_spectra(self, path: Path) -> int:
        """Count total number of spectra in .raw file.

        Args:
            path: Path to .raw file.

        Returns:
            Total spectrum count.

        Raises:
            ThermoReadError: If conversion or reading fails.
        """
        source_path = str(path)

        if not _is_raw_file(path):
            raise ThermoReadError(source_path, "Not a .raw file")

        # Convert to mzML
        temp_dir = _create_temp_dir()

        mzml_path = _convert_raw_to_mzml(path, temp_dir)

        count = self._mzml_reader.count_spectra(mzml_path)

        # Cleanup
        _cleanup_temp_dir(temp_dir)

        return count


__all__ = [
    "ThermoReader",
]
