"""Agilent .D directory reader implementation.

Provides typed reading of Agilent ChemStation/MassHunter data via rainbow-api.
Uses Protocol-based dynamic imports - rainbow is loaded at runtime.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from instrument_io._decoders.agilent import (
    _compute_chromatogram_stats,
    _decode_retention_times,
    _make_chromatogram_data,
    _narrow_tolist_1d,
    _narrow_tolist_2d,
)
from instrument_io._exceptions import AgilentReadError
from instrument_io._protocols.rainbow import (
    DataDirectoryProtocol,
    DataFileProtocol,
    _load_data_directory,
)
from instrument_io.types.chromatogram import (
    ChromatogramData,
    ChromatogramMeta,
    DADData,
    EICData,
    EICParams,
    TICData,
)
from instrument_io.types.common import SignalType
from instrument_io.types.metadata import RunInfo
from instrument_io.types.spectrum import (
    MSSpectrum,
    SpectrumData,
    SpectrumMeta,
    SpectrumStats,
)


def _is_agilent_d_directory(path: Path) -> bool:
    """Check if path is an Agilent .D directory."""
    return path.is_dir() and path.suffix.lower() == ".d"


def _find_ms_file_optional(datadir: DataDirectoryProtocol) -> DataFileProtocol | None:
    """Find MS data file in Agilent directory, return None if not found.

    Args:
        datadir: Loaded DataDirectory.

    Returns:
        DataFileProtocol for MS data, or None if not found.
    """
    ms_files = datadir.get_detector("MS")
    if ms_files:
        return ms_files[0]
    return None


def _find_ms_file(datadir: DataDirectoryProtocol) -> DataFileProtocol:
    """Find MS data file in Agilent directory.

    Args:
        datadir: Loaded DataDirectory.

    Returns:
        DataFileProtocol for MS data.

    Raises:
        AgilentReadError: If no MS data found.
    """
    result = _find_ms_file_optional(datadir)
    if result is None:
        raise AgilentReadError(datadir.directory, "No MS data file found")
    return result


def _build_chromatogram_meta(
    source_path: str,
    signal_type: SignalType,
    detector: str,
) -> ChromatogramMeta:
    """Build ChromatogramMeta with default values for unknown fields.

    Args:
        source_path: Path to source file.
        signal_type: Signal type literal.
        detector: Detector name string.

    Returns:
        ChromatogramMeta TypedDict.
    """
    return ChromatogramMeta(
        source_path=source_path,
        instrument="",
        method_name="",
        sample_name="",
        acquisition_date="",
        signal_type=signal_type,
        detector=detector,
    )


def _extract_eic_intensities(
    ms_data: list[list[float]],
    mz_axis: list[float],
    target_mz: float,
    mz_tolerance: float,
    source_path: str,
) -> list[float]:
    """Extract EIC intensities from 2D MS data.

    Args:
        ms_data: 2D intensity matrix [time_idx][mz_idx].
        mz_axis: List of m/z values (column indices).
        target_mz: Target m/z for extraction.
        mz_tolerance: Tolerance window.

    Returns:
        1D list of summed intensities within tolerance window.

    Raises:
        AgilentReadError: If no data points within tolerance.
    """
    if not ms_data or not mz_axis:
        raise AgilentReadError(source_path, "Empty MS data for EIC extraction")

    # Find m/z indices within tolerance
    mz_min = target_mz - mz_tolerance
    mz_max = target_mz + mz_tolerance

    matching_mz_indices: list[int] = []
    for i, mz in enumerate(mz_axis):
        if mz_min <= mz <= mz_max:
            matching_mz_indices.append(i)

    if not matching_mz_indices:
        raise AgilentReadError(
            source_path,
            f"No m/z values within {mz_tolerance} Da of target {target_mz}",
        )

    # Sum intensities across matching m/z channels
    # Data layout: ms_data[time_idx][mz_idx]
    num_timepoints = len(ms_data)
    result: list[float] = [0.0] * num_timepoints

    for time_idx in range(num_timepoints):
        for mz_idx in matching_mz_indices:
            result[time_idx] += ms_data[time_idx][mz_idx]

    return result


class AgilentReader:
    """Reader for Agilent .D directories via rainbow-api.

    Provides typed access to TIC, EIC, DAD, and MS data from Agilent
    ChemStation and MassHunter data directories.

    All methods raise exceptions on failure - no recovery or fallbacks.
    Data is loaded on demand; no state is cached between calls.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is an Agilent .D directory.

        Args:
            path: Path to check.

        Returns:
            True if path is an Agilent .D directory.
        """
        return _is_agilent_d_directory(path)

    def read_tic(self, path: Path) -> TICData:
        """Compute Total Ion Chromatogram from Agilent .D directory.

        Computes TIC from MS data by summing across m/z channels.

        Args:
            path: Path to .D directory.

        Returns:
            TICData TypedDict with complete chromatogram.

        Raises:
            AgilentReadError: If MS data not available.
        """
        datadir = _load_data_directory(path)

        # Compute from MS data
        ms_file = _find_ms_file_optional(datadir)
        if ms_file is not None:
            return self._compute_tic_from_ms(path, ms_file)

        raise AgilentReadError(str(path), "No MS data available for TIC computation")

    def _compute_tic_from_ms(
        self,
        path: Path,
        ms_file: DataFileProtocol,
    ) -> TICData:
        """Compute TIC from MS data by summing across m/z.

        Args:
            path: Path to .D directory.
            ms_file: MS data file.

        Returns:
            TICData TypedDict with computed TIC.
        """
        # Get retention times
        rt_list = _narrow_tolist_1d(ms_file.xlabels.tolist())
        retention_times = _decode_retention_times(rt_list)

        # Get 2D intensity matrix (rainbow-api always produces 2D MS data)
        # Data layout: ms_data[time_idx][mz_idx]
        ms_data = _narrow_tolist_2d(ms_file.data.tolist())

        # Sum across m/z for each time point
        intensities: list[float] = []
        for time_row in ms_data:
            total = sum(time_row)
            intensities.append(total)

        # Build typed structures (signal type TIC, detector name from MS)
        meta = _build_chromatogram_meta(str(path), "TIC", f"{ms_file.detector} (computed)")
        data = _make_chromatogram_data(retention_times, intensities)
        stats = _compute_chromatogram_stats(retention_times, intensities)

        return TICData(meta=meta, data=data, stats=stats)

    def read_eic(
        self,
        path: Path,
        target_mz: float,
        mz_tolerance: float,
    ) -> EICData:
        """Read Extracted Ion Chromatogram for target m/z.

        Args:
            path: Path to .D directory.
            target_mz: Target m/z value.
            mz_tolerance: Tolerance window in Daltons.

        Returns:
            EICData TypedDict with extracted chromatogram.

        Raises:
            AgilentReadError: If reading fails or MS data not found.
        """
        datadir = _load_data_directory(path)
        ms_file = _find_ms_file(datadir)

        # Get retention times
        rt_list = _narrow_tolist_1d(ms_file.xlabels.tolist())
        retention_times = _decode_retention_times(rt_list)

        # Get m/z axis
        mz_list = _narrow_tolist_1d(ms_file.ylabels.tolist())

        # Get 2D intensity matrix (rainbow-api always produces 2D MS data)
        ms_data = _narrow_tolist_2d(ms_file.data.tolist())

        # Extract EIC
        intensities = _extract_eic_intensities(
            ms_data,
            mz_list,
            target_mz,
            mz_tolerance,
            str(path),
        )

        # Build typed structures
        meta = _build_chromatogram_meta(str(path), "EIC", ms_file.detector)
        params = EICParams(target_mz=target_mz, mz_tolerance=mz_tolerance)
        data = ChromatogramData(retention_times=retention_times, intensities=intensities)
        stats = _compute_chromatogram_stats(retention_times, intensities)

        return EICData(meta=meta, params=params, data=data, stats=stats)

    def read_dad(self, path: Path) -> DADData:
        """Read full DAD data from Agilent .D directory.

        Note: rainbow-api returns UV detector data as "UV", not "DAD".
        This method always raises since DAD detector type is not produced.

        Args:
            path: Path to .D directory.

        Raises:
            AgilentReadError: Always raised (DAD not available via rainbow-api).
        """
        raise AgilentReadError(
            str(path),
            "DAD data not available (rainbow-api uses 'UV' detector type for UV files)",
        )

    def iter_spectra(self, path: Path) -> Generator[MSSpectrum, None, None]:
        """Iterate over all MS spectra in Agilent .D directory.

        Yields individual spectra at each time point from the MS data.

        Args:
            path: Path to .D directory.

        Yields:
            MSSpectrum TypedDict for each time point.

        Raises:
            AgilentReadError: If reading fails or MS data not found.
        """
        datadir = _load_data_directory(path)
        ms_file = _find_ms_file(datadir)

        # Get retention times
        rt_list = _narrow_tolist_1d(ms_file.xlabels.tolist())
        retention_times = _decode_retention_times(rt_list)

        # Get m/z axis
        mz_list = _narrow_tolist_1d(ms_file.ylabels.tolist())

        # Get 2D intensity matrix (rainbow-api always produces 2D MS data)
        ms_data = _narrow_tolist_2d(ms_file.data.tolist())

        # Iterate over time points
        # Data layout: ms_data[time_idx][mz_idx]
        for scan_idx, rt in enumerate(retention_times):
            # Extract spectrum at this time point (row length always matches mz_list)
            row = ms_data[scan_idx]
            intensities: list[float] = list(row)

            # Filter to non-zero peaks
            mz_values: list[float] = []
            intensity_values: list[float] = []
            for mz, intensity in zip(mz_list, intensities, strict=True):
                if intensity > 0:
                    mz_values.append(mz)
                    intensity_values.append(intensity)

            # Compute stats
            if mz_values:
                num_peaks = len(mz_values)
                mz_min = min(mz_values)
                mz_max = max(mz_values)
                max_intensity = max(intensity_values)
                max_idx = intensity_values.index(max_intensity)
                base_peak_mz = mz_values[max_idx]
            else:
                num_peaks = 0
                mz_min = 0.0
                mz_max = 0.0
                max_intensity = 0.0
                base_peak_mz = 0.0

            meta = SpectrumMeta(
                source_path=str(path),
                scan_number=scan_idx + 1,
                retention_time=rt,
                ms_level=1,
                polarity="unknown",
                total_ion_current=sum(intensity_values),
            )

            data = SpectrumData(
                mz_values=mz_values,
                intensities=intensity_values,
            )

            stats = SpectrumStats(
                num_peaks=num_peaks,
                mz_min=mz_min,
                mz_max=mz_max,
                base_peak_mz=base_peak_mz,
                base_peak_intensity=max_intensity,
            )

            yield MSSpectrum(meta=meta, data=data, stats=stats)

    def find_runs(self, data_root: Path) -> list[RunInfo]:
        """Find all Agilent .D directories under a root path.

        Recursively searches for directories ending in .D.

        Args:
            data_root: Root directory to search.

        Returns:
            List of RunInfo TypedDicts for each .D directory.

        Raises:
            AgilentReadError: If data_root is not a directory.
        """
        if not data_root.is_dir():
            raise AgilentReadError(str(data_root), "Not a directory")

        runs: list[RunInfo] = []

        for d_dir in data_root.rglob("*.D"):
            if not d_dir.is_dir():
                continue

            # Extract run_id from directory name
            run_id = d_dir.stem

            # Extract site from parent directory
            site = d_dir.parent.name

            # Check for data types
            has_tic = False
            has_ms = False
            has_dad = False
            file_count = 0

            # Count files and check types
            for f in d_dir.iterdir():
                if f.is_file():
                    file_count += 1
                    name_lower = f.name.lower()
                    if "tic" in name_lower:
                        has_tic = True
                    if "ms" in name_lower:
                        has_ms = True
                    if "dad" in name_lower:
                        has_dad = True

            runs.append(
                RunInfo(
                    path=str(d_dir),
                    run_id=run_id,
                    site=site,
                    has_tic=has_tic,
                    has_ms=has_ms,
                    has_dad=has_dad,
                    file_count=file_count,
                )
            )

        return runs


__all__ = [
    "AgilentReader",
]
