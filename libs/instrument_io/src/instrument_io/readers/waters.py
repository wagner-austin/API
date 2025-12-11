"""Waters .raw directory reader implementation.

Provides typed reading of Waters MassLynx data via rainbow-api.
Uses Protocol-based dynamic imports - rainbow is loaded at runtime.

Note: Waters .raw is a DIRECTORY format (unlike Thermo .raw which is a file).
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
from instrument_io._exceptions import WatersReadError
from instrument_io._protocols.rainbow import (
    DataDirectoryProtocol,
    DataFileProtocol,
)
from instrument_io.testing import hooks
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


def _is_waters_raw_directory(path: Path) -> bool:
    """Check if path is a Waters .raw directory.

    Waters .raw format is a directory (unlike Thermo .raw which is a file).

    Args:
        path: Path to check.

    Returns:
        True if path is a directory with .raw extension.
    """
    return path.is_dir() and path.suffix.lower() == ".raw"


def _find_tic_file_optional(datadir: DataDirectoryProtocol) -> DataFileProtocol | None:
    """Find TIC data file in Waters directory, return None if not found.

    Args:
        datadir: Loaded DataDirectory.

    Returns:
        DataFileProtocol for TIC data, or None if not found.
    """
    tic_files = datadir.get_detector("TIC")
    if tic_files:
        return tic_files[0]

    for df in datadir.datafiles:
        detector_lower = df.detector.lower()
        if "tic" in detector_lower or "total" in detector_lower:
            return df

    return None


def _find_ms_file_optional(datadir: DataDirectoryProtocol) -> DataFileProtocol | None:
    """Find MS data file in Waters directory, return None if not found.

    Args:
        datadir: Loaded DataDirectory.

    Returns:
        DataFileProtocol for MS data, or None if not found.
    """
    ms_files = datadir.get_detector("MS")
    if ms_files:
        return ms_files[0]

    for df in datadir.datafiles:
        detector_lower = df.detector.lower()
        if "ms" in detector_lower:
            return df

    return None


def _find_ms_file(datadir: DataDirectoryProtocol, source_path: str) -> DataFileProtocol:
    """Find MS data file in Waters directory.

    Args:
        datadir: Loaded DataDirectory.
        source_path: Path for error messages.

    Returns:
        DataFileProtocol for MS data.

    Raises:
        WatersReadError: If no MS data found.
    """
    result = _find_ms_file_optional(datadir)
    if result is None:
        raise WatersReadError(source_path, "No MS data file found")
    return result


def _find_uv_file(datadir: DataDirectoryProtocol, source_path: str) -> DataFileProtocol:
    """Find UV/PDA data file in Waters directory.

    Args:
        datadir: Loaded DataDirectory.
        source_path: Path for error messages.

    Returns:
        DataFileProtocol for UV data.

    Raises:
        WatersReadError: If no UV data found.
    """
    uv_files = datadir.get_detector("UV")
    if uv_files:
        return uv_files[0]

    for df in datadir.datafiles:
        detector_lower = df.detector.lower()
        if "uv" in detector_lower or "pda" in detector_lower:
            return df

    raise WatersReadError(source_path, "No UV data file found")


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
        source_path: Path for error messages.

    Returns:
        1D list of summed intensities within tolerance window.

    Raises:
        WatersReadError: If no data points within tolerance.
    """
    if not ms_data or not mz_axis:
        raise WatersReadError(source_path, "Empty MS data for EIC extraction")

    mz_min = target_mz - mz_tolerance
    mz_max = target_mz + mz_tolerance

    matching_mz_indices: list[int] = []
    for i, mz in enumerate(mz_axis):
        if mz_min <= mz <= mz_max:
            matching_mz_indices.append(i)

    if not matching_mz_indices:
        raise WatersReadError(
            source_path,
            f"No m/z values within {mz_tolerance} Da of target {target_mz}",
        )

    num_timepoints = len(ms_data)
    result: list[float] = [0.0] * num_timepoints

    for time_idx in range(num_timepoints):
        for mz_idx in matching_mz_indices:
            result[time_idx] += ms_data[time_idx][mz_idx]

    return result


class WatersReader:
    """Reader for Waters .raw directories via rainbow-api.

    Provides typed access to TIC, EIC, UV/PDA, and MS data from Waters
    MassLynx data directories.

    Note: Waters .raw is a DIRECTORY format (unlike Thermo .raw which is a file).
    All methods raise exceptions on failure - no recovery or fallbacks.
    Data is loaded on demand; no state is cached between calls.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is a Waters .raw directory.

        Args:
            path: Path to check.

        Returns:
            True if path is a Waters .raw directory (not a file).
        """
        return _is_waters_raw_directory(path)

    def read_tic(self, path: Path) -> TICData:
        """Read or compute Total Ion Chromatogram from Waters .raw directory.

        First attempts to find a dedicated TIC detector file.
        If not found, computes TIC from MS data by summing across m/z.

        Args:
            path: Path to .raw directory.

        Returns:
            TICData TypedDict with complete chromatogram.

        Raises:
            WatersReadError: If neither TIC nor MS data available.
        """
        datadir = hooks.load_data_directory(path)

        tic_file = _find_tic_file_optional(datadir)
        if tic_file is not None:
            return self._read_tic_from_detector(path, tic_file)

        ms_file = _find_ms_file_optional(datadir)
        if ms_file is not None:
            return self._compute_tic_from_ms(path, ms_file)

        raise WatersReadError(str(path), "No TIC or MS data available")

    def _read_tic_from_detector(
        self,
        path: Path,
        tic_file: DataFileProtocol,
    ) -> TICData:
        """Read TIC from dedicated detector file.

        Args:
            path: Path to .raw directory.
            tic_file: TIC detector data file.

        Returns:
            TICData TypedDict.

        Raises:
            WatersReadError: If data shape is unexpected.
        """
        rt_list = _narrow_tolist_1d(tic_file.xlabels.tolist())
        retention_times = _decode_retention_times(rt_list)

        data_shape = tic_file.data.shape
        raw_data = tic_file.data.tolist()

        if len(data_shape) == 1:
            data_list_1d = _narrow_tolist_1d(raw_data)
            intensities: list[float] = [float(v) for v in data_list_1d]
        elif len(data_shape) == 2:
            data_list_2d = _narrow_tolist_2d(raw_data)
            intensities = [sum(row) for row in data_list_2d]
        else:
            raise WatersReadError(
                str(path),
                f"Unexpected data shape: {data_shape}",
            )

        meta = _build_chromatogram_meta(str(path), "TIC", tic_file.detector)
        data = _make_chromatogram_data(retention_times, intensities)
        stats = _compute_chromatogram_stats(retention_times, intensities)

        return TICData(meta=meta, data=data, stats=stats)

    def _compute_tic_from_ms(
        self,
        path: Path,
        ms_file: DataFileProtocol,
    ) -> TICData:
        """Compute TIC from MS data by summing across m/z.

        Args:
            path: Path to .raw directory.
            ms_file: MS data file.

        Returns:
            TICData TypedDict with computed TIC.

        Raises:
            WatersReadError: If MS data is not 2D.
        """
        rt_list = _narrow_tolist_1d(ms_file.xlabels.tolist())
        retention_times = _decode_retention_times(rt_list)

        data_shape = ms_file.data.shape
        if len(data_shape) != 2:
            raise WatersReadError(
                str(path),
                f"MS data must be 2D to compute TIC, got shape {data_shape}",
            )

        ms_data = _narrow_tolist_2d(ms_file.data.tolist())

        intensities: list[float] = []
        for time_row in ms_data:
            total = sum(time_row)
            intensities.append(total)

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
            path: Path to .raw directory.
            target_mz: Target m/z value.
            mz_tolerance: Tolerance window in Daltons.

        Returns:
            EICData TypedDict with extracted chromatogram.

        Raises:
            WatersReadError: If reading fails or MS data not found.
        """
        datadir = hooks.load_data_directory(path)
        ms_file = _find_ms_file(datadir, str(path))

        rt_list = _narrow_tolist_1d(ms_file.xlabels.tolist())
        retention_times = _decode_retention_times(rt_list)

        mz_list = _narrow_tolist_1d(ms_file.ylabels.tolist())

        data_shape = ms_file.data.shape
        if len(data_shape) != 2:
            raise WatersReadError(
                str(path),
                f"MS data must be 2D for EIC, got shape {data_shape}",
            )

        ms_data = _narrow_tolist_2d(ms_file.data.tolist())

        intensities = _extract_eic_intensities(
            ms_data,
            mz_list,
            target_mz,
            mz_tolerance,
            str(path),
        )

        meta = _build_chromatogram_meta(str(path), "EIC", ms_file.detector)
        params = EICParams(target_mz=target_mz, mz_tolerance=mz_tolerance)
        data = ChromatogramData(retention_times=retention_times, intensities=intensities)
        stats = _compute_chromatogram_stats(retention_times, intensities)

        return EICData(meta=meta, params=params, data=data, stats=stats)

    def read_uv(self, path: Path) -> DADData:
        """Read UV/PDA data from Waters .raw directory.

        Args:
            path: Path to .raw directory.

        Returns:
            DADData TypedDict with full wavelength/intensity matrix.

        Raises:
            WatersReadError: If reading fails or UV data not found.
        """
        datadir = hooks.load_data_directory(path)
        uv_file = _find_uv_file(datadir, str(path))

        rt_list = _narrow_tolist_1d(uv_file.xlabels.tolist())
        retention_times = _decode_retention_times(rt_list)

        wl_list = _narrow_tolist_1d(uv_file.ylabels.tolist())

        data_shape = uv_file.data.shape
        if len(data_shape) != 2:
            raise WatersReadError(
                str(path),
                f"UV data must be 2D, got shape {data_shape}",
            )

        intensity_matrix = _narrow_tolist_2d(uv_file.data.tolist())

        meta = _build_chromatogram_meta(str(path), "DAD", uv_file.detector)

        return DADData(
            meta=meta,
            wavelengths=wl_list,
            retention_times=retention_times,
            intensity_matrix=intensity_matrix,
        )

    def iter_spectra(self, path: Path) -> Generator[MSSpectrum, None, None]:
        """Iterate over all MS spectra in Waters .raw directory.

        Yields individual spectra at each time point from the MS data.

        Args:
            path: Path to .raw directory.

        Yields:
            MSSpectrum TypedDict for each time point.

        Raises:
            WatersReadError: If reading fails or MS data not found.
        """
        datadir = hooks.load_data_directory(path)
        ms_file = _find_ms_file(datadir, str(path))

        rt_list = _narrow_tolist_1d(ms_file.xlabels.tolist())
        retention_times = _decode_retention_times(rt_list)

        mz_list = _narrow_tolist_1d(ms_file.ylabels.tolist())

        data_shape = ms_file.data.shape
        if len(data_shape) != 2:
            raise WatersReadError(
                str(path),
                f"MS data must be 2D for spectra, got shape {data_shape}",
            )

        ms_data = _narrow_tolist_2d(ms_file.data.tolist())

        for scan_idx, rt in enumerate(retention_times):
            row = ms_data[scan_idx]
            if len(row) != len(mz_list):
                raise WatersReadError(
                    str(path),
                    (
                        f"MS data row length {len(row)} != mz axis length {len(mz_list)} "
                        f"at scan {scan_idx + 1}"
                    ),
                )

            mz_values: list[float] = []
            intensity_values: list[float] = []
            for mz, intensity in zip(mz_list, row, strict=True):
                if intensity > 0:
                    mz_values.append(mz)
                    intensity_values.append(intensity)

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
        """Find all Waters .raw directories under a root path.

        Recursively searches for directories ending in .raw.

        Args:
            data_root: Root directory to search.

        Returns:
            List of RunInfo TypedDicts for each .raw directory.

        Raises:
            WatersReadError: If data_root is not a directory.
        """
        if not data_root.is_dir():
            raise WatersReadError(str(data_root), "Not a directory")

        runs: list[RunInfo] = []

        for raw_dir in data_root.rglob("*.raw"):
            if not raw_dir.is_dir():
                continue

            # Extract run_id from directory name
            run_id = raw_dir.stem

            # Extract site from parent directory
            site = raw_dir.parent.name

            # Check for data types
            has_tic = False
            has_ms = False
            has_dad = False
            file_count = 0

            # Count files and check types
            for f in raw_dir.iterdir():
                if f.is_file():
                    file_count += 1
                    name_lower = f.name.lower()
                    if "tic" in name_lower or "_func001" in name_lower:
                        has_tic = True
                    if "ms" in name_lower or "_func" in name_lower:
                        has_ms = True
                    if "dad" in name_lower or "pda" in name_lower or "uv" in name_lower:
                        has_dad = True

            runs.append(
                RunInfo(
                    path=str(raw_dir),
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
    "WatersReader",
    "_build_chromatogram_meta",
    "_extract_eic_intensities",
    "_find_ms_file",
    "_find_ms_file_optional",
    "_find_tic_file_optional",
    "_find_uv_file",
    "_is_waters_raw_directory",
]
