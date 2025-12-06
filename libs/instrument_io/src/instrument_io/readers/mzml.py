"""mzML/mzXML file reader implementation.

Provides typed reading of mass spectrometry data in mzML and mzXML formats
via pyteomics. Uses Protocol-based dynamic imports.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Protocol, runtime_checkable

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
from instrument_io._exceptions import MzMLReadError
from instrument_io._protocols.pyteomics import (
    MzMLReaderProtocol,
    MzXMLReaderProtocol,
    SpectrumDictProtocol,
    _open_mzml,
    _open_mzxml,
)
from instrument_io.types.chromatogram import (
    ChromatogramData,
    ChromatogramMeta,
    ChromatogramStats,
    EICData,
    EICParams,
    TICData,
)
from instrument_io.types.spectrum import (
    MS2Spectrum,
    MSSpectrum,
    PrecursorInfo,
    SpectrumMeta,
)


@runtime_checkable
class _ArrayLikeProtocol(Protocol):
    """Protocol for objects with tolist() method."""

    def tolist(self) -> list[float]: ...


def _compute_chromatogram_stats(
    retention_times: list[float],
    intensities: list[float],
) -> ChromatogramStats:
    """Compute statistics for chromatogram data.

    Args:
        retention_times: List of retention times.
        intensities: List of intensities.

    Returns:
        ChromatogramStats TypedDict.
    """
    n = len(retention_times)

    if n == 0:
        return ChromatogramStats(
            num_points=0,
            rt_min=0.0,
            rt_max=0.0,
            rt_step_mean=0.0,
            intensity_min=0.0,
            intensity_max=0.0,
            intensity_mean=0.0,
            intensity_p99=0.0,
        )

    rt_min = min(retention_times)
    rt_max = max(retention_times)
    rt_step_mean = (rt_max - rt_min) / (n - 1) if n > 1 else 0.0

    intensity_min = min(intensities)
    intensity_max = max(intensities)
    intensity_mean = sum(intensities) / n

    # Compute 99th percentile (int(n * 0.99) is always < n for n > 0)
    sorted_intensities = sorted(intensities)
    intensity_p99 = sorted_intensities[int(n * 0.99)]

    return ChromatogramStats(
        num_points=n,
        rt_min=rt_min,
        rt_max=rt_max,
        rt_step_mean=rt_step_mean,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        intensity_mean=intensity_mean,
        intensity_p99=intensity_p99,
    )


def _is_mzml_file(path: Path) -> bool:
    """Check if path is an mzML file."""
    return path.is_file() and path.suffix.lower() == ".mzml"


def _is_mzxml_file(path: Path) -> bool:
    """Check if path is an mzXML file."""
    return path.is_file() and path.suffix.lower() == ".mzxml"


def _extract_array_from_spectrum(
    spectrum: SpectrumDictProtocol,
    key: str,
    source_path: str,
) -> list[float]:
    """Extract array from spectrum dict and convert to list.

    Args:
        spectrum: Spectrum dictionary from pyteomics.
        key: Array key (e.g., "m/z array", "intensity array").

    Returns:
        List of float values.

    Raises:
        MzMLReadError: If array not found or invalid type.
    """
    if key not in spectrum:
        raise MzMLReadError(source_path, f"Missing required array: {key}")

    value = spectrum[key]
    # Check if value has tolist() method (numpy array or similar)
    if not isinstance(value, _ArrayLikeProtocol):
        raise MzMLReadError(
            source_path,
            f"Expected array for key '{key}', got {type(value).__name__}",
        )

    arr_list: list[float] = value.tolist()
    return arr_list


def _extract_float_or_zero(
    spectrum: SpectrumDictProtocol,
    key: str,
) -> float:
    """Extract float value from spectrum, defaulting to 0.0.

    Args:
        spectrum: Spectrum dictionary.
        key: Key to extract.

    Returns:
        Float value or 0.0 if not found.
    """
    value = spectrum.get(key)
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _extract_polarity_string(spectrum: SpectrumDictProtocol) -> str | None:
    """Extract polarity string from spectrum metadata.

    Args:
        spectrum: Spectrum dictionary.

    Returns:
        Polarity string or None if not found.
    """
    # pyteomics uses various keys for polarity
    for key in ["positive scan", "negative scan", "polarity"]:
        if key in spectrum:
            value = spectrum.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, bool) and value:
                if "positive" in key:
                    return "positive"
                if "negative" in key:
                    return "negative"
    return None


def _extract_precursor_info(spectrum: SpectrumDictProtocol) -> PrecursorInfo | None:
    """Extract precursor information from MS2 spectrum.

    Args:
        spectrum: Spectrum dictionary.

    Returns:
        PrecursorInfo or None if not MS2.
    """
    if "precursorMz" not in spectrum and "precursor" not in spectrum:
        return None

    # Try direct precursorMz (mzXML style)
    precursor_mz = spectrum.get("precursorMz")
    if precursor_mz is not None:
        mz_val: float
        if isinstance(precursor_mz, list) and len(precursor_mz) > 0:
            first_item = precursor_mz[0]
            mz_val = float(first_item) if isinstance(first_item, (int, float)) else 0.0
        elif isinstance(precursor_mz, (int, float)):
            mz_val = float(precursor_mz)
        else:
            mz_val = 0.0

        return PrecursorInfo(
            mz=mz_val,
            charge=None,
            intensity=None,
            isolation_window=None,
        )

    # Try precursor list (mzML style)
    precursor_list = spectrum.get("precursor")
    if isinstance(precursor_list, list) and len(precursor_list) > 0:
        first_precursor = precursor_list[0]
        if isinstance(first_precursor, dict):
            selected_ions = first_precursor.get("selectedIons", [])
            if isinstance(selected_ions, list) and len(selected_ions) > 0:
                ion = selected_ions[0]
                if isinstance(ion, dict):
                    mz_value = ion.get("selected ion m/z", 0.0)
                    charge_value = ion.get("charge state")
                    intensity_value = ion.get("peak intensity")

                    mz_float = float(mz_value) if isinstance(mz_value, (int, float)) else 0.0
                    charge_int: int | None = (
                        int(charge_value) if isinstance(charge_value, (int, float)) else None
                    )
                    intensity_float: float | None = (
                        float(intensity_value)
                        if isinstance(intensity_value, (int, float))
                        else None
                    )

                    return PrecursorInfo(
                        mz=mz_float,
                        charge=charge_int,
                        intensity=intensity_float,
                        isolation_window=None,
                    )

    return None


def _spectrum_to_msspectrum(
    spectrum: SpectrumDictProtocol,
    source_path: str,
) -> MSSpectrum:
    """Convert pyteomics spectrum dict to MSSpectrum TypedDict.

    Args:
        spectrum: Spectrum dictionary from pyteomics.
        source_path: Path to source file.

    Returns:
        MSSpectrum TypedDict.
    """
    # Extract arrays
    mz_raw = _extract_array_from_spectrum(spectrum, "m/z array", source_path)
    intensity_raw = _extract_array_from_spectrum(spectrum, "intensity array", source_path)

    mz_values = _decode_mz_array(mz_raw)
    intensities = _decode_intensity_array(intensity_raw)

    # Extract metadata
    scan_id = spectrum.get("id")
    scan_str: str | int | None = scan_id if isinstance(scan_id, (str, int)) else None
    scan_number = _decode_scan_number(scan_str)

    rt_raw = spectrum.get("scanList")
    rt_value: float = 0.0
    if isinstance(rt_raw, dict):
        scans = rt_raw.get("scan")
        if isinstance(scans, list) and len(scans) > 0:
            first_scan = scans[0]
            if isinstance(first_scan, dict):
                # Try "scan start time" key
                scan_time = first_scan.get("scan start time")
                if isinstance(scan_time, (int, float)):
                    rt_value = float(scan_time)
    else:
        # Try direct retention time
        direct_rt = spectrum.get("retentionTime")
        if isinstance(direct_rt, (int, float)):
            rt_value = float(direct_rt)

    retention_time = _decode_retention_time(rt_value)

    # MS level
    ms_level_raw_ml = spectrum.get("ms level")
    ms_level_val_raw: int | None = None
    if isinstance(ms_level_raw_ml, int):
        ms_level_val_raw = ms_level_raw_ml
    else:
        # mzXML uses 'msLevel' (camelCase)
        ms_level_raw_xml = spectrum.get("msLevel")
        if isinstance(ms_level_raw_xml, int):
            ms_level_val_raw = ms_level_raw_xml
    ms_level = _decode_ms_level(ms_level_val_raw)

    # Polarity
    polarity_str = _extract_polarity_string(spectrum)
    polarity = _decode_polarity(polarity_str)

    # TIC
    tic = _extract_float_or_zero(spectrum, "total ion current")

    # Build structures
    meta = SpectrumMeta(
        source_path=source_path,
        scan_number=scan_number,
        retention_time=retention_time,
        ms_level=ms_level,
        polarity=polarity,
        total_ion_current=tic,
    )

    data = _make_spectrum_data(mz_values, intensities)
    stats = _compute_spectrum_stats(mz_values, intensities)

    return MSSpectrum(meta=meta, data=data, stats=stats)


def _spectrum_to_ms2spectrum(
    spectrum: SpectrumDictProtocol,
    source_path: str,
) -> MS2Spectrum:
    """Convert pyteomics spectrum dict to MS2Spectrum TypedDict.

    Args:
        spectrum: Spectrum dictionary from pyteomics.
        source_path: Path to source file.

    Returns:
        MS2Spectrum TypedDict.

    Raises:
        MzMLReadError: If precursor info not found.
    """
    precursor = _extract_precursor_info(spectrum)
    if precursor is None:
        raise MzMLReadError(source_path, "No precursor info found for MS2 spectrum")

    # Get base spectrum data
    ms_spectrum = _spectrum_to_msspectrum(spectrum, source_path)

    return MS2Spectrum(
        meta=ms_spectrum["meta"],
        precursor=precursor,
        data=ms_spectrum["data"],
        stats=ms_spectrum["stats"],
    )


class MzMLReader:
    """Reader for mzML and mzXML mass spectrometry files.

    Provides typed access to spectrum data via pyteomics.
    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is an mzML or mzXML file.

        Args:
            path: Path to check.

        Returns:
            True if path is mzML or mzXML.
        """
        return _is_mzml_file(path) or _is_mzxml_file(path)

    def read_spectrum(self, path: Path, scan_number: int) -> MSSpectrum:
        """Read a single spectrum by scan number.

        Note: This iterates through the file to find the scan.
        For random access to many spectra, consider caching.

        Args:
            path: Path to mzML/mzXML file.
            scan_number: 1-based scan number.

        Returns:
            MSSpectrum TypedDict.

        Raises:
            MzMLReadError: If scan not found or reading fails.
        """
        for idx, spectrum in enumerate(self.iter_spectra(path), start=1):
            if idx == scan_number:
                return spectrum

        raise MzMLReadError(
            str(path),
            f"Scan number {scan_number} not found",
        )

    def iter_spectra(self, path: Path) -> Generator[MSSpectrum, None, None]:
        """Iterate over all spectra in file.

        Args:
            path: Path to mzML/mzXML file.

        Yields:
            MSSpectrum TypedDict for each spectrum.

        Raises:
            MzMLReadError: If reading fails.
        """
        source_path = str(path)

        if _is_mzml_file(path):
            reader: MzMLReaderProtocol = _open_mzml(path)
            with reader:
                for spectrum in reader:
                    yield _spectrum_to_msspectrum(spectrum, source_path)

        elif _is_mzxml_file(path):
            reader_xml: MzXMLReaderProtocol = _open_mzxml(path)
            with reader_xml:
                for spectrum in reader_xml:
                    yield _spectrum_to_msspectrum(spectrum, source_path)

        else:
            raise MzMLReadError(
                source_path,
                "Unsupported format (expected .mzML or .mzXML)",
            )

    def iter_ms2_spectra(self, path: Path) -> Generator[MS2Spectrum, None, None]:
        """Iterate over MS2 spectra only.

        Args:
            path: Path to mzML/mzXML file.

        Yields:
            MS2Spectrum TypedDict for each MS2 spectrum.

        Raises:
            MzMLReadError: If reading fails.
        """
        source_path = str(path)

        if _is_mzml_file(path):
            reader: MzMLReaderProtocol = _open_mzml(path)
            with reader:
                for spectrum in reader:
                    ms_level_raw = spectrum.get("ms level")
                    if isinstance(ms_level_raw, int) and ms_level_raw == 2:
                        yield _spectrum_to_ms2spectrum(spectrum, source_path)

        elif _is_mzxml_file(path):
            reader_xml: MzXMLReaderProtocol = _open_mzxml(path)
            with reader_xml:
                for spectrum in reader_xml:
                    ms_level_raw = spectrum.get("msLevel")
                    if isinstance(ms_level_raw, int) and ms_level_raw == 2:
                        yield _spectrum_to_ms2spectrum(spectrum, source_path)

        else:
            raise MzMLReadError(
                source_path,
                "Unsupported format (expected .mzML or .mzXML)",
            )

    def count_spectra(self, path: Path) -> int:
        """Count total number of spectra in file.

        Args:
            path: Path to mzML/mzXML file.

        Returns:
            Total spectrum count.

        Raises:
            MzMLReadError: If reading fails.
        """
        count = 0
        for _ in self.iter_spectra(path):
            count += 1
        return count

    def read_tic(self, path: Path) -> TICData:
        """Read Total Ion Chromatogram from mzML/mzXML file.

        Computes TIC by extracting total ion current from each spectrum.
        If total_ion_current is not available, sums all intensities.

        Args:
            path: Path to mzML/mzXML file.

        Returns:
            TICData TypedDict with complete chromatogram.

        Raises:
            MzMLReadError: If reading fails or no spectra found.
        """
        source_path = str(path)

        if not self.supports_format(path):
            raise MzMLReadError(source_path, "Unsupported format")

        retention_times: list[float] = []
        intensities: list[float] = []

        for spectrum in self.iter_spectra(path):
            rt = spectrum["meta"]["retention_time"]
            tic = spectrum["meta"]["total_ion_current"]

            # If TIC is zero, compute from intensities
            if tic == 0.0:
                tic = sum(spectrum["data"]["intensities"])

            retention_times.append(rt)
            intensities.append(tic)

        if not retention_times:
            raise MzMLReadError(source_path, "No spectra found in file")

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

        Sums intensities within m/z window for each spectrum.

        Args:
            path: Path to mzML/mzXML file.
            target_mz: Target m/z value.
            mz_tolerance: Tolerance window in Daltons (±).

        Returns:
            EICData TypedDict with extracted chromatogram.

        Raises:
            MzMLReadError: If reading fails or no spectra found.
        """
        source_path = str(path)

        if not self.supports_format(path):
            raise MzMLReadError(source_path, "Unsupported format")

        mz_low = target_mz - mz_tolerance
        mz_high = target_mz + mz_tolerance

        retention_times: list[float] = []
        intensities: list[float] = []

        for spectrum in self.iter_spectra(path):
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

        if not retention_times:
            raise MzMLReadError(source_path, "No spectra found in file")

        meta = ChromatogramMeta(
            source_path=source_path,
            instrument="",
            method_name="",
            sample_name="",
            acquisition_date="",
            signal_type="EIC",
            detector="MS",
        )
        data = ChromatogramData(
            retention_times=retention_times,
            intensities=intensities,
        )
        stats = _compute_chromatogram_stats(retention_times, intensities)
        params = EICParams(
            target_mz=target_mz,
            mz_tolerance=mz_tolerance,
        )

        return EICData(meta=meta, params=params, data=data, stats=stats)


__all__ = [
    "MzMLReader",
]
