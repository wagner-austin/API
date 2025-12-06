"""MGF (Mascot Generic Format) file reader implementation.

Provides typed reading of MS/MS peak list data in MGF format
via pyteomics. Uses Protocol-based dynamic imports.

Note: MGF is a peak list format containing MS/MS spectra only.
It does not support TIC/EIC extraction as those require raw data.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from instrument_io._decoders.mgf import (
    _compute_mgf_spectrum_stats,
    _decode_mgf_polarity,
    _decode_mgf_precursor,
    _decode_mgf_retention_time,
    _decode_mgf_scan_number,
    _make_mgf_spectrum_data,
    _make_mgf_spectrum_meta,
)
from instrument_io._exceptions import MGFReadError
from instrument_io._protocols.mgf import (
    MGFParamsDict,
    MGFReaderProtocol,
    MGFSpectrumProtocol,
    _open_mgf,
)
from instrument_io._protocols.numpy import NdArray1DProtocol
from instrument_io.types.spectrum import MS2Spectrum


def _is_mgf_file(path: Path) -> bool:
    """Check if path is an MGF file."""
    return path.is_file() and path.suffix.lower() == ".mgf"


def _get_spectrum_arrays(
    spectrum: MGFSpectrumProtocol,
) -> tuple[list[float], list[float]]:
    """Extract m/z and intensity arrays from spectrum.

    Args:
        spectrum: Spectrum dictionary from pyteomics.

    Returns:
        Tuple of (mz_values, intensities) as lists.
    """
    mz_array: NdArray1DProtocol = spectrum["m/z array"]
    intensity_array: NdArray1DProtocol = spectrum["intensity array"]
    mz_values: list[float] = mz_array.tolist()
    intensities: list[float] = intensity_array.tolist()
    return mz_values, intensities


def _get_spectrum_params(
    spectrum: MGFSpectrumProtocol,
) -> MGFParamsDict:
    """Extract params dict from spectrum.

    Args:
        spectrum: Spectrum dictionary from pyteomics.

    Returns:
        Params dict.
    """
    params: MGFParamsDict = spectrum["params"]
    return params


def _spectrum_to_ms2spectrum(
    spectrum: MGFSpectrumProtocol,
    source_path: str,
    index: int,
) -> MS2Spectrum:
    """Convert pyteomics MGF spectrum dict to MS2Spectrum TypedDict.

    Args:
        spectrum: Spectrum dictionary from pyteomics.
        source_path: Path to source file.
        index: 0-based index of spectrum in file.

    Returns:
        MS2Spectrum TypedDict.
    """
    # Extract arrays and params
    mz_values, intensities = _get_spectrum_arrays(spectrum)
    params = _get_spectrum_params(spectrum)

    # Decode metadata
    scan_number = _decode_mgf_scan_number(params, index)
    retention_time = _decode_mgf_retention_time(params)
    polarity = _decode_mgf_polarity(params)
    total_ion_current = sum(intensities)

    # Decode precursor
    precursor = _decode_mgf_precursor(params)

    # Build structures
    meta = _make_mgf_spectrum_meta(
        source_path=source_path,
        scan_number=scan_number,
        retention_time=retention_time,
        polarity=polarity,
        total_ion_current=total_ion_current,
    )
    data = _make_mgf_spectrum_data(mz_values, intensities)
    stats = _compute_mgf_spectrum_stats(mz_values, intensities)

    return MS2Spectrum(meta=meta, precursor=precursor, data=data, stats=stats)


class MGFReader:
    """Reader for MGF (Mascot Generic Format) peak list files.

    Provides typed access to MS/MS spectrum data via pyteomics.
    All methods raise exceptions on failure - no recovery or fallbacks.

    Note: MGF is a peak list format containing pre-processed MS/MS spectra.
    It does not support read_tic() or read_eic() as those methods require
    raw chromatographic data, which MGF files do not contain.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is an MGF file.

        Args:
            path: Path to check.

        Returns:
            True if path is .mgf file.
        """
        return _is_mgf_file(path)

    def iter_spectra(self, path: Path) -> Generator[MS2Spectrum, None, None]:
        """Iterate over all MS/MS spectra in MGF file.

        Args:
            path: Path to .mgf file.

        Yields:
            MS2Spectrum TypedDict for each spectrum.

        Raises:
            MGFReadError: If reading fails.
        """
        source_path = str(path)

        if not _is_mgf_file(path):
            raise MGFReadError(source_path, "Not an MGF file")

        reader: MGFReaderProtocol = _open_mgf(path)
        with reader:
            for index, spectrum in enumerate(reader):
                yield _spectrum_to_ms2spectrum(spectrum, source_path, index)

    def read_spectrum(self, path: Path, index: int) -> MS2Spectrum:
        """Read a single spectrum by 0-based index.

        Note: MGF files are indexed by position, not scan number.

        Args:
            path: Path to .mgf file.
            index: 0-based index of spectrum in file.

        Returns:
            MS2Spectrum TypedDict.

        Raises:
            MGFReadError: If spectrum not found or reading fails.
        """
        source_path = str(path)

        if not _is_mgf_file(path):
            raise MGFReadError(source_path, "Not an MGF file")

        if index < 0:
            raise MGFReadError(source_path, f"Invalid index: {index}")

        for current_idx, spectrum in enumerate(self.iter_spectra(path)):
            if current_idx == index:
                return spectrum

        raise MGFReadError(source_path, f"Spectrum index {index} not found")

    def count_spectra(self, path: Path) -> int:
        """Count total number of spectra in MGF file.

        Args:
            path: Path to .mgf file.

        Returns:
            Total spectrum count.

        Raises:
            MGFReadError: If reading fails.
        """
        source_path = str(path)

        if not _is_mgf_file(path):
            raise MGFReadError(source_path, "Not an MGF file")

        count = 0
        for _ in self.iter_spectra(path):
            count += 1
        return count


__all__ = [
    "MGFReader",
]
