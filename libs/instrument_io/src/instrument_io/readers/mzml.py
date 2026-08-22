"""mzML/mzXML file reader implementation.

Provides typed reading of mass spectrometry data in mzML and mzXML formats
via pyteomics. Uses Protocol-based dynamic imports.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from instrument_io._exceptions import MzMLReadError
from instrument_io._protocols.pyteomics import (
    MzMLReaderProtocol,
    MzXMLReaderProtocol,
    _open_mzml,
    _open_mzxml,
)
from instrument_io.readers.mzml_spectrum import (
    _compute_chromatogram_stats,
    _is_mzml_file,
    _is_mzxml_file,
    _spectrum_to_ms2spectrum,
    _spectrum_to_msspectrum,
)
from instrument_io.types.chromatogram import (
    ChromatogramData,
    ChromatogramMeta,
    EICData,
    EICParams,
    TICData,
)
from instrument_io.types.spectrum import (
    MS2Spectrum,
    MSSpectrum,
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
