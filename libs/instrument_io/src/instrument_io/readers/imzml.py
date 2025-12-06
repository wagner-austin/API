"""imzML (Imaging Mass Spectrometry) file reader implementation.

Provides typed reading of imaging MS data in imzML format via pyimzML.
Uses Protocol-based dynamic imports.

Note: imzML contains mass spectra with spatial (x, y, z) coordinates.
Each spectrum represents data from a specific pixel location.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from instrument_io._decoders.imzml import (
    _compute_imzml_spectrum_stats,
    _decode_coordinate,
    _decode_imzml_polarity,
    _decode_spectrum_mode,
    _make_imzml_spectrum_data,
    _make_imzml_spectrum_meta,
)
from instrument_io._exceptions import ImzMLReadError
from instrument_io._protocols.imzml import (
    ImzMLParserProtocol,
    _open_imzml,
)
from instrument_io._protocols.numpy import NdArray1DProtocol
from instrument_io.types.imaging import (
    ImagingSpectrum,
    ImzMLFileInfo,
    SpatialCoordinate,
)


def _is_imzml_file(path: Path) -> bool:
    """Check if path is an imzML file."""
    return path.is_file() and path.suffix.lower() == ".imzml"


def _get_spectrum_arrays(
    parser: ImzMLParserProtocol,
    index: int,
) -> tuple[list[float], list[float]]:
    """Extract m/z and intensity arrays from spectrum.

    Args:
        parser: ImzML parser instance.
        index: 0-based spectrum index.

    Returns:
        Tuple of (mz_values, intensities) as lists.
    """
    mz_array: NdArray1DProtocol
    intensity_array: NdArray1DProtocol
    mz_array, intensity_array = parser.getspectrum(index)
    mz_values: list[float] = mz_array.tolist()
    intensities: list[float] = intensity_array.tolist()
    return mz_values, intensities


def _spectrum_to_imaging_spectrum(
    parser: ImzMLParserProtocol,
    source_path: str,
    index: int,
    polarity_str: str,
) -> ImagingSpectrum:
    """Convert pyimzML spectrum to ImagingSpectrum TypedDict.

    Args:
        parser: ImzML parser instance.
        source_path: Path to source file.
        index: 0-based index of spectrum.
        polarity_str: Polarity string from parser.

    Returns:
        ImagingSpectrum TypedDict.
    """
    # Get arrays
    mz_values, intensities = _get_spectrum_arrays(parser, index)

    # Get coordinate for this spectrum
    coord_tuple = parser.coordinates[index]
    coordinate = _decode_coordinate(coord_tuple)

    # Decode polarity
    polarity = _decode_imzml_polarity(polarity_str)

    # Compute total ion current
    total_ion_current = sum(intensities)

    # Build structures
    meta = _make_imzml_spectrum_meta(
        source_path=source_path,
        index=index,
        coordinate=coordinate,
        polarity=polarity,
        total_ion_current=total_ion_current,
    )
    data = _make_imzml_spectrum_data(mz_values, intensities)
    stats = _compute_imzml_spectrum_stats(mz_values, intensities)

    return ImagingSpectrum(meta=meta, data=data, stats=stats)


def _compute_image_dimensions(
    coordinates: list[tuple[int, int, int]],
) -> tuple[int, int]:
    """Compute image dimensions from coordinate list.

    Args:
        coordinates: List of (x, y, z) tuples.

    Returns:
        Tuple of (x_pixels, y_pixels).
    """
    if not coordinates:
        return 0, 0
    x_values = [c[0] for c in coordinates]
    y_values = [c[1] for c in coordinates]
    return max(x_values), max(y_values)


class ImzMLReader:
    """Reader for imzML (Imaging Mass Spectrometry) files.

    Provides typed access to imaging MS data via pyimzML.
    All methods raise exceptions on failure - no recovery or fallbacks.

    Note: imzML files contain spectra with spatial coordinates.
    Use get_coordinates() to get the list of pixel positions,
    and read_spectrum_at_coordinate() to access by position.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is an imzML file.

        Args:
            path: Path to check.

        Returns:
            True if path is .imzML file.
        """
        return _is_imzml_file(path)

    def get_file_info(self, path: Path) -> ImzMLFileInfo:
        """Get file-level metadata.

        Args:
            path: Path to .imzML file.

        Returns:
            ImzMLFileInfo TypedDict.

        Raises:
            ImzMLReadError: If reading fails.
        """
        source_path = str(path)

        if not _is_imzml_file(path):
            raise ImzMLReadError(source_path, "Not an imzML file")

        parser: ImzMLParserProtocol = _open_imzml(path)
        with parser:
            num_spectra = len(parser.coordinates)
            polarity = _decode_imzml_polarity(parser.polarity)
            spectrum_mode = _decode_spectrum_mode(parser.spectrum_mode)
            x_pixels, y_pixels = _compute_image_dimensions(parser.coordinates)

        return ImzMLFileInfo(
            source_path=source_path,
            num_spectra=num_spectra,
            polarity=polarity,
            spectrum_mode=spectrum_mode,
            x_pixels=x_pixels,
            y_pixels=y_pixels,
        )

    def get_coordinates(self, path: Path) -> list[SpatialCoordinate]:
        """Get list of all pixel coordinates.

        Args:
            path: Path to .imzML file.

        Returns:
            List of SpatialCoordinate TypedDicts.

        Raises:
            ImzMLReadError: If reading fails.
        """
        source_path = str(path)

        if not _is_imzml_file(path):
            raise ImzMLReadError(source_path, "Not an imzML file")

        parser: ImzMLParserProtocol = _open_imzml(path)
        with parser:
            return [_decode_coordinate(c) for c in parser.coordinates]

    def iter_spectra(self, path: Path) -> Generator[ImagingSpectrum, None, None]:
        """Iterate over all spectra in imzML file.

        Args:
            path: Path to .imzML file.

        Yields:
            ImagingSpectrum TypedDict for each spectrum.

        Raises:
            ImzMLReadError: If reading fails.
        """
        source_path = str(path)

        if not _is_imzml_file(path):
            raise ImzMLReadError(source_path, "Not an imzML file")

        parser: ImzMLParserProtocol = _open_imzml(path)
        with parser:
            polarity_str = parser.polarity
            for index in range(len(parser.coordinates)):
                yield _spectrum_to_imaging_spectrum(parser, source_path, index, polarity_str)

    def read_spectrum(self, path: Path, index: int) -> ImagingSpectrum:
        """Read a single spectrum by 0-based index.

        Args:
            path: Path to .imzML file.
            index: 0-based index of spectrum.

        Returns:
            ImagingSpectrum TypedDict.

        Raises:
            ImzMLReadError: If spectrum not found or reading fails.
        """
        source_path = str(path)

        if not _is_imzml_file(path):
            raise ImzMLReadError(source_path, "Not an imzML file")

        if index < 0:
            raise ImzMLReadError(source_path, f"Invalid index: {index}")

        parser: ImzMLParserProtocol = _open_imzml(path)
        with parser:
            num_spectra = len(parser.coordinates)
            if index >= num_spectra:
                raise ImzMLReadError(source_path, f"Spectrum index {index} not found")
            return _spectrum_to_imaging_spectrum(parser, source_path, index, parser.polarity)

    def read_spectrum_at_coordinate(
        self, path: Path, x: int, y: int, z: int = 1
    ) -> ImagingSpectrum:
        """Read spectrum at specific spatial coordinate.

        Args:
            path: Path to .imzML file.
            x: X coordinate (1-based).
            y: Y coordinate (1-based).
            z: Z coordinate (1-based, default 1 for 2D).

        Returns:
            ImagingSpectrum TypedDict.

        Raises:
            ImzMLReadError: If coordinate not found or reading fails.
        """
        source_path = str(path)

        if not _is_imzml_file(path):
            raise ImzMLReadError(source_path, "Not an imzML file")

        parser: ImzMLParserProtocol = _open_imzml(path)
        with parser:
            # Find index for coordinate
            target = (x, y, z)
            for index, coord in enumerate(parser.coordinates):
                if coord == target:
                    return _spectrum_to_imaging_spectrum(
                        parser, source_path, index, parser.polarity
                    )

        raise ImzMLReadError(source_path, f"Coordinate ({x}, {y}, {z}) not found")

    def count_spectra(self, path: Path) -> int:
        """Count total number of spectra in imzML file.

        Args:
            path: Path to .imzML file.

        Returns:
            Total spectrum count.

        Raises:
            ImzMLReadError: If reading fails.
        """
        source_path = str(path)

        if not _is_imzml_file(path):
            raise ImzMLReadError(source_path, "Not an imzML file")

        parser: ImzMLParserProtocol = _open_imzml(path)
        with parser:
            return len(parser.coordinates)


__all__ = [
    "ImzMLReader",
]
