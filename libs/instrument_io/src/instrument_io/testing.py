"""Test hooks for instrument_io library.

This module provides hooks for testing without mocking or monkeypatching.
Production code calls hooks directly; tests set hooks to fakes.

Usage:
    from instrument_io.testing import hooks, reset_hooks

    # In tests:
    def test_something() -> None:
        hooks.load_data_directory = _fake_loader
        # ... test code ...

    # Use reset_hooks() in conftest.py fixtures to restore defaults.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Protocol

# Import types needed for Protocol matching (these are Protocol classes, not heavy dependencies)
from instrument_io._protocols.imzml import ImzMLParserProtocol
from instrument_io._protocols.pdfplumber import PDFProtocol
from instrument_io._protocols.rainbow import DataDirectoryProtocol
from instrument_io.types.spectrum import MSSpectrum

# ---------------------------------------------------------------------------
# Protocol for MzMLReader (to avoid circular imports)
# ---------------------------------------------------------------------------


class MzMLReaderProtocol(Protocol):
    """Protocol for MzMLReader to enable hook typing."""

    def iter_spectra(self, path: Path) -> Generator[MSSpectrum, None, None]:
        """Iterate over spectra."""
        ...

    def read_spectrum(self, path: Path, scan_number: int) -> MSSpectrum:
        """Read spectrum by scan number."""
        ...

    def count_spectra(self, path: Path) -> int:
        """Count spectra in file."""
        ...


# ---------------------------------------------------------------------------
# Type aliases for hooks
# ---------------------------------------------------------------------------

# Thermo hooks
CreateTempDirFn = Callable[[], Path]
CleanupTempDirFn = Callable[[Path], None]
ConvertRawToMzmlFn = Callable[[Path, Path], Path]
GetBundledExePathFn = Callable[[], Path]
FindThermoRawFileParserFn = Callable[[], Path]
ShutilWhichFn = Callable[[str], str | None]
MzMLReaderFactory = Callable[[], MzMLReaderProtocol]

# shutil.which hook
ShutilWhichHookFn = Callable[[str], str | None]

# Rainbow/Waters hooks
LoadDataDirectoryFn = Callable[[Path], DataDirectoryProtocol]

# ImzML hooks
OpenImzmlFn = Callable[[Path], ImzMLParserProtocol]

# PDF hooks
OpenPdfFn = Callable[[Path], PDFProtocol]

# SMPS hooks
SMPSReadLinesFn = Callable[[Path], list[str]]

# TXT hooks
TXTDetectEncodingFn = Callable[[Path], str]
TXTReadTextFn = Callable[[Path, str], str]
TXTReadLinesFn = Callable[[Path, str], list[str]]


# ---------------------------------------------------------------------------
# Hooks container
# ---------------------------------------------------------------------------


class _HooksContainer:
    """Container for all hookable functions.

    Hooks are set to production implementations at module load time.
    Tests override hooks to use fakes.
    """

    # Thermo hooks
    create_temp_dir: CreateTempDirFn
    cleanup_temp_dir: CleanupTempDirFn
    convert_raw_to_mzml: ConvertRawToMzmlFn
    get_bundled_exe_path: GetBundledExePathFn
    find_thermorawfileparser: FindThermoRawFileParserFn
    shutil_which: ShutilWhichHookFn
    mzml_reader_factory: MzMLReaderFactory

    # Rainbow/Waters hooks
    load_data_directory: LoadDataDirectoryFn

    # ImzML hooks
    open_imzml: OpenImzmlFn

    # PDF hooks
    open_pdf: OpenPdfFn

    # SMPS hooks
    smps_read_lines: SMPSReadLinesFn

    # TXT hooks
    txt_detect_encoding: TXTDetectEncodingFn
    txt_read_text: TXTReadTextFn
    txt_read_lines: TXTReadLinesFn

    def reset(self) -> None:
        """Restore every hook to its production implementation.

        The restoration `reset_hooks()` performs, exposed as a method so an
        autouse fixture can name the container it protects.
        """
        reset_hooks()


hooks = _HooksContainer()


# ---------------------------------------------------------------------------
# Production implementations (wrappers that call real modules)
# ---------------------------------------------------------------------------


def _prod_create_temp_dir() -> Path:
    """Production implementation: create temp directory."""
    from instrument_io._protocols.thermo import _create_temp_dir

    return _create_temp_dir()


def _prod_cleanup_temp_dir(temp_dir: Path) -> None:
    """Production implementation: cleanup temp directory."""
    from instrument_io._protocols.thermo import _cleanup_temp_dir

    _cleanup_temp_dir(temp_dir)


def _prod_convert_raw_to_mzml(raw_path: Path, output_dir: Path) -> Path:
    """Production implementation: convert raw to mzML."""
    from instrument_io._protocols.thermo import _convert_raw_to_mzml

    return _convert_raw_to_mzml(raw_path, output_dir)


def _prod_get_bundled_exe_path() -> Path:
    """Production implementation: get bundled exe path."""
    from instrument_io._protocols.thermo import _get_bundled_exe_path

    return _get_bundled_exe_path()


def _prod_find_thermorawfileparser() -> Path:
    """Production implementation: find ThermoRawFileParser."""
    from instrument_io._protocols.thermo import _find_thermorawfileparser

    return _find_thermorawfileparser()


def _prod_shutil_which(cmd: str) -> str | None:
    """Production implementation: call shutil.which."""
    import shutil

    return shutil.which(cmd)


def _prod_mzml_reader_factory() -> MzMLReaderProtocol:
    """Production implementation: create MzMLReader."""
    from instrument_io.readers.mzml import MzMLReader

    return MzMLReader()


def _prod_load_data_directory(path: Path) -> DataDirectoryProtocol:
    """Production implementation: load Waters data directory via rainbow."""
    from instrument_io._protocols.rainbow import _load_data_directory

    return _load_data_directory(path)


def _prod_open_imzml(path: Path) -> ImzMLParserProtocol:
    """Production implementation: open imzML file."""
    from instrument_io._protocols.imzml import _open_imzml

    return _open_imzml(path)


def _prod_open_pdf(path: Path) -> PDFProtocol:
    """Production implementation: open PDF file."""
    from instrument_io._protocols.pdfplumber import _open_pdf

    return _open_pdf(path)


def _prod_smps_read_lines(path: Path) -> list[str]:
    """Production implementation: read SMPS file lines."""
    from instrument_io._exceptions import SMPSReadError

    try:
        with path.open("r", encoding="utf-8") as f:
            return [line.rstrip("\r\n") for line in f]
    except UnicodeDecodeError:
        try:
            with path.open("r", encoding="cp1252") as f:
                return [line.rstrip("\r\n") for line in f]
        except (UnicodeDecodeError, OSError) as e:
            raise SMPSReadError(str(path), f"Failed to read file: {e}") from e
    except OSError as e:
        raise SMPSReadError(str(path), f"Failed to read file: {e}") from e


def _prod_txt_detect_encoding(path: Path) -> str:
    """Production implementation: detect text file encoding."""
    preferred_encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252"]

    for encoding in preferred_encodings:
        try:
            with path.open("r", encoding=encoding) as f:
                f.read()
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue

    return "latin-1"


def _prod_txt_read_text(path: Path, encoding: str) -> str:
    """Production implementation: read text file content."""
    from instrument_io._exceptions import TXTReadError

    try:
        with path.open("r", encoding=encoding) as f:
            return f.read()
    except OSError as e:
        raise TXTReadError(str(path), f"Failed to read file: {e}") from e


def _prod_txt_read_lines(path: Path, encoding: str) -> list[str]:
    """Production implementation: read text file lines."""
    from instrument_io._exceptions import TXTReadError

    try:
        with path.open("r", encoding=encoding) as f:
            return [line.rstrip("\r\n") for line in f]
    except OSError as e:
        raise TXTReadError(str(path), f"Failed to read file: {e}") from e


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _init_production_hooks() -> None:
    """Initialize hooks to production implementations.

    Called at module load time and by reset_hooks().
    """
    hooks.create_temp_dir = _prod_create_temp_dir
    hooks.cleanup_temp_dir = _prod_cleanup_temp_dir
    hooks.convert_raw_to_mzml = _prod_convert_raw_to_mzml
    hooks.get_bundled_exe_path = _prod_get_bundled_exe_path
    hooks.find_thermorawfileparser = _prod_find_thermorawfileparser
    hooks.shutil_which = _prod_shutil_which
    hooks.mzml_reader_factory = _prod_mzml_reader_factory
    hooks.load_data_directory = _prod_load_data_directory
    hooks.open_imzml = _prod_open_imzml
    hooks.open_pdf = _prod_open_pdf
    hooks.smps_read_lines = _prod_smps_read_lines
    hooks.txt_detect_encoding = _prod_txt_detect_encoding
    hooks.txt_read_text = _prod_txt_read_text
    hooks.txt_read_lines = _prod_txt_read_lines


def reset_hooks() -> None:
    """Reset all hooks to production implementations.

    Use in conftest.py autouse fixture for test isolation.
    """
    _init_production_hooks()


# Initialize hooks to production implementations at module load
_init_production_hooks()


# ---------------------------------------------------------------------------
# Fake implementations for tests
# ---------------------------------------------------------------------------


__all__ = [
    "MzMLReaderProtocol",
    "_prod_cleanup_temp_dir",
    "_prod_convert_raw_to_mzml",
    "_prod_create_temp_dir",
    "_prod_find_thermorawfileparser",
    "_prod_get_bundled_exe_path",
    "_prod_load_data_directory",
    "_prod_mzml_reader_factory",
    "_prod_open_imzml",
    "_prod_open_pdf",
    "_prod_shutil_which",
    "_prod_smps_read_lines",
    "_prod_txt_detect_encoding",
    "_prod_txt_read_lines",
    "_prod_txt_read_text",
    "hooks",
    "reset_hooks",
]
