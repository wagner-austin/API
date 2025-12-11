"""Protocol and helpers for Thermo .raw file processing.

Uses ThermoRawFileParser CLI to convert .raw files to mzML,
then reads with MzMLReader for fully typed output.

ThermoRawFileParser: https://github.com/compomics/ThermoRawFileParser
- Cross-platform (Windows native, Linux/Mac via Mono)
- Official Thermo libraries under the hood
- Outputs standard mzML format
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def _get_bundled_exe_path() -> Path:
    """Get path to bundled ThermoRawFileParser executable.

    Returns:
        Path to bundled executable (may not exist).
    """
    package_root = Path(__file__).parent.parent.parent.parent
    return package_root / "tools" / "ThermoRawFileParser" / "ThermoRawFileParser.exe"


def _find_thermorawfileparser() -> Path:
    """Find ThermoRawFileParser executable.

    Searches common installation locations.

    Returns:
        Path to ThermoRawFileParser executable.

    Raises:
        FileNotFoundError: If executable not found.
    """
    from instrument_io.testing import hooks

    # Common locations to check (bundled first, then system locations)
    candidates: list[Path] = [
        # Bundled with package
        hooks.get_bundled_exe_path(),
        # Windows installed via dotnet tool
        Path.home() / ".dotnet" / "tools" / "ThermoRawFileParser.exe",
        # Linux/Mac via Mono - typically in PATH or /usr/local/bin
        Path("/usr/local/bin/ThermoRawFileParser"),
        Path("/usr/bin/ThermoRawFileParser"),
        # Windows Program Files
        Path("C:/Program Files/ThermoRawFileParser/ThermoRawFileParser.exe"),
        Path("C:/Program Files (x86)/ThermoRawFileParser/ThermoRawFileParser.exe"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Try to find in PATH via 'which' or 'where'
    # On Windows, shutil.which auto-appends PATHEXT extensions (including .exe)
    which_result = hooks.shutil_which("ThermoRawFileParser")
    if which_result is not None:
        return Path(which_result)

    msg = (
        "ThermoRawFileParser not found. Install via: "
        "dotnet tool install -g ThermoRawFileParser "
        "or download from https://github.com/compomics/ThermoRawFileParser"
    )
    raise FileNotFoundError(msg)


def _convert_raw_to_mzml(raw_path: Path, output_dir: Path) -> Path:
    """Convert Thermo .raw file to mzML using ThermoRawFileParser.

    Args:
        raw_path: Path to input .raw file.
        output_dir: Directory for output mzML file.

    Returns:
        Path to generated mzML file.

    Raises:
        FileNotFoundError: If ThermoRawFileParser not found.
        subprocess.CalledProcessError: If conversion fails.
    """
    from instrument_io._exceptions import ThermoReadError
    from instrument_io.testing import hooks

    parser_path = hooks.find_thermorawfileparser()

    # Build command
    # -i: input file
    # -o: output directory
    # -f: format (0 = mzML)
    cmd: list[str] = [
        str(parser_path),
        "-i",
        str(raw_path),
        "-o",
        str(output_dir),
        "-f",
        "1",  # mzML format
    ]

    # Run conversion
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise ThermoReadError(
            str(raw_path),
            f"ThermoRawFileParser failed: {result.stderr}",
        )

    # Output file has same stem as input with .mzML extension
    output_path = output_dir / f"{raw_path.stem}.mzML"

    if not output_path.exists():
        raise ThermoReadError(
            str(raw_path),
            f"Expected output file not found: {output_path}",
        )

    return output_path


def _create_temp_dir() -> Path:
    """Create a temporary directory for mzML output.

    Returns:
        Path to temporary directory.
    """
    return Path(tempfile.mkdtemp(prefix="thermo_"))


def _cleanup_temp_dir(temp_dir: Path) -> None:
    """Remove temporary directory and contents.

    Args:
        temp_dir: Temporary directory to remove.
    """
    import shutil

    if temp_dir.exists():
        shutil.rmtree(temp_dir)


__all__ = [
    "_cleanup_temp_dir",
    "_convert_raw_to_mzml",
    "_create_temp_dir",
    "_find_thermorawfileparser",
    "_get_bundled_exe_path",
]
