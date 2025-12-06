"""MATLAB data file (.mat) reader implementation.

Provides typed reading of MATLAB .mat files via scipy.io.
Uses Protocol-based dynamic imports for external libraries.
"""

from __future__ import annotations

from pathlib import Path

from instrument_io._decoders.mat import _decode_mat_dict, _is_metadata_key
from instrument_io._exceptions import MATReadError
from instrument_io._protocols.scipy_io import PythonValue, _load_mat


def _is_mat_file(path: Path) -> bool:
    """Check if path is a MATLAB data file."""
    return path.is_file() and path.suffix.lower() == ".mat"


class MATReader:
    """Reader for MATLAB data files (.mat).

    Provides typed access to MATLAB variables and metadata.
    Uses scipy.io.loadmat for reading with Protocol-based typing for strict type safety.

    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is a MATLAB data file.

        Args:
            path: Path to check.

        Returns:
            True if path is a MATLAB data file (.mat).
        """
        return _is_mat_file(path)

    def list_variables(self, path: Path) -> list[str]:
        """List all variable names in .mat file.

        Excludes MATLAB metadata variables (names starting/ending with __).

        Args:
            path: Path to .mat file.

        Returns:
            List of variable names.

        Raises:
            MATReadError: If reading fails.
        """
        if not path.exists():
            raise MATReadError(str(path), "File does not exist")

        if not _is_mat_file(path):
            raise MATReadError(str(path), "Not a MATLAB data file")

        mat_dict = _load_mat(path)
        return [key for key in mat_dict if not _is_metadata_key(key)]

    def read_variable(
        self,
        path: Path,
        var_name: str,
    ) -> PythonValue | str | int | float:
        """Read a single variable from .mat file.

        Args:
            path: Path to .mat file.
            var_name: Name of variable to read.

        Returns:
            Variable value (numpy arrays are converted to Python lists).

        Raises:
            MATReadError: If reading fails or variable not found.
        """
        if not path.exists():
            raise MATReadError(str(path), "File does not exist")

        if not _is_mat_file(path):
            raise MATReadError(str(path), "Not a MATLAB data file")

        mat_dict = _load_mat(path)
        decoded = _decode_mat_dict(mat_dict)

        if var_name not in decoded:
            available = ", ".join(decoded.keys())
            raise MATReadError(
                str(path),
                f"Variable '{var_name}' not found. Available: {available}",
            )

        return decoded[var_name]

    def read_all(
        self,
        path: Path,
    ) -> dict[str, PythonValue | str | int | float]:
        """Read all variables from .mat file.

        Excludes MATLAB metadata variables.

        Args:
            path: Path to .mat file.

        Returns:
            Dictionary mapping variable names to values.
            Numpy arrays are converted to Python lists.

        Raises:
            MATReadError: If reading fails.
        """
        if not path.exists():
            raise MATReadError(str(path), "File does not exist")

        if not _is_mat_file(path):
            raise MATReadError(str(path), "Not a MATLAB data file")

        mat_dict = _load_mat(path)
        return _decode_mat_dict(mat_dict)

    def get_metadata(self, path: Path) -> dict[str, str]:
        """Get MATLAB file metadata.

        Args:
            path: Path to .mat file.

        Returns:
            Dictionary with metadata keys (__header__, __version__, __globals__).

        Raises:
            MATReadError: If reading fails.
        """
        if not path.exists():
            raise MATReadError(str(path), "File does not exist")

        if not _is_mat_file(path):
            raise MATReadError(str(path), "Not a MATLAB data file")

        mat_dict = _load_mat(path)
        metadata: dict[str, str] = {}

        for key, value in mat_dict.items():
            if _is_metadata_key(key) and isinstance(value, (str, bytes)):
                if isinstance(value, bytes):
                    metadata[key] = value.decode("utf-8", errors="ignore")
                else:
                    metadata[key] = value

        return metadata


__all__ = [
    "MATReader",
]
