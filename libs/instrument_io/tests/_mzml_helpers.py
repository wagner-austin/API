"""Shared test helpers for the mzML reader tests.

``SpectrumDict`` stands in for a pyteomics spectrum mapping across the mzML
test modules. It lives here rather than being redefined per module so the four
test files that drive the reader share one definition.
"""

from __future__ import annotations

from collections.abc import Generator

from instrument_io._protocols.pyteomics import SpectrumValue


class SpectrumDict:
    """Test helper implementing SpectrumDictProtocol."""

    def __init__(self, data: dict[str, SpectrumValue]) -> None:
        """Store the backing mapping.

        Args:
            data: Spectrum fields keyed by name.
        """
        self._data = data

    def __getitem__(self, key: str) -> SpectrumValue:
        """Return one field.

        Args:
            key: Field name.

        Returns:
            The field's value.
        """
        return self._data[key]

    def get(self, key: str) -> SpectrumValue:
        """Return one field, or None when absent.

        Args:
            key: Field name.

        Returns:
            The field's value, or None.
        """
        return self._data.get(key)

    def keys(self) -> Generator[str, None, None]:
        """Yield every field name.

        Yields:
            Each field name in insertion order.
        """
        yield from self._data.keys()

    def __contains__(self, key: str) -> bool:
        """Report whether a field is present.

        Args:
            key: Field name.

        Returns:
            True when the field is present.
        """
        return key in self._data


__all__ = ["SpectrumDict"]
