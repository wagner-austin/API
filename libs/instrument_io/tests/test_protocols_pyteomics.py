"""Tests for _protocols.pyteomics module."""

from __future__ import annotations

import warnings
from pathlib import Path

from instrument_io._protocols.pyteomics import (
    _open_mzml,
    _open_mzxml,
)


def _suppress_pyteomics_iterator_warning() -> None:
    """Suppress pyteomics non-indexed iterator warning.

    This warning is expected when iterating over minimal test mzML/mzXML files
    that don't have index offsets. It's from the pyteomics library internals.
    """
    warnings.filterwarnings(
        "ignore",
        message="Non-indexed iterator created from.*",
        category=UserWarning,
        module="pyteomics.xml",
    )


class TestOpenMzml:
    """Tests for _open_mzml function."""

    def test_open_mzml_returns_reader_protocol(self, tmp_path: Path) -> None:
        _suppress_pyteomics_iterator_warning()

        # Create a minimal mzML file for testing
        mzml_content = """<?xml version="1.0" encoding="UTF-8"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
  <mzML xmlns="http://psi.hupo.org/ms/mzml">
    <run id="test">
      <spectrumList count="0" defaultDataProcessingRef="dp1">
      </spectrumList>
    </run>
  </mzML>
</indexedmzML>"""
        mzml_file = tmp_path / "test.mzML"
        mzml_file.write_text(mzml_content)

        reader = _open_mzml(mzml_file)
        # Reader should be usable as context manager
        with reader:
            # Should be iterable
            spectra = list(reader)
            assert spectra == []


class TestOpenMzxml:
    """Tests for _open_mzxml function."""

    def test_open_mzxml_returns_reader_protocol(self, tmp_path: Path) -> None:
        _suppress_pyteomics_iterator_warning()

        # Create a minimal mzXML file for testing
        mzxml_content = """<?xml version="1.0" encoding="UTF-8"?>
<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2">
  <msRun scanCount="0">
  </msRun>
</mzXML>"""
        mzxml_file = tmp_path / "test.mzXML"
        mzxml_file.write_text(mzxml_content)

        reader = _open_mzxml(mzxml_file)
        # Reader should be usable as context manager
        with reader:
            # Should be iterable
            spectra = list(reader)
            assert spectra == []
