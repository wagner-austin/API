"""Tests for the dataset file vocabularies FileFormat and FileEncoding."""

from __future__ import annotations

from covenant_ml.datasets.types import FileEncoding, FileFormat


def test_file_format_members_are_their_wire_words() -> None:
    """Each format member equals the word a registry entry names it by."""
    assert [str(m) for m in FileFormat] == ["csv", "arff", "excel"]


def test_file_encoding_members_are_their_wire_words() -> None:
    """Each encoding member equals the codec name a registry entry carries."""
    assert [str(m) for m in FileEncoding] == ["utf-8", "utf-8-sig", "latin-1", "cp1252"]


def test_utf8_encodings_read_as_strict_polars_utf8() -> None:
    """Both UTF-8 spellings read strictly; polars strips a BOM itself."""
    assert FileEncoding.UTF_8.polars_encoding == "utf8"
    assert FileEncoding.UTF_8_SIG.polars_encoding == "utf8"


def test_single_byte_encodings_read_as_lossy_utf8() -> None:
    """Polars has no single-byte codecs, so those files read lossily."""
    assert FileEncoding.LATIN_1.polars_encoding == "utf8-lossy"
    assert FileEncoding.CP1252.polars_encoding == "utf8-lossy"
