"""Tests for scripts extract standards: DeduplicateHeaders."""

from __future__ import annotations

from pathlib import Path

from scripts.extract_standards import (
    StandardsExtractor,
    _deduplicate_headers,
)

from instrument_io._protocols.openpyxl import _load_workbook


class TestDeduplicateHeaders:
    """Tests for _deduplicate_headers function."""

    def test_unique_headers_unchanged(self) -> None:
        """Test unique headers remain unchanged."""
        headers = ["Name", "CAS", "Amount"]
        result = _deduplicate_headers(headers)
        assert result == ["Name", "CAS", "Amount"]

    def test_duplicate_headers_get_suffix(self) -> None:
        """Test duplicate headers get numeric suffix."""
        headers = ["Name", "Value", "Value", "Name"]
        result = _deduplicate_headers(headers)
        assert result == ["Name", "Value", "Value_1", "Name_1"]

    def test_multiple_duplicates(self) -> None:
        """Test multiple duplicates of same header."""
        headers = ["Col", "Col", "Col", "Col"]
        result = _deduplicate_headers(headers)
        assert result == ["Col", "Col_1", "Col_2", "Col_3"]

    def test_empty_list(self) -> None:
        """Test with empty list."""
        result = _deduplicate_headers([])
        assert result == []


class TestStandardsExtractorValidation:
    """Tests for StandardsExtractor validation methods."""

    def test_is_valid_chemical_name_valid(self) -> None:
        """Test valid chemical names."""
        extractor = StandardsExtractor()
        assert extractor._is_valid_chemical_name("alpha-Pinene") is True
        assert extractor._is_valid_chemical_name("Limonene") is True
        assert extractor._is_valid_chemical_name("Acetone") is True
        assert extractor._is_valid_chemical_name("beta-Caryophyllene") is True

    def test_is_valid_chemical_name_too_short(self) -> None:
        """Test names that are too short."""
        extractor = StandardsExtractor()
        assert extractor._is_valid_chemical_name("") is False
        assert extractor._is_valid_chemical_name("ab") is False

    def test_is_valid_chemical_name_skip_exact(self) -> None:
        """Test names that match exact skip list."""
        extractor = StandardsExtractor()
        assert extractor._is_valid_chemical_name("null") is False
        assert extractor._is_valid_chemical_name("none") is False
        assert extractor._is_valid_chemical_name("total") is False
        assert extractor._is_valid_chemical_name("sample") is False

    def test_is_valid_chemical_name_skip_startswith(self) -> None:
        """Test names that start with skip prefixes."""
        extractor = StandardsExtractor()
        assert extractor._is_valid_chemical_name("Sample1") is False
        assert extractor._is_valid_chemical_name("MT1") is False
        assert extractor._is_valid_chemical_name("Unknown5") is False

    def test_is_valid_chemical_name_skip_contains(self) -> None:
        """Test names containing skip substrings (line 186)."""
        extractor = StandardsExtractor()
        # Test each skip_contains pattern
        assert extractor._is_valid_chemical_name("path\\data-ms\\file") is False
        assert extractor._is_valid_chemical_name("sample-d\\result") is False
        assert extractor._is_valid_chemical_name("compound injected here") is False
        assert extractor._is_valid_chemical_name("compound response factor test") is False
        assert extractor._is_valid_chemical_name("value and units") is False
        assert extractor._is_valid_chemical_name("Standard ran? (y/n) test") is False

    def test_is_valid_chemical_name_formula_pattern(self) -> None:
        """Test formula/equation strings are rejected (line 194)."""
        extractor = StandardsExtractor()
        # Pattern: *x + (equations like y = 2*x + 5)
        assert extractor._is_valid_chemical_name("y = 2*x + 5") is False
        assert extractor._is_valid_chemical_name("slope*x + intercept") is False
        assert extractor._is_valid_chemical_name("m*x  +  b") is False

    def test_is_valid_chemical_name_pure_numbers(self) -> None:
        """Test pure number strings are rejected."""
        extractor = StandardsExtractor()
        assert extractor._is_valid_chemical_name("12345") is False
        assert extractor._is_valid_chemical_name("-123") is False

    def test_is_valid_chemical_name_too_long(self) -> None:
        """Test names that are too long are rejected."""
        extractor = StandardsExtractor()
        long_name = "A" * 100
        assert extractor._is_valid_chemical_name(long_name) is False


class TestStandardsExtractorNormalization:
    """Tests for StandardsExtractor normalization methods."""

    def test_normalize_name_removes_spaces(self) -> None:
        """Test that normalize_name removes spaces."""
        extractor = StandardsExtractor()
        assert extractor._normalize_name("alpha pinene") == "alphapinene"

    def test_normalize_name_handles_greek_prefixes(self) -> None:
        """Test normalization of Greek letter prefixes."""
        extractor = StandardsExtractor()
        assert extractor._normalize_name("alpha-Pinene") == "alphapinene"
        assert extractor._normalize_name("beta-Myrcene") == "betamyrcene"

    def test_clean_display_name_alpha(self) -> None:
        """Test cleaning display name with alpha prefix."""
        extractor = StandardsExtractor()
        # Clean display name normalizes alpha- prefix
        result = extractor._clean_display_name("alpha-pinene")
        assert "Pinene" in result

    def test_clean_display_name_a_prefix(self) -> None:
        """Test cleaning display name with a- prefix (line 239)."""
        extractor = StandardsExtractor()
        result = extractor._clean_display_name("a-pinene")
        assert result == "α-Pinene"

    def test_clean_display_name_beta(self) -> None:
        """Test cleaning display name with beta prefix."""
        extractor = StandardsExtractor()
        result = extractor._clean_display_name("beta-myrcene")
        assert "Myrcene" in result

    def test_clean_display_name_b_prefix(self) -> None:
        """Test cleaning display name with b- prefix."""
        extractor = StandardsExtractor()
        result = extractor._clean_display_name("b-myrcene")
        assert result == "β-Myrcene"

    def test_clean_display_name_gamma(self) -> None:
        """Test cleaning display name with gamma prefix."""
        extractor = StandardsExtractor()
        result = extractor._clean_display_name("gamma-terpinene")
        assert "Terpinene" in result

    def test_clean_display_name_y_prefix(self) -> None:
        """Test cleaning display name with y- prefix (line 247)."""
        extractor = StandardsExtractor()
        result = extractor._clean_display_name("y-terpinene")
        assert result == "γ-Terpinene"

    def test_clean_display_name_capitalizes(self) -> None:
        """Test that first letter is capitalized."""
        extractor = StandardsExtractor()
        assert extractor._clean_display_name("limonene") == "Limonene"


class TestStandardsExtractorAddStandard:
    """Tests for StandardsExtractor.add_standard method."""

    def test_add_valid_standard(self) -> None:
        """Test adding a valid standard."""
        extractor = StandardsExtractor()
        result = extractor.add_standard(
            "Limonene", "Test Source", "2025-01-01", "Standard", "Test details"
        )
        assert result is True
        assert len(extractor.standards_list) == 1
        assert extractor.standards_list[0]["chemical_name"] == "Limonene"

    def test_add_invalid_standard(self) -> None:
        """Test adding an invalid standard."""
        extractor = StandardsExtractor()
        result = extractor.add_standard(
            "null", "Test Source", "2025-01-01", "Standard", "Test details"
        )
        assert result is False
        assert len(extractor.standards_list) == 0

    def test_add_duplicate_standard(self) -> None:
        """Test that duplicates are rejected."""
        extractor = StandardsExtractor()
        extractor.add_standard("Limonene", "Source1", "2025-01-01", "Std", "Details")
        result = extractor.add_standard("limonene", "Source2", "2025-01-02", "Std", "Details")
        assert result is False
        assert len(extractor.standards_list) == 1

    def test_add_standard_strips_x_prefix(self) -> None:
        """Test that X. prefix is stripped from R-style names."""
        extractor = StandardsExtractor()
        result = extractor.add_standard("X.Limonene", "Source", "2025-01-01", "Std", "Details")
        assert result is True
        # Name should be cleaned
        assert len(extractor.standards_list) == 1

    def test_add_standard_converts_dots_to_hyphens(self) -> None:
        """Test that dots are converted to hyphens."""
        extractor = StandardsExtractor()
        result = extractor.add_standard("alpha.pinene", "Source", "2025-01-01", "Std", "Details")
        assert result is True

    def test_add_none_returns_false(self) -> None:
        """Test adding None returns False."""
        extractor = StandardsExtractor()
        result = extractor.add_standard(None, "Source", "2025-01-01", "Std", "Details")
        assert result is False


class TestStandardsExtractorFileDate:
    """Tests for StandardsExtractor._get_file_date method."""

    def test_get_file_date(self, tmp_path: Path) -> None:
        """Test getting file modification date."""
        extractor = StandardsExtractor()
        test_file = tmp_path / "test.txt"
        test_file.touch()

        result = extractor._get_file_date(test_file)

        # Should be in YYYY-MM-DD format
        assert len(result) == 10
        assert result.count("-") == 2
        # Year should be valid
        year = int(result[:4])
        assert year >= 2020


class TestStandardsExtractorWriteOutput:
    """Tests for StandardsExtractor.write_output method."""

    def test_write_output(self, tmp_path: Path) -> None:
        """Test writing output to Excel."""
        extractor = StandardsExtractor()
        extractor.add_standard("Limonene", "Source", "2025-01-01", "Std", "Details")
        extractor.add_standard("alpha-Pinene", "Source", "2025-01-01", "Std", "Details")

        output_path = tmp_path / "output.xlsx"
        extractor.write_output(output_path)

        assert output_path.exists()

        wb = _load_workbook(output_path)
        ws = wb.active
        assert ws.cell(row=1, column=1).value == "Chemical Name"
        assert ws.max_row == 3  # Header + 2 standards
        wb.close()

    def test_write_output_empty(self, tmp_path: Path) -> None:
        """Test writing empty output."""
        extractor = StandardsExtractor()

        output_path = tmp_path / "output.xlsx"
        extractor.write_output(output_path)

        assert output_path.exists()
