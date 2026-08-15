"""Tests for svg renderer: CalculateRank."""

from __future__ import annotations

from github_stats_api.renderers._common import escape_xml, format_number
from github_stats_api.renderers.stats import _calculate_rank


class TestCalculateRank:
    """Tests for _calculate_rank function.

    Rank calculation uses: score = commits*1 + prs*2 + issues*1 + stars*4
    Then: percentile = 100 - log10(score+1) * 15

    Thresholds:
    - S+: percentile <= 1 (score >= ~4 million)
    - S: percentile <= 12.5 (score >= ~680k)
    - A+: percentile <= 25 (score >= 100k)
    - A: percentile <= 37.5 (score >= ~15k)
    - B+: percentile <= 50 (score >= ~2k)
    - B: percentile <= 62.5 (score >= ~300)
    - C: everything else
    """

    def test_calculate_rank_s_plus(self) -> None:
        """Test S+ rank calculation."""
        # Need score >= 4 million for S+ (percentile <= 1)
        # Using stars=1,000,000 gives score = 4,000,000
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=1_000_000,
        )
        assert rank == "S+"
        assert percentile <= 1

    def test_calculate_rank_s(self) -> None:
        """Test S rank calculation."""
        # Need score ~680k-4M for S (percentile 1-12.5)
        # Using stars=200,000 gives score = 800,000
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=200_000,
        )
        assert rank == "S"
        assert 1 < percentile <= 12.5

    def test_calculate_rank_a_plus(self) -> None:
        """Test A+ rank calculation."""
        # Need score ~100k-680k for A+ (percentile 12.5-25)
        # Using stars=50,000 gives score = 200,000
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=50_000,
        )
        assert rank == "A+"
        assert 12.5 < percentile <= 25

    def test_calculate_rank_a(self) -> None:
        """Test A rank calculation."""
        # Need score ~15k-100k for A (percentile 25-37.5)
        # Using stars=10,000 gives score = 40,000
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=10_000,
        )
        assert rank == "A"
        assert 25 < percentile <= 37.5

    def test_calculate_rank_b_plus(self) -> None:
        """Test B+ rank calculation."""
        # Need score ~2k-15k for B+ (percentile 37.5-50)
        # Using stars=1,000 gives score = 4,000
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=1_000,
        )
        assert rank == "B+"
        assert 37.5 < percentile <= 50

    def test_calculate_rank_b(self) -> None:
        """Test B rank calculation."""
        # Need score ~300-2k for B (percentile 50-62.5)
        # Using stars=200 gives score = 800
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=200,
        )
        assert rank == "B"
        assert 50 < percentile <= 62.5

    def test_calculate_rank_c(self) -> None:
        """Test C rank calculation."""
        # Low activity gives C rank (percentile > 62.5)
        rank, percentile = _calculate_rank(
            commits=1,
            prs=0,
            issues=0,
            stars=0,
        )
        assert rank == "C"
        assert percentile > 62.5

    def test_calculate_rank_zero_activity(self) -> None:
        """Test rank calculation with zero activity."""
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=0,
        )
        assert rank == "C"
        assert percentile == 100.0


class TestFormatNumber:
    """Tests for format_number function."""

    def testformat_number_millions(self) -> None:
        """Test formatting numbers in millions."""
        assert format_number(1_000_000) == "1.0M"
        assert format_number(2_500_000) == "2.5M"
        assert format_number(10_000_000) == "10.0M"

    def testformat_number_thousands(self) -> None:
        """Test formatting numbers in thousands."""
        assert format_number(1_000) == "1.0k"
        assert format_number(2_500) == "2.5k"
        assert format_number(999_999) == "1000.0k"

    def testformat_number_small(self) -> None:
        """Test formatting small numbers."""
        assert format_number(0) == "0"
        assert format_number(1) == "1"
        assert format_number(999) == "999"


class TestEscapeXml:
    """Tests for escape_xml function."""

    def testescape_xml_ampersand(self) -> None:
        """Test escaping ampersand."""
        assert escape_xml("A & B") == "A &amp; B"

    def testescape_xml_less_than(self) -> None:
        """Test escaping less than."""
        assert escape_xml("A < B") == "A &lt; B"

    def testescape_xml_greater_than(self) -> None:
        """Test escaping greater than."""
        assert escape_xml("A > B") == "A &gt; B"

    def testescape_xml_quotes(self) -> None:
        """Test escaping quotes."""
        assert escape_xml('A "B" C') == "A &quot;B&quot; C"
        assert escape_xml("A 'B' C") == "A &apos;B&apos; C"

    def testescape_xml_no_special_chars(self) -> None:
        """Test no escaping needed."""
        assert escape_xml("Hello World") == "Hello World"
