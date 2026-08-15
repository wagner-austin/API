"""Reading the source each rule file declares.

The point of a machine-readable header is that a test can inherit its
citation from the rules rather than restating it. So these tests cover both
the reader and the real headers: every IPA rule file must declare a complete,
resolvable source, because a rule set whose provenance cannot be read is a
rule set whose claims cannot be checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from turkic_api.core.rule_provenance import (
    AUTHORS_FIELD,
    CONTAINER_FIELD,
    ERR_FIELD_EMPTY,
    ERR_FIELD_MISSING,
    ERR_MALFORMED_LINE,
    FIELD_PREFIX,
    IDENTIFIER_FIELD,
    REQUIRED_FIELDS,
    TITLE_FIELD,
    YEAR_FIELD,
    RuleSource,
    RuleSourceFieldEmptyError,
    RuleSourceFieldMissingError,
    RuleSourceMalformedLineError,
    decode_rule_source,
    encode_rule_source,
    parse_rule_source,
    read_rule_source,
)
from turkic_api.core.translit import _RULE_DIR

# Every rule file that maps to IPA. The Latin files carry prose headers rather
# than the structured block, so they are not in scope for the reader.
IPA_RULE_FILES: Final[tuple[str, ...]] = (
    "az_ipa.rules",
    "fi_ipa.rules",
    "kk_ipa.rules",
    "ky_ipa.rules",
    "ru_ipa.rules",
    "tr_ipa.rules",
    "ug_ipa.rules",
    "uz_ipa.rules",
    "uzc_ipa.rules",
)

COMPLETE_HEADER = (
    "# Kyrgyz → IPA transliteration rules\n"
    "#\n"
    f"{FIELD_PREFIX}{AUTHORS_FIELD}: McCollum, A. G.\n"
    f"{FIELD_PREFIX}{YEAR_FIELD}: 2020\n"
    f"{FIELD_PREFIX}{TITLE_FIELD}: Vowel harmony in Kyrgyz\n"
    f"{FIELD_PREFIX}{CONTAINER_FIELD}: Laboratory Phonology 11(1)\n"
    f"{FIELD_PREFIX}{IDENTIFIER_FIELD}: https://doi.org/10.5334/labphon.247\n"
    "#\n"
    "а > ɑ ;\n"
)


def _record() -> RuleSource:
    """Build a valid source record."""
    return RuleSource(
        authors="McCollum, A. G.",
        year=2020,
        title="Vowel harmony in Kyrgyz",
        container="Laboratory Phonology 11(1)",
        identifier="https://doi.org/10.5334/labphon.247",
    )


class TestParseRuleSource:
    """Reading a header out of rule text."""

    def test_a_complete_header_parses(self) -> None:
        assert parse_rule_source(COMPLETE_HEADER, "ky_ipa.rules") == _record()

    def test_ordinary_comments_and_rules_are_ignored(self) -> None:
        """Only ``# Source-`` lines are read, so prose cannot confuse it."""
        text = COMPLETE_HEADER + "# Corroborated: some other work\nб > b ;\n"

        assert parse_rule_source(text, "ky_ipa.rules") == _record()

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_a_missing_field_is_refused_by_name(self, field: str) -> None:
        text = "\n".join(
            line
            for line in COMPLETE_HEADER.splitlines()
            if not line.startswith(f"{FIELD_PREFIX}{field}:")
        )

        with pytest.raises(RuleSourceFieldMissingError) as caught:
            parse_rule_source(text, "ky_ipa.rules")

        assert caught.value.field == field
        assert caught.value.origin == "ky_ipa.rules"
        assert caught.value.code == ERR_FIELD_MISSING

    def test_a_line_without_a_separator_is_refused(self) -> None:
        text = COMPLETE_HEADER + f"{FIELD_PREFIX}Nonsense\n"

        with pytest.raises(RuleSourceMalformedLineError) as caught:
            parse_rule_source(text, "ky_ipa.rules")

        assert caught.value.code == ERR_MALFORMED_LINE

    def test_a_field_given_twice_is_refused(self) -> None:
        """Two answers to one question is no answer."""
        text = COMPLETE_HEADER + f"{FIELD_PREFIX}{YEAR_FIELD}: 1999\n"

        with pytest.raises(RuleSourceMalformedLineError, match="unreadable source line"):
            parse_rule_source(text, "ky_ipa.rules")

    def test_a_non_numeric_year_is_refused(self) -> None:
        text = COMPLETE_HEADER.replace(f"{YEAR_FIELD}: 2020", f"{YEAR_FIELD}: two thousand")

        with pytest.raises(RuleSourceMalformedLineError, match="two thousand"):
            parse_rule_source(text, "ky_ipa.rules")

    def test_an_empty_field_is_refused_because_it_looks_answered(self) -> None:
        text = COMPLETE_HEADER.replace("McCollum, A. G.", "")

        with pytest.raises(RuleSourceFieldEmptyError) as caught:
            parse_rule_source(text, "ky_ipa.rules")

        assert caught.value.field == "authors"
        assert caught.value.code == ERR_FIELD_EMPTY

    def test_values_are_stripped_of_surrounding_space(self) -> None:
        text = COMPLETE_HEADER.replace(
            f"{TITLE_FIELD}: Vowel harmony in Kyrgyz",
            f"{TITLE_FIELD}:    Vowel harmony in Kyrgyz   ",
        )

        assert parse_rule_source(text, "ky_ipa.rules")["title"] == "Vowel harmony in Kyrgyz"

    def test_a_value_containing_a_colon_keeps_it(self) -> None:
        """Container strings carry volume and page ranges after a colon."""
        text = COMPLETE_HEADER.replace(
            f"{CONTAINER_FIELD}: Laboratory Phonology 11(1)",
            f"{CONTAINER_FIELD}: Laboratory Phonology 11(1): article 25",
        )

        assert parse_rule_source(text, "ky_ipa.rules")["container"] == (
            "Laboratory Phonology 11(1): article 25"
        )


class TestRuleSourceCoding:
    """Records survive a round trip."""

    def test_encode_then_decode_returns_the_same_record(self) -> None:
        assert decode_rule_source(encode_rule_source(_record()), "x.rules") == _record()

    def test_encode_carries_exactly_the_five_fields(self) -> None:
        assert set(encode_rule_source(_record())) == {
            "authors",
            "year",
            "title",
            "container",
            "identifier",
        }

    def test_a_missing_field_is_refused(self) -> None:
        source: JSONObject = {"authors": "someone"}

        with pytest.raises(JSONTypeError):
            decode_rule_source(source, "x.rules")

    def test_a_non_integer_year_is_refused(self) -> None:
        source: JSONObject = dict(encode_rule_source(_record()))
        source["year"] = "2020"

        with pytest.raises(JSONTypeError):
            decode_rule_source(source, "x.rules")

    def test_an_empty_text_field_is_refused(self) -> None:
        source: JSONObject = dict(encode_rule_source(_record()))
        source["identifier"] = ""

        with pytest.raises(RuleSourceFieldEmptyError) as caught:
            decode_rule_source(source, "x.rules")

        assert caught.value.field == "identifier"


class TestReadRuleSource:
    """Reading from disk."""

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_rule_source(tmp_path / "absent.rules")

    def test_a_file_is_read_and_named_in_its_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.rules"
        path.write_text(f"{FIELD_PREFIX}{AUTHORS_FIELD}: only this\n", encoding="utf-8")

        with pytest.raises(RuleSourceFieldMissingError) as caught:
            read_rule_source(path)

        assert caught.value.origin == "broken.rules"


class TestVendoredHeaders:
    """The real rule files, which the gold-standard tests cite."""

    @pytest.mark.parametrize("name", IPA_RULE_FILES)
    def test_every_ipa_rule_file_declares_a_complete_source(self, name: str) -> None:
        declared = read_rule_source(_RULE_DIR / name)

        assert declared["authors"]
        assert declared["title"]
        assert declared["container"]
        assert declared["year"] > 1900

    @pytest.mark.parametrize("name", IPA_RULE_FILES)
    def test_every_declared_identifier_is_resolvable(self, name: str) -> None:
        """A citation that cannot be resolved is not a citation."""
        identifier = read_rule_source(_RULE_DIR / name)["identifier"]

        assert identifier.startswith("https://")

    def test_the_kyrgyz_rules_cite_the_article_its_tests_inherit(self) -> None:
        """test_kyrgyz_ipa_letters declares this source; the rules must agree."""
        declared = read_rule_source(_RULE_DIR / "ky_ipa.rules")

        assert declared["identifier"] == "https://doi.org/10.5334/labphon.247"
        assert declared["authors"] == "McCollum, A. G."
        assert declared["year"] == 2020
