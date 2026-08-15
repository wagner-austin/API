"""The ICU-subset rule interpreter, driven end to end.

The interpreter is three modules — :mod:`turkic_api.core.rule_lexer`,
:mod:`turkic_api.core.rule_parser` and :mod:`turkic_api.core.rule_engine` —
and these tests drive all three together, because that is the only level at
which its behaviour is observable. Asserting on token streams or parsed
``Rule`` tuples would pin the internal shape of the split rather than what a
rule file does, and would have to be rewritten every time the split moved.

Each case names the reading it protects. Cases are grouped by the concern
that owns them: quoting and comments belong to the lexer, refusals to the
parser, contexts and anchors to the engine.

Rule text lives in the test rather than in temporary files. An earlier
version wrote ``.rules`` files into the packaged rules directory, which put
files that were not rules where anything enumerating that directory could
see them — and raced under ``pytest-xdist``.

The goldens in ``test_rule_engine_goldens.py`` prove the interpreter agrees
with ICU on the real rule files. These cover the other half: what it does
with input the real files never contain, above all which malformed rules it
refuses. A rule accepted but misread transliterates silently and wrongly, so
every refusal here is load-bearing.
"""

from __future__ import annotations

import re

import pytest

from turkic_api.core.rule_engine import apply_rules
from turkic_api.core.rule_errors import RuleParseError
from turkic_api.core.rule_parser import load_rules, parse_rules


def run(rules: str, text: str) -> str:
    """Parse ``rules`` and apply them to ``text``."""
    return apply_rules(text, parse_rules(rules))


class TestPlainRules:
    """Rules with no context."""

    def test_single_character_mapping(self) -> None:
        assert run("a > b ;", "aaa") == "bbb"

    def test_unmatched_character_passes_through(self) -> None:
        assert run("a > b ;", "axa") == "bxb"

    def test_multi_character_source_consumes_all_of_it(self) -> None:
        assert run("ng > N ; n > 1 ; g > 2 ;", "ngng") == "NN"
        assert run("ng > N ; n > 1 ; g > 2 ;", "nng") == "1N"

    def test_empty_output_deletes(self) -> None:
        assert run("x > ; y > z ;", "xyx") == "z"

    def test_output_is_never_reprocessed(self) -> None:
        """ICU moves the cursor past the replacement, so b stays b."""
        assert run("a > b ; b > c ;", "a") == "b"
        assert run("a > b ; b > c ;", "b") == "c"

    def test_first_matching_rule_wins(self) -> None:
        assert run("ab > X ; a > Y ;", "ab") == "X"
        assert run("ab > X ; a > Y ;", "a") == "Y"


class TestContexts:
    """The three context asymmetries, each tested in both directions."""

    def test_before_context_matches_converted_output_not_source(self) -> None:
        """The decisive case: the left context sees b, which a became."""
        assert run("a > b ; b { c > Q ;", "ac") == "bQ"
        assert run("a > b ; b { c > Q ;", "bc") == "bQ"
        assert run("a > b ; b { c > Q ;", "xc") == "xc"

    def test_before_context_may_be_a_variable(self) -> None:
        rules = "$V = [xy] ; a > x ; $V { c > Q ;"
        assert run(rules, "ac") == "xQ"
        assert run(rules, "yc") == "yQ"
        assert run(rules, "zc") == "zc"

    def test_after_context_matches_the_untouched_source(self) -> None:
        assert run("x } y > Q ; y > z ;", "xy") == "Qz"
        assert run("x } y > Q ; y > z ;", "xz") == "xz"

    def test_after_context_does_not_see_conversions(self) -> None:
        """z is what y becomes, so an after-context of z must not match y."""
        assert run("x } z > Q ; y > z ;", "xy") == "xz"
        assert run("x } z > Q ; y > z ;", "xz") == "Qz"

    def test_after_context_at_end_of_text_cannot_match(self) -> None:
        assert run("x } y > Q ;", "x") == "x"

    def test_both_contexts_together(self) -> None:
        rules = "$V = [ae] ; $V { n } $V > N ;"
        assert run(rules, "ana") == "aNa"
        assert run(rules, "anx") == "anx"
        assert run(rules, "xna") == "xna"

    def test_before_context_longer_than_the_output_cannot_match(self) -> None:
        assert run("ab { c > Q ; c > z ;", "c") == "z"
        assert run("ab { c > Q ; c > z ;", "abc") == "abQ"

    def test_inline_sets_in_source_and_context(self) -> None:
        assert run("[nN] } [kK] > G ;", "nk") == "Gk"
        assert run("[nN] } [kK] > G ;", "NK") == "GK"
        assert run("[nN] } [kK] > G ;", "na") == "na"


class TestAnchors:
    """``^`` restricts a rule to the start of the text."""

    def test_start_anchor_applies_only_at_position_zero(self) -> None:
        assert run("^ a > S ; a > A ;", "aa") == "SA"
        assert run("^ a > S ; a > A ;", "ba") == "bA"

    def test_start_anchor_after_a_deletion_still_means_position_zero(self) -> None:
        """Deleting the first character must not make the second one initial."""
        assert run("x > ; ^ a > S ; a > A ;", "xa") == "A"

    def test_anchor_away_from_the_front_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="only allowed at the start"):
            parse_rules("a ^ b > c ;")


class TestQuotingAndEscapes:
    """``'`` and ``\\`` make operators into ordinary characters."""

    def test_quoted_run_is_literal(self) -> None:
        assert run("a > 'x;y' ;", "a") == "x;y"

    def test_quote_hides_the_statement_separator(self) -> None:
        """This is the ar_lat defect in miniature: the b rule is swallowed."""
        assert run("a > ' ; b > ' ;", "ab") == " ; b > b"

    def test_doubled_quote_is_one_apostrophe(self) -> None:
        assert run("a > '' ;", "a") == "'"

    def test_backslash_escapes_the_next_character(self) -> None:
        assert run("\\' > Q ;", "'") == "Q"
        assert run("\\> > Q ;", ">") == "Q"

    def test_escape_inside_a_set(self) -> None:
        assert run("$S = [\\]a] ; $S > Q ;", "]a") == "QQ"

    def test_quoting_does_not_apply_inside_a_set(self) -> None:
        """A bare apostrophe in a set is a member, as ``$Apo`` relies on."""
        assert run("$S = [a ' b] ; $S > Q ;", "a'b") == "QQQ"

    def test_unterminated_quote_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="unterminated quoted literal"):
            parse_rules("a > 'bc ;")

    def test_trailing_backslash_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="trailing backslash"):
            parse_rules("a > b \\")


class TestComments:
    """``#`` runs to end of line, and whitespace is insignificant."""

    def test_comment_is_ignored(self) -> None:
        assert run("# a > z ;\na > b ;", "a") == "b"

    def test_trailing_comment_is_ignored(self) -> None:
        assert run("a > b ; # not a rule", "a") == "b"

    def test_comment_without_a_trailing_newline_is_ignored(self) -> None:
        assert run("a > b ; # end", "a") == "b"

    def test_whitespace_between_elements_is_insignificant(self) -> None:
        assert run("s h > S ;", "sh") == "S"

    def test_statement_without_a_trailing_semicolon_is_still_a_rule(self) -> None:
        assert run("a > b", "a") == "b"

    def test_empty_statements_are_skipped(self) -> None:
        assert run("a > b ;; ; c > d ;", "ac") == "bd"


class TestVariables:
    """``$Name = [ ... ]`` defines a set, usable anywhere a set is."""

    def test_variable_in_source_position(self) -> None:
        assert run("$V = [abc] ; $V > Q ;", "abcd") == "QQQd"

    def test_variable_spanning_several_lines(self) -> None:
        rules = "$V = [ a\n b\n c ] ;\n$V > Q ;"
        assert run(rules, "abc") == "QQQ"

    def test_undefined_variable_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match=r"\$Missing' is not defined"):
            parse_rules("$Missing { a > b ;")

    def test_bare_dollar_is_refused(self) -> None:
        """ICU reads a bare ``$`` as an end anchor; this subset does not."""
        with pytest.raises(RuleParseError, match="not followed by a variable name"):
            parse_rules("a $ > b ;")

    def test_variable_must_be_defined_as_one_set(self) -> None:
        with pytest.raises(RuleParseError, match=re.escape("one '[...]' set")):
            parse_rules("$A = b ;")
        with pytest.raises(RuleParseError, match=re.escape("one '[...]' set")):
            parse_rules("$A = [a] [b] ;")

    def test_unterminated_set_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="unterminated character set"):
            parse_rules("$A = [abc")

    def test_nested_set_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="nested character set"):
            parse_rules("$A = [a[b]] ;")

    def test_empty_set_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="empty character set"):
            parse_rules("$A = [] ;")

    def test_trailing_backslash_in_a_set_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="trailing backslash in character set"):
            parse_rules("$A = [a\\")


class TestMalformedRules:
    """Every refusal the parser makes."""

    def test_statement_with_no_arrow_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="has no '>'"):
            parse_rules("abc ;")

    def test_two_arrows_are_refused(self) -> None:
        with pytest.raises(RuleParseError, match="more than one '>'"):
            parse_rules("a > b > c ;")

    def test_two_braces_of_one_kind_are_refused(self) -> None:
        with pytest.raises(RuleParseError, match=re.escape("more than one '{'")):
            parse_rules("a { b { c > d ;")

    def test_closing_brace_before_opening_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match=re.escape("'}' appears before '{'")):
            parse_rules("a } b { c > d ;")

    def test_rule_matching_nothing_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="matches nothing"):
            parse_rules("a { } > b ;")
        with pytest.raises(RuleParseError, match="matches nothing"):
            parse_rules("^ > b ;")

    def test_stray_operator_in_a_pattern_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="unexpected '=' in a rule pattern"):
            parse_rules("a = b > c ;")

    def test_set_on_the_output_side_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="must be literal characters"):
            parse_rules("a > [bc] ;")

    def test_error_reports_the_line_it_happened_on(self) -> None:
        with pytest.raises(RuleParseError) as caught:
            parse_rules("a > b ;\nc > d ;\ne = f > g ;")
        assert caught.value.line == 3
        assert "e = f > g" in caught.value.statement


class TestDirectives:
    """Only ``:: NFC`` is implemented, and only as the last statement."""

    def test_nfc_normalises_the_output(self) -> None:
        assert run("a > é ; :: NFC ;", "a") == "é"

    def test_without_the_directive_output_is_left_as_written(self) -> None:
        assert run("a > é ;", "a") == "é"

    def test_unsupported_directive_is_refused(self) -> None:
        with pytest.raises(RuleParseError, match="unsupported directive '::NFD'"):
            parse_rules("a > b ; :: NFD ;")

    def test_statement_after_the_directive_is_refused(self) -> None:
        """ICU would apply it after normalising, which the file does not mean."""
        with pytest.raises(RuleParseError, match="statement follows ':: NFC'"):
            parse_rules(":: NFC ; a > b ;")


class TestLoadRules:
    """Loading the packaged rule files."""

    def test_load_rules_reads_a_packaged_file(self) -> None:
        ruleset = load_rules("kk_ipa.rules")
        assert ruleset.rules
        assert ruleset.normalize_nfc is True

    def test_load_rules_on_a_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_rules("no_such_language.rules")
