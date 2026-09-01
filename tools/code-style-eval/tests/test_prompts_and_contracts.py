"""Prompt construction from a holdout corpus, and the outcome codecs."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, dump_json_str

from code_style_eval.contracts.outcomes import (
    CHECKERS,
    CheckOutcome,
    ItemOutcome,
    PairedCounts,
    as_checker,
    decode_check_outcome,
    decode_item_outcome,
    decode_paired_counts,
    encode_check_outcome,
    encode_item_outcome,
    encode_paired_counts,
)
from code_style_eval.core.prompts import (
    MalformedRecordError,
    build_prompts,
    split_document,
)

_SOURCE = "".join(f"line{i}\n" for i in range(10))


def _require_split(split: tuple[str, str] | None) -> tuple[str, str]:
    """Narrow a split, failing loudly when the document was too short.

    Args:
        split: What ``split_document`` returned.

    Returns:
        The prompt and the reference.

    Raises:
        AssertionError: When the document produced no split, which would
            otherwise turn a continuation assertion into a skipped one.
    """
    if split is None:
        raise AssertionError("document was too short to split")
    return split


def _record(path: str, text: str) -> str:
    """Render one holdout JSONL line.

    Args:
        path: The file's repository-relative path.
        text: Its contents.

    Returns:
        The JSONL line, without a trailing newline.
    """
    return dump_json_str({"repo": "api", "path": path, "text": text})


class TestSplittingADocument:
    """The prompt/reference split is on a line boundary."""

    def test_the_split_is_on_a_line_boundary(self) -> None:
        """Cutting mid-token would make this a repair task, not a continuation."""
        prompt, reference = _require_split(split_document(_SOURCE, 3))
        assert prompt == "line0\nline1\nline2\n"
        assert reference.startswith("line3\n")
        assert prompt + reference == _SOURCE

    def test_a_file_no_longer_than_the_prompt_is_refused(self) -> None:
        """Scoring an empty target would record a pass for writing nothing."""
        assert split_document("a\nb\n", 2) is None

    def test_a_file_shorter_than_the_prompt_is_refused(self) -> None:
        """Same reason, one line short."""
        assert split_document("a\n", 5) is None


class TestBuildingPrompts:
    """Whole-file continuation prompts from holdout records."""

    def test_one_prompt_per_usable_record(self) -> None:
        """Order follows the corpus, so runs line up across arms."""
        records = [_record("a.py", _SOURCE), _record("b.py", _SOURCE)]

        prompts = build_prompts(records, 3)

        assert [p["item_id"] for p in prompts] == ["a.py", "b.py"]
        assert all(p["prompt"] + p["reference"] == _SOURCE for p in prompts)

    def test_blank_lines_between_records_are_skipped(self) -> None:
        """Framing is a property of the file, not a record."""
        records = ["", _record("a.py", _SOURCE), "   ", _record("b.py", _SOURCE)]

        assert len(build_prompts(records, 3)) == 2

    def test_a_file_too_short_to_continue_is_dropped(self) -> None:
        """Dropped, not failed: it is not a continuation task."""
        records = [_record("short.py", "a\n"), _record("long.py", _SOURCE)]

        prompts = build_prompts(records, 3)

        assert [p["item_id"] for p in prompts] == ["long.py"]

    def test_a_zero_line_prompt_is_refused(self) -> None:
        """Writing a file from nothing is a different experiment."""
        with pytest.raises(ValueError, match="prompt_lines must be positive"):
            _ = build_prompts([_record("a.py", _SOURCE)], 0)

    def test_a_negative_prompt_length_is_refused(self) -> None:
        """Same rule from the other side."""
        with pytest.raises(ValueError, match="prompt_lines must be positive"):
            _ = build_prompts([_record("a.py", _SOURCE)], -1)

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("{not json", "not valid JSON"),
            ('["a"]', "not a JSON object"),
            (dump_json_str({"text": "x\ny\n"}), "no string 'path'"),
            (dump_json_str({"path": "", "text": "x\ny\n"}), "no string 'path'"),
            (dump_json_str({"path": "a.py"}), "no non-empty 'text'"),
            (dump_json_str({"path": "a.py", "text": ""}), "no non-empty 'text'"),
            (dump_json_str({"path": "a.py", "text": 7}), "no non-empty 'text'"),
        ],
    )
    def test_a_malformed_record_is_refused_by_name(self, line: str, expected: str) -> None:
        """Each rejection says what is wrong and which line it was on.

        Args:
            line: The malformed record.
            expected: Substring the message must carry.
        """
        with pytest.raises(MalformedRecordError, match=expected):
            _ = build_prompts([line], 1)

    def test_the_error_names_the_line_number(self) -> None:
        """A holdout of hundreds needs a locator, not just a reason."""
        records = [_record("a.py", _SOURCE), "{not json"]

        with pytest.raises(MalformedRecordError, match="line 2"):
            _ = build_prompts(records, 3)


class TestCheckerNarrowing:
    """The checker set is closed."""

    @pytest.mark.parametrize("checker", CHECKERS)
    def test_every_declared_checker_narrows(self, checker: str) -> None:
        """Iterating CHECKERS keeps this honest as the set grows.

        Args:
            checker: The checker name.
        """
        assert as_checker(checker, "checker") == checker

    def test_an_unknown_checker_is_refused(self) -> None:
        """A typo must not reach the runner."""
        with pytest.raises(JSONTypeError, match="checker"):
            _ = as_checker("pylint", "checker")


class TestOutcomeCodecs:
    """Every record round-trips, and a self-contradicting one is refused."""

    def test_a_check_outcome_round_trips(self) -> None:
        """Exit code survives, because a crash and a clean run differ."""
        outcome = CheckOutcome(checker="mypy", passed=False, exit_code=2, detail="boom")

        assert decode_check_outcome(encode_check_outcome(outcome)) == outcome

    def test_an_item_outcome_round_trips(self) -> None:
        """The per-checker rows survive, not just the summary."""
        checks = (
            CheckOutcome(checker="ruff", passed=True, exit_code=0, detail=""),
            CheckOutcome(checker="mypy", passed=True, exit_code=0, detail=""),
            CheckOutcome(checker="guards", passed=True, exit_code=0, detail=""),
        )
        outcome = ItemOutcome(item_id="a.py", arm="base", checks=checks, all_passed=True)

        assert decode_item_outcome(encode_item_outcome(outcome)) == outcome

    def test_a_summary_contradicting_its_rows_is_refused(self) -> None:
        """A record claiming a pass over a failing checker cannot be compared."""
        encoded = encode_item_outcome(
            ItemOutcome(
                item_id="a.py",
                arm="base",
                checks=(CheckOutcome(checker="ruff", passed=False, exit_code=1, detail="x"),),
                all_passed=False,
            )
        )
        encoded["all_passed"] = True

        with pytest.raises(JSONTypeError, match="disagrees with"):
            _ = decode_item_outcome(encoded)

    def test_paired_counts_round_trip(self) -> None:
        """The 2x2 table survives serialization."""
        counts = PairedCounts(both_passed=1, baseline_only=2, candidate_only=3, neither=4)

        assert decode_paired_counts(encode_paired_counts(counts)) == counts
