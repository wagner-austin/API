"""The continuation-task contract two packages have to agree on.

These assertions are worth more than they look. The generator and the scorer
never call each other, so nothing at runtime notices when they disagree --
the scorer simply finds no files and reports having scored nothing, which is
what a crashed generation also looks like. This file is where the agreement
is actually checked.
"""

from __future__ import annotations

import pathlib

import pytest

from platform_core.continuation_task import (
    EvalPrompt,
    GenerationEntry,
    MalformedRecordError,
    batches,
    build_prompts,
    decode_generation_entry,
    encode_generation_entry,
    finishable,
    flatten_item_id,
    generated_path,
    item_root,
    manifest_path,
    split_document,
)
from platform_core.json_utils import JSONTypeError, dump_json_str

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


def _prompt(item_id: str, prompt: str, reference: str) -> EvalPrompt:
    """Build one prompt directly, without going through a corpus.

    Args:
        item_id: The item's path within its repository.
        prompt: What the model is shown.
        reference: What it must write.

    Returns:
        The prompt.
    """
    return EvalPrompt(item_id=item_id, prompt=prompt, reference=reference)


def _characters(text: str) -> int:
    """Count a string in characters, standing in for a tokenizer.

    Everything in this module is about shape, so the measure only has to be
    monotonic in length -- which is exactly why it is a Protocol and not a
    tokenizer.

    Args:
        text: The string to measure.

    Returns:
        Its length.
    """
    return len(text)


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


class TestWhereAGeneratedFileGoes:
    """The layout the generator writes and the scorer reads.

    Stated once here because the two live in different packages and a
    disagreement between them is silent: the scorer finds nothing and
    reports scoring nothing, which is what a crashed generation looks like.
    """

    def test_a_path_becomes_one_segment(self) -> None:
        """Joining instead would let a '..' in an id escape the directory."""
        assert flatten_item_id("src/pkg/mod.py") == "src__pkg__mod.py"

    def test_a_windows_separator_flattens_the_same_way(self) -> None:
        """A corpus emitted on Windows must land where a Linux run looks."""
        assert flatten_item_id("src\\pkg\\mod.py") == "src__pkg__mod.py"

    def test_an_id_that_is_not_python_is_refused(self) -> None:
        """The guards glob for ``*.py``; anything else scores a vacuous pass."""
        with pytest.raises(ValueError, match="not a Python file"):
            _ = flatten_item_id("README.md")

    def test_every_item_gets_its_own_guard_root(self) -> None:
        """A shared root returns one guards verdict for the whole sweep."""
        root = item_root(pathlib.Path("out"), "src/pkg/mod.py")
        other = item_root(pathlib.Path("out"), "src/pkg/two.py")
        assert root != other

    def test_the_file_sits_under_src_inside_that_root(self) -> None:
        """``src`` is one of the directories the guards scan."""
        target = generated_path(pathlib.Path("out"), "src/pkg/mod.py")
        assert target == pathlib.Path("out") / "src__pkg__mod.py" / "src" / "src__pkg__mod.py"

    def test_the_file_lives_inside_its_own_guard_root(self) -> None:
        """Otherwise the guards would be pointed somewhere the file is not."""
        root = item_root(pathlib.Path("out"), "src/pkg/mod.py")
        target = generated_path(pathlib.Path("out"), "src/pkg/mod.py")
        assert root in target.parents

    def test_the_manifest_is_a_sibling_of_the_directory(self) -> None:
        """Inside it, every reader walking the tree would have to skip it."""
        assert manifest_path(pathlib.Path("runs/base")) == pathlib.Path(
            "runs/base.generation.jsonl"
        )

    def test_two_arms_get_two_manifests(self) -> None:
        """The name is derived from the directory, so arms cannot collide."""
        assert manifest_path(pathlib.Path("runs/base")) != manifest_path(
            pathlib.Path("runs/candidate")
        )


class TestTheGenerationManifest:
    """Whether a completion ended or ran out of budget, recorded not inferred."""

    def test_an_entry_round_trips(self) -> None:
        entry = GenerationEntry(item_id="src/pkg/mod.py", finished=True)

        assert decode_generation_entry(encode_generation_entry(entry)) == entry

    def test_an_unfinished_entry_round_trips(self) -> None:
        """False is the interesting value and must survive as itself."""
        entry = GenerationEntry(item_id="src/pkg/mod.py", finished=False)

        assert decode_generation_entry(encode_generation_entry(entry)) == entry

    def test_a_row_naming_no_item_is_refused(self) -> None:
        """A row nothing can be attributed to is not a row."""
        with pytest.raises(JSONTypeError, match="must not be empty"):
            _ = decode_generation_entry({"item_id": "", "finished": True})

    def test_a_missing_item_id_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="item_id"):
            _ = decode_generation_entry({"finished": True})

    def test_a_missing_finish_flag_is_refused(self) -> None:
        """Absent is not the same as False, and guessing loses the distinction."""
        with pytest.raises(JSONTypeError, match="finished"):
            _ = decode_generation_entry({"item_id": "a.py"})

    def test_a_non_boolean_finish_flag_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="finished"):
            _ = decode_generation_entry({"item_id": "a.py", "finished": "yes"})


class TestWhichItemsAreAnswerable:
    """An item whose reference cannot fit the budget is not a fair item."""

    def test_an_item_whose_reference_fits_is_kept(self) -> None:
        prompts = [_prompt("a.py", "head", "tail")]

        assert finishable(prompts, _characters, 4) == prompts

    def test_an_item_whose_reference_exceeds_the_budget_is_dropped(self) -> None:
        """It would end mid-expression and fail every checker on syntax."""
        prompts = [_prompt("a.py", "head", "far too long")]

        assert finishable(prompts, _characters, 4) == []

    def test_the_budget_is_inclusive(self) -> None:
        """An item that exactly fills the budget can still finish."""
        prompts = [_prompt("a.py", "head", "abcd")]

        assert finishable(prompts, _characters, 4) == prompts

    def test_the_surviving_order_is_the_order_given(self) -> None:
        """Both arms iterate this list, so its order carries the pairing."""
        prompts = [
            _prompt("a.py", "h", "x"),
            _prompt("b.py", "h", "far too long"),
            _prompt("c.py", "h", "y"),
        ]

        assert [p["item_id"] for p in finishable(prompts, _characters, 4)] == ["a.py", "c.py"]

    def test_a_zero_budget_is_refused(self) -> None:
        """An empty sweep reporting success is worse than one that refuses."""
        with pytest.raises(ValueError, match="budget must be positive"):
            _ = finishable([_prompt("a.py", "h", "x")], _characters, 0)

    def test_a_negative_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="budget must be positive"):
            _ = finishable([_prompt("a.py", "h", "x")], _characters, -1)


class TestHowPromptsAreBatched:
    """Batch composition is part of the arm, not an implementation detail."""

    def test_prompts_are_grouped_up_to_the_batch_size(self) -> None:
        prompts = [_prompt(f"{index}.py", "h", "x") for index in range(5)]

        assert [len(batch) for batch in batches(prompts, _characters, 2)] == [2, 2, 1]

    def test_no_prompt_is_lost_or_duplicated(self) -> None:
        prompts = [_prompt(f"{index}.py", "h" * index, "x") for index in range(5)]

        grouped = batches(prompts, _characters, 2)

        assert sorted(p["item_id"] for batch in grouped for p in batch) == sorted(
            p["item_id"] for p in prompts
        )

    def test_batches_are_sorted_by_prompt_length(self) -> None:
        """Uniform batches make padding a few tokens rather than the sweep's range."""
        prompts = [
            _prompt("long.py", "hhhh", "x"),
            _prompt("short.py", "h", "x"),
            _prompt("mid.py", "hh", "x"),
        ]

        grouped = batches(prompts, _characters, 1)

        assert [batch[0]["item_id"] for batch in grouped] == ["short.py", "mid.py", "long.py"]

    def test_equal_lengths_break_ties_on_item_id(self) -> None:
        """Without a total order, two arms could batch the same sweep differently."""
        prompts = [_prompt("b.py", "h", "x"), _prompt("a.py", "h", "x")]

        grouped = batches(prompts, _characters, 2)

        assert [p["item_id"] for p in grouped[0]] == ["a.py", "b.py"]

    def test_the_same_prompts_in_another_order_batch_identically(self) -> None:
        """This is what makes an item sit with the same neighbours in both arms."""
        prompts = [_prompt(f"{index}.py", "h" * (index % 3), "x") for index in range(6)]
        reversed_prompts = list(reversed(prompts))

        assert batches(prompts, _characters, 2) == batches(reversed_prompts, _characters, 2)

    def test_no_prompts_produce_no_batches(self) -> None:
        assert batches([], _characters, 4) == []

    def test_a_zero_batch_size_is_refused(self) -> None:
        """A batch of zero would loop forever producing nothing."""
        with pytest.raises(ValueError, match="size must be positive"):
            _ = batches([_prompt("a.py", "h", "x")], _characters, 0)

    def test_a_negative_batch_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="size must be positive"):
            _ = batches([_prompt("a.py", "h", "x")], _characters, -1)
