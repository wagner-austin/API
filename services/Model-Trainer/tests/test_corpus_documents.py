"""The document corpus format, read end to end against real files.

A ``documents`` corpus exists because the ``lines`` reader strips every line
and drops the blank ones. That is right for prose and destroys source code:
indentation is Python's block syntax, so a stripped file does not parse, and a
model trained on it is being shown text that could not have come from the
corpus it is meant to be learning. Every test here writes real bytes to disk
and reads them back through the real reader; nothing is mocked, because the
property under test is what survives a round trip through the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str

from model_trainer.core.contracts.dataset import (
    CORPUS_FORMATS,
    DatasetConfig,
    as_corpus_format,
    require_corpus_format,
)
from model_trainer.core.encoding import ListEncoded
from model_trainer.core.services.data.corpus import (
    list_document_files,
    list_text_files,
    read_corpus_documents,
)
from model_trainer.core.services.training.dataset_builder import (
    CausalLMDataset,
    corpus_suffixes,
    split_corpus,
    unit_noun,
)

# A Python source file whose meaning is entirely in its whitespace. Read as
# stripped lines every one of these bodies dedents to column zero and the file
# stops parsing; read as a document it survives byte for byte.
INDENTED_SOURCE = (
    "def outer(value: int) -> int:\n    if value > 0:\n        return value\n\n    return -value\n"
)


def _write_records(path: Path, texts: list[str]) -> Path:
    """Write one JSONL record per text, the way the emitter writes them.

    Args:
        path: File to write.
        texts: Document bodies, in order.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(dump_json_str({"text": t}) + "\n" for t in texts), encoding="utf-8")
    return path


def _write_raw(path: Path, body: str) -> Path:
    """Write a corpus file verbatim, for the malformed cases.

    Args:
        path: File to write.
        body: Exact contents.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class TestIndentationSurvives:
    """The property the whole format exists for."""

    def test_a_document_keeps_every_leading_space(self, tmp_path: Path) -> None:
        """Read as a document, source code comes back byte for byte."""
        corpus = _write_records(tmp_path / "code.jsonl", [INDENTED_SOURCE])

        documents = read_corpus_documents([str(corpus)])

        assert documents == (INDENTED_SOURCE,)
        assert "        return value\n" in documents[0]

    def test_the_document_still_compiles_after_the_round_trip(self, tmp_path: Path) -> None:
        """The strongest available check that nothing was dedented."""
        corpus = _write_records(tmp_path / "code.jsonl", [INDENTED_SOURCE])

        documents = read_corpus_documents([str(corpus)])

        compile(documents[0], "<corpus>", "exec")

    def test_the_line_reader_would_have_destroyed_it(self, tmp_path: Path) -> None:
        """Names the defect this format was built to avoid, as a live check.

        Not a redundant assertion about someone else's function: it is the
        reason ``documents`` exists, and it fails loudly if the line reader is
        ever changed to preserve whitespace and the two paths become the same.
        """
        text_corpus = tmp_path / "code.txt"
        text_corpus.write_text(INDENTED_SOURCE, encoding="utf-8")

        cfg = DatasetConfig(
            corpus_path=str(text_corpus),
            corpus_format="lines",
            holdout_fraction=0.0,
            test_split_ratio=0.0,
        )
        rejoined = "\n".join(split_corpus(cfg)["train"])

        assert "        return value" not in rejoined
        with pytest.raises(IndentationError):
            compile(rejoined, "<corpus>", "exec")


class TestListingDocumentFiles:
    """Which files a document corpus path names."""

    def test_a_file_path_names_itself(self, tmp_path: Path) -> None:
        """Pointing straight at a corpus file is the prevailing layout."""
        corpus = _write_records(tmp_path / "one.jsonl", ["a"])

        assert list_document_files(str(corpus)) == [str(corpus)]

    def test_a_directory_is_walked_for_jsonl_only(self, tmp_path: Path) -> None:
        """A ``.txt`` beside the corpus belongs to the other format."""
        _write_records(tmp_path / "kept.jsonl", ["a"])
        (tmp_path / "ignored.txt").write_text("not a document corpus\n", encoding="utf-8")

        found = list_document_files(str(tmp_path))

        assert [Path(p).name for p in found] == ["kept.jsonl"]

    def test_the_listing_is_sorted(self, tmp_path: Path) -> None:
        """File order decides the split, so it cannot be filesystem order."""
        for name in ("c.jsonl", "a.jsonl", "b.jsonl"):
            _write_records(tmp_path / name, ["x"])

        found = list_document_files(str(tmp_path))

        assert [Path(p).name for p in found] == ["a.jsonl", "b.jsonl", "c.jsonl"]

    def test_nested_directories_are_included(self, tmp_path: Path) -> None:
        """The walk is recursive, like the text lister's."""
        _write_records(tmp_path / "top.jsonl", ["a"])
        _write_records(tmp_path / "nested" / "deep.jsonl", ["b"])

        found = list_document_files(str(tmp_path))

        assert sorted(Path(p).name for p in found) == ["deep.jsonl", "top.jsonl"]

    def test_a_jsonl_is_invisible_to_the_text_lister(self, tmp_path: Path) -> None:
        """The two listers must not overlap, or a corpus reads under both."""
        _write_records(tmp_path / "code.jsonl", ["a"])

        assert list_text_files(str(tmp_path)) == []


class TestReadingRecords:
    """Framing, ordering, and what counts as a record."""

    def test_records_come_back_in_file_order(self, tmp_path: Path) -> None:
        """Order is the split's basis, so it is part of the contract."""
        corpus = _write_records(tmp_path / "c.jsonl", ["first", "second", "third"])

        assert read_corpus_documents([str(corpus)]) == ("first", "second", "third")

    def test_files_concatenate_in_the_order_given(self, tmp_path: Path) -> None:
        """Several files are one corpus, joined in the caller's order."""
        a = _write_records(tmp_path / "a.jsonl", ["from a"])
        b = _write_records(tmp_path / "b.jsonl", ["from b"])

        assert read_corpus_documents([str(a), str(b)]) == ("from a", "from b")

    def test_blank_framing_lines_between_records_are_skipped(self, tmp_path: Path) -> None:
        """A blank separator is the file's framing, not a record."""
        corpus = _write_raw(
            tmp_path / "c.jsonl",
            dump_json_str({"text": "one"}) + "\n\n" + dump_json_str({"text": "two"}) + "\n\n",
        )

        assert read_corpus_documents([str(corpus)]) == ("one", "two")

    def test_extra_record_fields_are_ignored(self, tmp_path: Path) -> None:
        """The emitter writes repo/path/sha256 beside the text."""
        corpus = _write_raw(
            tmp_path / "c.jsonl",
            dump_json_str({"repo": "api", "path": "a.py", "sha256": "ab", "text": "body"}) + "\n",
        )

        assert read_corpus_documents([str(corpus)]) == ("body",)

    def test_an_empty_file_yields_no_records(self, tmp_path: Path) -> None:
        """Emptiness is reported by the caller, not invented here."""
        corpus = _write_raw(tmp_path / "c.jsonl", "")

        assert read_corpus_documents([str(corpus)]) == ()


class TestMalformedRecordsAreRefused:
    """Every rejection names the file and the 1-based line."""

    def _raises(self, tmp_path: Path, body: str) -> AppError[ModelTrainerErrorCode]:
        """Read a corpus expected to be refused and return the error.

        Args:
            tmp_path: Directory to write the corpus into.
            body: Exact corpus contents.

        Returns:
            The raised error, for the caller to assert on.
        """
        corpus = _write_raw(tmp_path / "bad.jsonl", body)
        with pytest.raises(AppError) as raised:
            read_corpus_documents([str(corpus)])
        error: AppError[ModelTrainerErrorCode] = raised.value
        assert error.code is ModelTrainerErrorCode.CORPUS_MALFORMED_RECORD
        return error

    def test_a_line_that_is_not_json(self, tmp_path: Path) -> None:
        """Truncated or hand-edited output lands here."""
        error = self._raises(tmp_path, '{"text": "unterminated\n')

        assert "is not valid JSON" in error.message

    def test_a_json_array(self, tmp_path: Path) -> None:
        """A record must be an object, not any JSON value."""
        error = self._raises(tmp_path, "[1, 2]\n")

        assert "is a JSON array, not an object" in error.message

    def test_a_json_string(self, tmp_path: Path) -> None:
        """A bare string is the shape a lines corpus would have."""
        error = self._raises(tmp_path, '"just a string"\n')

        assert "is a JSON string, not an object" in error.message

    def test_a_json_number(self, tmp_path: Path) -> None:
        """Covers the integer arm of the type namer."""
        error = self._raises(tmp_path, "42\n")

        assert "is a JSON number, not an object" in error.message

    def test_a_json_float(self, tmp_path: Path) -> None:
        """Covers the float arm, which is a separate isinstance."""
        error = self._raises(tmp_path, "1.5\n")

        assert "is a JSON number, not an object" in error.message

    def test_a_json_boolean(self, tmp_path: Path) -> None:
        """bool is checked before int, because bool IS an int in Python."""
        error = self._raises(tmp_path, "true\n")

        assert "is a JSON boolean, not an object" in error.message

    def test_a_json_null(self, tmp_path: Path) -> None:
        """Covers the fallthrough arm of the type namer."""
        error = self._raises(tmp_path, "null\n")

        assert "is a JSON null, not an object" in error.message

    def test_an_object_with_no_text_field(self, tmp_path: Path) -> None:
        """A record carrying only provenance trains on nothing."""
        error = self._raises(tmp_path, dump_json_str({"path": "a.py"}) + "\n")

        assert "carries no 'text' field" in error.message

    def test_a_text_field_that_is_not_a_string(self, tmp_path: Path) -> None:
        """A number where the body belongs is a broken emitter."""
        error = self._raises(tmp_path, dump_json_str({"text": 7}) + "\n")

        assert "has a JSON number 'text', not a string" in error.message

    def test_a_text_field_that_is_a_json_object(self, tmp_path: Path) -> None:
        """Covers the object arm of the type namer on the text field."""
        error = self._raises(tmp_path, dump_json_str({"text": {"a": 1}}) + "\n")

        assert "has a JSON object 'text', not a string" in error.message

    def test_an_empty_text_field(self, tmp_path: Path) -> None:
        """An empty document contributes only an end-of-sequence token."""
        error = self._raises(tmp_path, dump_json_str({"text": ""}) + "\n")

        assert "carries an empty 'text'" in error.message

    def test_the_message_names_the_file_and_line(self, tmp_path: Path) -> None:
        """A 4,000-record corpus needs a locator, not just a reason."""
        body = dump_json_str({"text": "fine"}) + "\n" + "{not json\n"
        error = self._raises(tmp_path, body)

        assert "bad.jsonl" in error.message
        assert "line 2" in error.message


class TestSplittingADocumentCorpus:
    """``split_corpus`` under the documents format."""

    def _cfg(self, path: Path, *, holdout: float, test: float) -> DatasetConfig:
        """Build a documents-mode config.

        Args:
            path: Corpus path.
            holdout: Validation fraction.
            test: Test fraction.

        Returns:
            The configuration.
        """
        return DatasetConfig(
            corpus_path=str(path),
            corpus_format="documents",
            holdout_fraction=holdout,
            test_split_ratio=test,
        )

    def test_documents_partition_disjointly(self, tmp_path: Path) -> None:
        """The split is over documents, not over lines within them."""
        texts = [f"def f{n}():\n    return {n}\n" for n in range(10)]
        corpus = _write_records(tmp_path / "c.jsonl", texts)

        split = split_corpus(self._cfg(corpus, holdout=0.1, test=0.2))

        assert split["train"] == tuple(texts[:7])
        assert split["validation"] == (texts[7],)
        assert split["test"] == tuple(texts[8:])

    def test_a_directory_with_no_jsonl_is_reported_as_empty(self, tmp_path: Path) -> None:
        """And the message says which suffix was looked for."""
        (tmp_path / "prose.txt").write_text("not this format\n", encoding="utf-8")

        with pytest.raises(AppError) as raised:
            split_corpus(self._cfg(tmp_path, holdout=0.1, test=0.1))

        error: AppError[ModelTrainerErrorCode] = raised.value
        assert error.code is ModelTrainerErrorCode.CORPUS_EMPTY
        assert ".jsonl" in error.message
        assert "documents" in error.message

    def test_a_jsonl_holding_no_records_is_reported_as_empty(self, tmp_path: Path) -> None:
        """Present but recordless is a different failure from absent."""
        _write_raw(tmp_path / "c.jsonl", "\n\n")

        with pytest.raises(AppError) as raised:
            split_corpus(self._cfg(tmp_path, holdout=0.1, test=0.1))

        error: AppError[ModelTrainerErrorCode] = raised.value
        assert error.code is ModelTrainerErrorCode.CORPUS_EMPTY
        assert "hold no document" in error.message

    def test_an_unsatisfiable_holdout_is_described_in_documents(self, tmp_path: Path) -> None:
        """The message must not call a document a line."""
        corpus = _write_records(tmp_path / "c.jsonl", ["a", "b"])

        with pytest.raises(AppError) as raised:
            split_corpus(self._cfg(corpus, holdout=0.5, test=0.5))

        error: AppError[ModelTrainerErrorCode] = raised.value
        assert error.code is ModelTrainerErrorCode.CORPUS_HOLDOUT_UNSATISFIABLE
        assert "document(s)" in error.message
        assert "line(s)" not in error.message


class TestUnitNaming:
    """The two message helpers, both arms each."""

    def test_documents_are_called_documents(self) -> None:
        """Guards the noun used in every documents-mode message."""
        assert unit_noun("documents") == "document"

    def test_lines_are_called_lines(self) -> None:
        """The unchanged arm, so a refactor cannot silently swap them."""
        assert unit_noun("lines") == "line"

    def test_documents_read_jsonl(self) -> None:
        """The suffix named in the empty-corpus message."""
        assert corpus_suffixes("documents") == ".jsonl"

    def test_lines_read_text_suffixes(self) -> None:
        """Matches what ``list_text_files`` actually accepts."""
        assert corpus_suffixes("lines") == ".txt/.text"


class _CharTok:
    """One token id per character, so packed ids are countable by hand.

    A real encoder satisfying the whole :class:`Encoder` protocol, returning
    the production :class:`ListEncoded` rather than a stand-in for it. The
    character mapping is what makes the assertions readable: every id in an
    expectation is an ``ord``, so a wrong pack is legible instead of opaque.
    """

    def encode(self: _CharTok, text: str) -> ListEncoded:
        """Encode text as one id per character.

        Args:
            text: Text to encode.

        Returns:
            The encoded ids.
        """
        return ListEncoded([ord(c) for c in text])

    def token_to_id(self: _CharTok, token: str) -> int | None:
        """Map a single-character token to its id.

        Args:
            token: Token to look up.

        Returns:
            The character's ordinal, or None when the token is not one char.
        """
        if len(token) == 1:
            return ord(token)
        return None

    def get_vocab_size(self: _CharTok) -> int:
        """Report the vocabulary size.

        Returns:
            The number of code points this encoder can emit.
        """
        return 0x110000

    def decode(self: _CharTok, ids: list[int]) -> str:
        """Turn ids back into text.

        Args:
            ids: Token ids to decode.

        Returns:
            The decoded string.
        """
        return "".join(chr(i) for i in ids)


class TestOneEndOfSequencePerDocument:
    """The dataset packs documents exactly as it packs lines.

    This is why the format needed no change to ``CausalLMDataset``: the packer
    already appends one end-of-sequence token per element of ``lines``, so
    feeding it whole documents terminates each document exactly once.
    """

    def test_each_document_is_terminated_exactly_once(self, tmp_path: Path) -> None:
        """Three 2-character documents pack to three blocks of three ids.

        Two characters plus one eos is three ids per document, and a max_len of
        three therefore packs to exactly three full blocks with no padding, so
        every id in the assertion is a real token rather than a pad.
        """
        texts = ["ab", "cd", "ef"]
        corpus = _write_records(tmp_path / "c.jsonl", texts)
        documents = read_corpus_documents([str(corpus)])

        dataset = CausalLMDataset(
            lines=documents, tokenizer=_CharTok(), max_len=3, eos_id=1, pad_id=0
        )

        assert len(dataset) == 3
        packed: list[int] = []
        for block in range(len(dataset)):
            input_ids, _ = dataset[block]
            packed.extend(int(input_ids[i].item()) for i in range(3))
        assert packed == [
            ord("a"),
            ord("b"),
            1,
            ord("c"),
            ord("d"),
            1,
            ord("e"),
            ord("f"),
            1,
        ]

    def test_a_documents_newlines_are_tokenized_not_stripped(self, tmp_path: Path) -> None:
        """The packer must see the whitespace the reader preserved."""
        corpus = _write_records(tmp_path / "c.jsonl", ["a\n  b"])
        documents = read_corpus_documents([str(corpus)])

        dataset = CausalLMDataset(
            lines=documents, tokenizer=_CharTok(), max_len=6, eos_id=1, pad_id=0
        )

        input_ids, _ = dataset[0]
        assert [int(input_ids[i].item()) for i in range(6)] == [
            ord("a"),
            ord("\n"),
            ord(" "),
            ord(" "),
            ord("b"),
            1,
        ]


class TestNarrowingTheFormat:
    """``as_corpus_format`` and ``require_corpus_format``."""

    def test_every_declared_format_narrows(self) -> None:
        """Iterating CORPUS_FORMATS keeps this test honest as it grows."""
        for declared in CORPUS_FORMATS:
            assert as_corpus_format(declared, "corpus_format") == declared

    def test_an_unknown_format_is_refused(self) -> None:
        """The set is closed; a typo must not reach the reader."""
        with pytest.raises(JSONTypeError) as raised:
            as_corpus_format("document", "corpus_format")

        assert "corpus_format" in str(raised.value)

    def test_a_required_field_is_read_and_narrowed(self) -> None:
        """The decoder path both decoders share."""
        assert require_corpus_format({"corpus_format": "documents"}, "corpus_format") == (
            "documents"
        )

    def test_an_absent_field_is_refused(self) -> None:
        """No default: omission is not a format."""
        with pytest.raises(JSONTypeError):
            require_corpus_format({}, "corpus_format")

    def test_a_non_string_field_is_refused(self) -> None:
        """require_str is what rejects this, and it must stay in the path."""
        with pytest.raises(JSONTypeError):
            require_corpus_format({"corpus_format": 1}, "corpus_format")
