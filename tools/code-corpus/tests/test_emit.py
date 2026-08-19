"""Tests for document rendering, deduplication, splitting and the manifest."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.json_utils import dump_json_str

from code_corpus.contracts.corpus import (
    LanguageStats,
    RepoPin,
    SourceFileRecord,
    decode_code_corpus_manifest,
    encode_code_corpus_manifest,
    encode_source_file_record,
)
from code_corpus.core import _test_hooks as core_hooks
from code_corpus.core.emit import (
    build_manifest,
    dedup_files,
    jsonl_text,
    language_stats,
    render_document,
    split_holdout,
    to_record,
)
from code_corpus.core.select import SelectedFile, approx_tokens


def _file(
    path: str, *, repo: str = "api", text: str = "x = 1\n", language: str = "python"
) -> SelectedFile:
    return SelectedFile(
        repo=repo,
        path=path,
        language=language,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        tokens_approx=approx_tokens(text),
        text=text,
    )


class TestRenderDocument:
    def test_prefixes_the_path_header(self) -> None:
        document = render_document(_file("src/m.py", text="a = 1\n"))
        assert document == "# api/src/m.py\na = 1\n"

    def test_terminates_an_unterminated_file(self) -> None:
        document = render_document(_file("src/m.py", text="a = 1"))
        assert document == "# api/src/m.py\na = 1\n"


class TestDedup:
    def test_first_occurrence_wins(self) -> None:
        files = [
            _file("first.py", text="SHARED = True\n"),
            _file("other.py", text="OTHER = True\n"),
            _file("copy.py", repo="mcp", text="SHARED = True\n"),
        ]
        kept, duplicates = dedup_files(files)
        assert [file.path for file in kept] == ["first.py", "other.py"]
        assert duplicates == 1


class TestSplitHoldout:
    def test_is_reproducible_from_the_seed(self) -> None:
        files = [_file(f"m{index}.py", text=f"x = {index}\n") for index in range(10)]
        first_train, first_holdout = split_holdout(files, fraction=0.3, seed=7)
        second_train, second_holdout = split_holdout(files, fraction=0.3, seed=7)
        assert [file.path for file in first_train] == [file.path for file in second_train]
        assert [file.path for file in first_holdout] == [file.path for file in second_holdout]

    def test_partitions_without_loss_or_overlap(self) -> None:
        files = [_file(f"m{index}.py", text=f"x = {index}\n") for index in range(10)]
        train, holdout = split_holdout(files, fraction=0.3, seed=7)
        assert len(train) == 7
        assert len(holdout) == 3
        together = sorted(file.path for file in [*train, *holdout])
        assert together == sorted(file.path for file in files)

    def test_holdout_is_sorted_by_repo_and_path(self) -> None:
        files = [_file(f"m{index}.py", text=f"x = {index}\n") for index in range(10)]
        _, holdout = split_holdout(files, fraction=0.3, seed=7)
        paths = [(file.repo, file.path) for file in holdout]
        assert paths == sorted(paths)

    def test_a_zero_fraction_holds_nothing_out(self) -> None:
        files = [_file(f"m{index}.py", text=f"x = {index}\n") for index in range(4)]
        train, holdout = split_holdout(files, fraction=0.0, seed=7)
        assert holdout == []
        assert sorted(file.path for file in train) == ["m0.py", "m1.py", "m2.py", "m3.py"]

    @pytest.mark.parametrize("fraction", [-0.1, 1.0])
    def test_rejects_a_fraction_outside_the_unit_interval(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="holdout fraction must be in"):
            split_holdout([_file("m.py")], fraction=fraction, seed=0)

    def test_rejects_an_empty_file_list(self) -> None:
        with pytest.raises(ValueError, match="no files to split; nothing to emit"):
            split_holdout([], fraction=0.1, seed=0)

    def test_rejects_a_fraction_that_selects_zero_files(self) -> None:
        files = [_file(f"m{index}.py") for index in range(5)]
        with pytest.raises(ValueError, match="selects zero holdout"):
            split_holdout(files, fraction=0.1, seed=0)


class TestRecordsAndJsonl:
    def test_record_carries_the_rendered_document(self) -> None:
        record = to_record(_file("src/m.py", text="a = 1\n"))
        assert record == SourceFileRecord(
            repo="api",
            path="src/m.py",
            language="python",
            sha256=hashlib.sha256(b"a = 1\n").hexdigest(),
            tokens_approx=1,
            text="# api/src/m.py\na = 1\n",
        )

    def test_jsonl_is_one_compact_object_per_line(self) -> None:
        record = to_record(_file("src/m.py"))
        expected = dump_json_str(encode_source_file_record(record)) + "\n"
        assert jsonl_text([record]) == expected

    def test_jsonl_of_nothing_is_empty(self) -> None:
        assert jsonl_text([]) == ""


class TestLanguageStats:
    def test_totals_by_language_in_sorted_order(self) -> None:
        files = [
            _file("a.rs", language="rust", text="fn main() {}\n"),
            _file("b.py", text="b = 1\n"),
            _file("c.py", text="c = 22\n"),
        ]
        stats = language_stats(files)
        assert list(stats) == ["python", "rust"]
        assert stats["python"] == LanguageStats(files=2, tokens_approx=2)
        assert stats["rust"] == LanguageStats(files=1, tokens_approx=3)


class TestBuildManifest:
    def test_digests_the_bytes_actually_written(self, tmp_path: pathlib.Path) -> None:
        train = [_file("a.py", text="a = 1\n"), _file("b.py", text="b = 2\n")]
        holdout = [_file("c.py", text="c = 3\n")]
        train_path = tmp_path / "corpus.jsonl"
        holdout_path = tmp_path / "corpus.holdout.jsonl"
        core_hooks.write_text(train_path, jsonl_text([to_record(file) for file in train]))
        core_hooks.write_text(holdout_path, jsonl_text([to_record(file) for file in holdout]))

        manifest = build_manifest(
            train_output=train_path,
            holdout_output=holdout_path,
            seed=5,
            holdout_fraction=0.34,
            repos=[RepoPin(name="api", commit="c" * 40, dirty=False)],
            train=train,
            holdout=holdout,
            excluded_generated=2,
            excluded_duplicate=1,
            excluded_empty=0,
        )
        assert manifest["train_sha256"] == hashlib.sha256(train_path.read_bytes()).hexdigest()
        assert manifest["holdout_sha256"] == hashlib.sha256(holdout_path.read_bytes()).hexdigest()
        assert manifest["files_train"] == 2
        assert manifest["files_holdout"] == 1
        assert manifest["languages"] == {"python": LanguageStats(files=3, tokens_approx=3)}
        assert manifest["tokens_approx_train"] == 2
        assert manifest["tokens_approx_holdout"] == 1

    def test_an_emitted_manifest_satisfies_its_own_contract(self, tmp_path: pathlib.Path) -> None:
        train = [_file("a.py", text="a = 1\n")]
        holdout = [_file("c.py", text="c = 3\n")]
        train_path = tmp_path / "corpus.jsonl"
        holdout_path = tmp_path / "corpus.holdout.jsonl"
        core_hooks.write_text(train_path, jsonl_text([to_record(file) for file in train]))
        core_hooks.write_text(holdout_path, jsonl_text([to_record(file) for file in holdout]))

        manifest = build_manifest(
            train_output=train_path,
            holdout_output=holdout_path,
            seed=5,
            holdout_fraction=0.5,
            repos=[RepoPin(name="api", commit="c" * 40, dirty=True)],
            train=train,
            holdout=holdout,
            excluded_generated=0,
            excluded_duplicate=0,
            excluded_empty=0,
        )
        encoded = encode_code_corpus_manifest(manifest)
        assert decode_code_corpus_manifest(encoded) == manifest
