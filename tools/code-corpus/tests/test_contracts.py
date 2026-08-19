"""Tests for the corpus contracts.

Every decode invariant gets a value that violates it, because a validator
that accepts everything is a validator in name only. Round trips go through
real JSON serialisation, so what the manifest file will actually hold is what
is proven to decode.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str, load_json_str

from code_corpus.contracts.corpus import (
    CodeCorpusManifest,
    LanguageStats,
    RepoPin,
    SourceFileRecord,
    decode_code_corpus_manifest,
    decode_language_stats,
    decode_repo_pin,
    decode_source_file_record,
    encode_code_corpus_manifest,
    encode_language_stats,
    encode_repo_pin,
    encode_source_file_record,
)


def _record() -> SourceFileRecord:
    return SourceFileRecord(
        repo="api",
        path="libs/platform_core/src/platform_core/errors.py",
        language="python",
        sha256="a" * 64,
        tokens_approx=12,
        text="# api/libs/platform_core/src/platform_core/errors.py\nx = 1\n",
    )


def _pin() -> RepoPin:
    return RepoPin(name="api", commit="c" * 40, dirty=False)


def _manifest() -> CodeCorpusManifest:
    return CodeCorpusManifest(
        train_output="corpus.jsonl",
        train_sha256="a" * 64,
        holdout_output="corpus.holdout.jsonl",
        holdout_sha256="b" * 64,
        seed=7,
        holdout_fraction=0.25,
        repos=[_pin(), RepoPin(name="mcp", commit="d" * 40, dirty=True)],
        files_train=3,
        files_holdout=1,
        excluded_generated=1,
        excluded_duplicate=2,
        excluded_empty=0,
        languages={"python": LanguageStats(files=4, tokens_approx=100)},
        tokens_approx_train=75,
        tokens_approx_holdout=25,
    )


def _manifest_json(**overrides: JSONValue) -> JSONValue:
    obj = encode_code_corpus_manifest(_manifest())
    for key, value in overrides.items():
        obj[key] = value
    return obj


class TestRoundTrips:
    def test_record_survives_json(self) -> None:
        raw = dump_json_str(encode_source_file_record(_record()))
        assert decode_source_file_record(load_json_str(raw)) == _record()

    def test_pin_survives_json(self) -> None:
        raw = dump_json_str(encode_repo_pin(_pin()))
        assert decode_repo_pin(load_json_str(raw)) == _pin()

    def test_language_stats_survive_json(self) -> None:
        stats = LanguageStats(files=4, tokens_approx=100)
        raw = dump_json_str(encode_language_stats(stats))
        assert decode_language_stats(load_json_str(raw), "python") == stats

    def test_manifest_survives_json(self) -> None:
        raw = dump_json_str(encode_code_corpus_manifest(_manifest()), indent=2)
        assert decode_code_corpus_manifest(load_json_str(raw)) == _manifest()

    def test_manifest_accepts_an_integer_zero_fraction(self) -> None:
        obj = _manifest_json(
            holdout_fraction=0,
            files_holdout=0,
            files_train=4,
            tokens_approx_train=100,
            tokens_approx_holdout=0,
        )
        assert decode_code_corpus_manifest(obj)["holdout_fraction"] == 0.0


class TestRecordInvariants:
    def test_rejects_a_non_object(self) -> None:
        with pytest.raises(JSONTypeError, match="source file record must be a JSON object"):
            decode_source_file_record("nope")

    def test_rejects_a_missing_field(self) -> None:
        obj = encode_source_file_record(_record())
        del obj["path"]
        with pytest.raises(JSONTypeError, match="Missing required field 'path'"):
            decode_source_file_record(obj)

    def test_rejects_an_empty_repo(self) -> None:
        obj = encode_source_file_record(_record())
        obj["repo"] = ""
        with pytest.raises(JSONTypeError, match="Field 'repo' must not be empty"):
            decode_source_file_record(obj)

    @pytest.mark.parametrize("repo", ["a/b", "a\\b"])
    def test_rejects_a_slashed_repo_name(self, repo: str) -> None:
        obj = encode_source_file_record(_record())
        obj["repo"] = repo
        with pytest.raises(JSONTypeError, match="Field 'repo' must not contain a slash"):
            decode_source_file_record(obj)

    def test_rejects_a_backslashed_path(self) -> None:
        obj = encode_source_file_record(_record())
        obj["path"] = "src\\m.py"
        with pytest.raises(JSONTypeError, match="Field 'path' must be forward-slashed"):
            decode_source_file_record(obj)

    def test_rejects_an_absolute_path(self) -> None:
        obj = encode_source_file_record(_record())
        obj["path"] = "/etc/m.py"
        with pytest.raises(JSONTypeError, match="Field 'path' must be repository-relative"):
            decode_source_file_record(obj)

    def test_rejects_an_escaping_path(self) -> None:
        obj = encode_source_file_record(_record())
        obj["path"] = "src/../../m.py"
        with pytest.raises(JSONTypeError, match="Field 'path' must not escape its repository"):
            decode_source_file_record(obj)

    @pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64])
    def test_rejects_a_malformed_digest(self, digest: str) -> None:
        obj = encode_source_file_record(_record())
        obj["sha256"] = digest
        with pytest.raises(JSONTypeError, match="Field 'sha256' must be 64 lowercase hex"):
            decode_source_file_record(obj)

    def test_rejects_a_zero_token_estimate(self) -> None:
        obj = encode_source_file_record(_record())
        obj["tokens_approx"] = 0
        with pytest.raises(JSONTypeError, match="Field 'tokens_approx' must be at least 1"):
            decode_source_file_record(obj)

    def test_rejects_an_empty_document(self) -> None:
        obj = encode_source_file_record(_record())
        obj["text"] = ""
        with pytest.raises(JSONTypeError, match="Field 'text' must not be empty"):
            decode_source_file_record(obj)


class TestPinInvariants:
    def test_rejects_a_non_object(self) -> None:
        with pytest.raises(JSONTypeError, match="repo pin must be a JSON object"):
            decode_repo_pin([])

    def test_rejects_a_short_commit(self) -> None:
        obj = encode_repo_pin(_pin())
        obj["commit"] = "c" * 39
        with pytest.raises(JSONTypeError, match="Field 'commit' must be 40 lowercase hex"):
            decode_repo_pin(obj)

    def test_rejects_a_slashed_name(self) -> None:
        obj = encode_repo_pin(_pin())
        obj["name"] = "a/b"
        with pytest.raises(JSONTypeError, match="Field 'name' must not contain a slash"):
            decode_repo_pin(obj)


class TestLanguageStatsInvariants:
    def test_rejects_a_non_object_naming_the_language(self) -> None:
        with pytest.raises(JSONTypeError, match="language stats for 'python' must be a JSON"):
            decode_language_stats(3, "python")

    def test_rejects_zero_files(self) -> None:
        obj = encode_language_stats(LanguageStats(files=1, tokens_approx=10))
        obj["files"] = 0
        with pytest.raises(JSONTypeError, match="Field 'files' for language 'python'"):
            decode_language_stats(obj, "python")

    def test_rejects_zero_tokens(self) -> None:
        obj = encode_language_stats(LanguageStats(files=1, tokens_approx=10))
        obj["tokens_approx"] = 0
        with pytest.raises(JSONTypeError, match="Field 'tokens_approx' for language 'python'"):
            decode_language_stats(obj, "python")


class TestManifestInvariants:
    def test_rejects_a_non_object(self) -> None:
        with pytest.raises(JSONTypeError, match="code corpus manifest must be a JSON object"):
            decode_code_corpus_manifest(None)

    @pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
    def test_rejects_a_fraction_outside_the_unit_interval(self, fraction: float) -> None:
        obj = _manifest_json(holdout_fraction=fraction)
        with pytest.raises(JSONTypeError, match="Field 'holdout_fraction' must be in"):
            decode_code_corpus_manifest(obj)

    def test_rejects_a_positive_fraction_with_no_holdout_files(self) -> None:
        obj = _manifest_json(files_holdout=0)
        with pytest.raises(JSONTypeError, match="Field 'files_holdout' must be positive"):
            decode_code_corpus_manifest(obj)

    def test_rejects_holdout_files_with_a_zero_fraction(self) -> None:
        obj = _manifest_json(holdout_fraction=0.0)
        with pytest.raises(JSONTypeError, match="Field 'files_holdout' must be 0"):
            decode_code_corpus_manifest(obj)

    def test_rejects_a_negative_count(self) -> None:
        obj = _manifest_json(excluded_generated=-1)
        with pytest.raises(JSONTypeError, match="Field 'excluded_generated' must not be negative"):
            decode_code_corpus_manifest(obj)

    def test_rejects_an_empty_repo_list(self) -> None:
        obj = _manifest_json(repos=[])
        with pytest.raises(JSONTypeError, match="Field 'repos' must not be empty"):
            decode_code_corpus_manifest(obj)

    def test_rejects_a_repeated_repo_name(self) -> None:
        obj = _manifest_json(repos=[encode_repo_pin(_pin()), encode_repo_pin(_pin())])
        with pytest.raises(JSONTypeError, match="Field 'repos' must not repeat a name"):
            decode_code_corpus_manifest(obj)

    def test_rejects_an_empty_language_map(self) -> None:
        obj = _manifest_json(languages={})
        with pytest.raises(JSONTypeError, match="Field 'languages' must not be empty"):
            decode_code_corpus_manifest(obj)

    def test_rejects_a_file_total_that_disagrees_with_the_split(self) -> None:
        obj = _manifest_json(files_train=2)
        with pytest.raises(JSONTypeError, match="counts 4 files, but the split holds 3"):
            decode_code_corpus_manifest(obj)

    def test_rejects_a_token_total_that_disagrees_with_the_split(self) -> None:
        obj = _manifest_json(tokens_approx_train=74)
        with pytest.raises(JSONTypeError, match="counts 100 tokens, but the split holds 99"):
            decode_code_corpus_manifest(obj)
