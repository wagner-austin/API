"""Tests for the emit CLI: argument parsing and the end-to-end emission.

The integration tests run against real git repositories built in the test's
temporary directory, so every production hook -- git, file reads, file
writes -- runs for real, and what is asserted is what an operator's run
would produce.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from platform_core.json_utils import load_json_str

from code_corpus.cli.emit_corpus import main, parse_args
from code_corpus.contracts.corpus import (
    RepoPin,
    SourceFileRecord,
    decode_code_corpus_manifest,
    decode_source_file_record,
)
from tests.conftest import make_repo

A_PY = b"import pathlib\n\n\ndef f() -> int:\n    return 1\n"


def _decode_jsonl(path: pathlib.Path) -> list[SourceFileRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [decode_source_file_record(load_json_str(line)) for line in lines]


def _build_repos(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str, pathlib.Path, str]:
    repo1 = tmp_path / "repo1"
    head1 = make_repo(
        repo1,
        {
            "a.py": A_PY,
            "document_categories.py": b"X = 1\n",
            "empty.py": b"   \n",
            "notes.txt": b"hi\n",
            "dup.py": b"SHARED = True\n",
        },
    )
    repo2 = tmp_path / "repo2"
    head2 = make_repo(repo2, {"b.py": b"VALUE = 2\n", "copy.py": b"SHARED = True\n"})
    return repo1, head1, repo2, head2


def _run(repo1: pathlib.Path, repo2: pathlib.Path, out: pathlib.Path) -> int:
    return main(
        [
            "--repo",
            f"api={repo1}",
            "--repo",
            f"mcp={repo2}",
            "--out",
            str(out),
            "--holdout-fraction",
            "0.34",
            "--seed",
            "3",
        ]
    )


class TestParseArgs:
    def test_defaults(self) -> None:
        args = parse_args(["--repo", "api=.", "--out", "corpus.jsonl"])
        assert args.repos == (("api", pathlib.Path(".")),)
        assert args.out == pathlib.Path("corpus.jsonl")
        assert args.holdout_out == pathlib.Path("corpus.holdout.jsonl")
        assert args.holdout_fraction == 0.1
        assert args.seed == 0
        assert args.languages == ("python",)

    def test_derives_a_holdout_name_for_a_suffixless_output(self) -> None:
        args = parse_args(["--repo", "api=.", "--out", "corpus"])
        assert args.holdout_out == pathlib.Path("corpus.holdout")

    def test_explicit_values_override_the_defaults(self) -> None:
        args = parse_args(
            [
                "--repo",
                "api=repo",
                "--out",
                "c.jsonl",
                "--holdout-out",
                "h.jsonl",
                "--holdout-fraction",
                "0.25",
                "--seed",
                "42",
                "--language",
                "python",
            ]
        )
        assert args.holdout_out == pathlib.Path("h.jsonl")
        assert args.holdout_fraction == 0.25
        assert args.seed == 42
        assert args.languages == ("python",)

    def test_keeps_repos_in_command_line_order(self) -> None:
        args = parse_args(["--repo", "b=r1", "--repo", "a=r2", "--out", "c.jsonl"])
        assert args.repos == (("b", pathlib.Path("r1")), ("a", pathlib.Path("r2")))

    def test_rejects_an_unknown_argument(self) -> None:
        with pytest.raises(ValueError, match="unknown argument '--bogus'"):
            parse_args(["--repo", "api=.", "--out", "c.jsonl", "--bogus", "1"])

    def test_rejects_a_flag_without_its_value(self) -> None:
        with pytest.raises(ValueError, match="--out requires a value"):
            parse_args(["--repo", "api=.", "--out"])

    def test_rejects_a_missing_repo(self) -> None:
        with pytest.raises(ValueError, match="--repo is required"):
            parse_args(["--out", "c.jsonl"])

    def test_rejects_a_missing_out(self) -> None:
        with pytest.raises(ValueError, match="--out is required"):
            parse_args(["--repo", "api=."])

    def test_rejects_a_repeated_single_flag(self) -> None:
        with pytest.raises(ValueError, match="duplicate argument '--seed'"):
            parse_args(["--repo", "api=.", "--out", "c.jsonl", "--seed", "1", "--seed", "2"])

    def test_rejects_a_repo_without_an_equals(self) -> None:
        with pytest.raises(ValueError, match="--repo expects name=path, got 'api'"):
            parse_args(["--repo", "api", "--out", "c.jsonl"])

    @pytest.mark.parametrize("spec", ["=path", "name="])
    def test_rejects_an_empty_repo_name_or_path(self, spec: str) -> None:
        with pytest.raises(ValueError, match="--repo expects a non-empty name and path"):
            parse_args(["--repo", spec, "--out", "c.jsonl"])

    @pytest.mark.parametrize("name", ["a/b", "a\\b"])
    def test_rejects_a_slashed_repo_name(self, name: str) -> None:
        with pytest.raises(ValueError, match="--repo name must not contain a slash"):
            parse_args(["--repo", f"{name}=path", "--out", "c.jsonl"])

    def test_rejects_a_repeated_repo_name(self) -> None:
        with pytest.raises(ValueError, match="--repo names must be unique"):
            parse_args(["--repo", "api=r1", "--repo", "api=r2", "--out", "c.jsonl"])

    def test_rejects_a_repeated_language(self) -> None:
        with pytest.raises(ValueError, match="--language values must be unique"):
            args = ["--repo", "api=.", "--out", "c.jsonl"]
            parse_args([*args, "--language", "python", "--language", "python"])

    def test_rejects_a_non_numeric_fraction(self) -> None:
        with pytest.raises(ValueError, match="--holdout-fraction expects a number, got 'x'"):
            parse_args(["--repo", "api=.", "--out", "c.jsonl", "--holdout-fraction", "x"])

    def test_rejects_a_non_integer_seed(self) -> None:
        with pytest.raises(ValueError, match=re.escape("--seed expects an integer, got '1.5'")):
            parse_args(["--repo", "api=.", "--out", "c.jsonl", "--seed", "1.5"])


class TestEmitEndToEnd:
    def test_emits_corpus_holdout_and_manifest(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        repo1, head1, repo2, head2 = _build_repos(tmp_path)
        out = tmp_path / "corpus.jsonl"
        assert _run(repo1, repo2, out) == 0

        holdout_path = tmp_path / "corpus.holdout.jsonl"
        manifest_path = tmp_path / "corpus.jsonl.manifest.json"
        manifest = decode_code_corpus_manifest(
            load_json_str(manifest_path.read_text(encoding="utf-8"))
        )
        assert manifest["repos"] == [
            RepoPin(name="api", commit=head1, dirty=False),
            RepoPin(name="mcp", commit=head2, dirty=False),
        ]
        assert manifest["files_train"] == 2
        assert manifest["files_holdout"] == 1
        assert manifest["excluded_generated"] == 1
        assert manifest["excluded_duplicate"] == 1
        assert manifest["excluded_empty"] == 1
        assert manifest["train_output"] == "corpus.jsonl"
        assert manifest["holdout_output"] == "corpus.holdout.jsonl"
        assert manifest["seed"] == 3
        assert manifest["holdout_fraction"] == 0.34

        train = _decode_jsonl(out)
        holdout = _decode_jsonl(holdout_path)
        kept = sorted((record["repo"], record["path"]) for record in [*train, *holdout])
        assert kept == [("api", "a.py"), ("api", "dup.py"), ("mcp", "b.py")]

    def test_documents_carry_the_path_header_and_exact_content(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        repo1, _, repo2, _ = _build_repos(tmp_path)
        out = tmp_path / "corpus.jsonl"
        assert _run(repo1, repo2, out) == 0
        records = _decode_jsonl(out) + _decode_jsonl(tmp_path / "corpus.holdout.jsonl")
        by_path = {(record["repo"], record["path"]): record for record in records}
        assert by_path[("api", "a.py")]["text"] == "# api/a.py\n" + A_PY.decode("utf-8")

    def test_reports_the_run_in_summary_lines(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        repo1, head1, repo2, head2 = _build_repos(tmp_path)
        out = tmp_path / "corpus.jsonl"
        assert _run(repo1, repo2, out) == 0
        manifest = decode_code_corpus_manifest(
            load_json_str((tmp_path / "corpus.jsonl.manifest.json").read_text(encoding="utf-8"))
        )
        assert emitted == [
            f"repo             api @ {head1}",
            f"repo             mcp @ {head2}",
            "files            3 kept (2 train, 1 holdout)",
            "excluded         generated 1, duplicate 1, empty 1",
            f"tokens approx    {manifest['tokens_approx_train']:,} train, "
            f"{manifest['tokens_approx_holdout']:,} holdout",
            f"written          {out}",
            f"holdout          {tmp_path / 'corpus.holdout.jsonl'}",
            "manifest         corpus.jsonl.manifest.json",
        ]

    def test_two_runs_with_one_seed_emit_identical_bytes(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        repo1, _, repo2, _ = _build_repos(tmp_path)
        first = tmp_path / "one" / "corpus.jsonl"
        second = tmp_path / "two" / "corpus.jsonl"
        assert _run(repo1, repo2, first) == 0
        assert _run(repo1, repo2, second) == 0
        assert first.read_bytes() == second.read_bytes()
        first_holdout = tmp_path / "one" / "corpus.holdout.jsonl"
        second_holdout = tmp_path / "two" / "corpus.holdout.jsonl"
        assert first_holdout.read_bytes() == second_holdout.read_bytes()

    def test_refuses_a_dirty_working_tree_and_writes_nothing(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        """Against a REAL git repository, not the faked seam.

        This asserted the opposite until 2026-09-03: it checked that a dirty
        tree was recorded as dirty and let the emission proceed. It WAS
        recorded, and it was ignored -- code-corpus-v1 shipped with both
        repositories dirty, and was trained on, evaluated and reported.

        Nothing is written, which is the half that matters: a corpus on disk
        gets used, and a manifest saying it is unreproducible stops nobody.

        Args:
            tmp_path: Temporary directory holding two real repositories.
            emitted: Captured summary lines.
        """
        repo1, _, repo2, _ = _build_repos(tmp_path)
        (repo1 / "scratch.py").write_bytes(b"SCRATCH = True\n")
        out = tmp_path / "corpus.jsonl"

        with pytest.raises(ValueError, match="refusing to emit: api has uncommitted changes"):
            _ = _run(repo1, repo2, out)

        assert not out.exists()
        assert not (tmp_path / "corpus.jsonl.manifest.json").exists()

    def test_a_clean_pair_of_repositories_still_emits(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        """The refusal must not have closed the ordinary path.

        Args:
            tmp_path: Temporary directory holding two real repositories.
            emitted: Captured summary lines.
        """
        repo1, head1, repo2, _ = _build_repos(tmp_path)
        out = tmp_path / "corpus.jsonl"

        assert _run(repo1, repo2, out) == 0

        manifest = decode_code_corpus_manifest(
            load_json_str((tmp_path / "corpus.jsonl.manifest.json").read_text(encoding="utf-8"))
        )
        assert manifest["repos"][0] == RepoPin(name="api", commit=head1, dirty=False)

    def test_refuses_to_emit_an_empty_corpus(self, tmp_path: pathlib.Path) -> None:
        repo = tmp_path / "repo"
        make_repo(repo, {"notes.txt": b"hi\n"})
        with pytest.raises(ValueError, match="no files to split; nothing to emit"):
            main(["--repo", f"api={repo}", "--out", str(tmp_path / "corpus.jsonl")])

    def test_refuses_an_unknown_language(self, tmp_path: pathlib.Path) -> None:
        repo = tmp_path / "repo"
        make_repo(repo, {"a.py": b"x = 1\n"})
        with pytest.raises(ValueError, match="unknown language 'go'"):
            main(
                [
                    "--repo",
                    f"api={repo}",
                    "--out",
                    str(tmp_path / "corpus.jsonl"),
                    "--language",
                    "go",
                ]
            )
