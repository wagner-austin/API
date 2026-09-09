"""The entry that materialises a reshaped corpus, over real files on disk.

NOTHING IS FAKED HERE, AND THAT IS THE POINT. The one seam this entry has --
:data:`~model_trainer.cli._test_hooks.read_corpus_documents` -- is faked
elsewhere because it stands in for a wiki checkout that a GPU benchmark cannot
carry. This module wants the opposite: the whole claim of the reshaping is
that the second corpus directory it writes can be handed to the UNCHANGED
benchmark, and a fake reader would test that claim against a reader the
benchmark does not use. So the documents are written as markdown into
``tmp_path`` and read back by the production reader, frontmatter and all.

WHAT IS ASSERTED, since a delete-only rule already has its own suite in
`test_corpus_reshape.py`: this file is about the MATERIALISATION -- that the
filenames reproduce the read order, that the returned digest is the digest of
what was written rather than of what was read, and that a document which
cannot survive the gate stops the run instead of quietly shrinking the corpus.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.cli import reshape_corpus as cli
from model_trainer.core.services.model.cartridge_plans import corpus_digest
from model_trainer.core.services.model.corpus_reshape import reshape_document

#: Prose plus every kind of scaffolding the splitter removes, so the terms
#: this page loses are a fact the report has to be able to state.
_MESSY = """---
title: Twelve backends
---

## Twelve backends behind one interface

Seven classifiers sit behind one interface, XGBoost and ClearGBM among them.

| backend | language |
|---|---|
| TankpitBot | Python |

The team measured NavProbe against the usual baseline over many weeks.

```python
DTypeLike = str
```
"""

#: Pure prose. Nothing in it is scaffolding, so reshaping costs it nothing --
#: which is the other half of the terms-lost branch.
_CLEAN = "The registry lists NavProbe beside the other clients. It has done so for weeks.\n"

#: A table and nothing else. Its body is not empty, so the corpus reader keeps
#: it; its reshape is empty, so the gate refuses it. That gap is the only way
#: a real corpus reaches the refusal, and it is why this document is a table
#: rather than a blank file.
_SCAFFOLDING_ONLY = "| backend | language |\n|---|---|\n| ClearGBM | Rust |\n"


def _write_corpus(root: pathlib.Path, bodies: dict[str, str]) -> pathlib.Path:
    """Write a corpus directory the production reader can read.

    Args:
        root: Directory to create the corpus under.
        bodies: Filename to document text. The reader sorts by filename, so
            the keys decide the order every downstream assertion is about.

    Returns:
        The corpus directory.
    """
    corpus = root / "corpus"
    corpus.mkdir()
    for name, body in bodies.items():
        (corpus / name).write_text(body, encoding="utf-8")
    return corpus


class TestWriteReshapedCorpus:
    def test_it_writes_one_document_per_source_in_read_order(self, tmp_path: pathlib.Path) -> None:
        """Order decides which windows the stride holds out, so a reshaped
        corpus that reordered its pages would be a different experiment even
        though it held the same sentences.
        """
        corpus = _write_corpus(tmp_path, {"b.md": _CLEAN, "a.md": _MESSY})
        out = tmp_path / "reshaped"

        cli.write_reshaped_corpus(corpus=corpus, out=out)

        written = sorted(path.name for path in out.glob("*.md"))
        assert written == ["doc000.md", "doc001.md"]
        assert "Seven classifiers" in (out / "doc000.md").read_text(encoding="utf-8")
        assert "The registry lists NavProbe" in (out / "doc001.md").read_text(encoding="utf-8")

    def test_the_written_documents_are_the_reshaped_text(self, tmp_path: pathlib.Path) -> None:
        """The scaffolding the splitter drops must be absent from the FILE,
        not merely from the report -- the benchmark reads the file.
        """
        corpus = _write_corpus(tmp_path, {"a.md": _MESSY})

        cli.write_reshaped_corpus(corpus=corpus, out=tmp_path / "reshaped")

        body = (tmp_path / "reshaped" / "doc000.md").read_text(encoding="utf-8")
        assert "DTypeLike" not in body
        assert "TankpitBot" not in body
        assert "| Python |" not in body

    def test_no_frontmatter_is_written_back(self, tmp_path: pathlib.Path) -> None:
        """Adding YAML back would be adding text the source's body never had,
        which is exactly the invention the reshaping is built to avoid.
        """
        corpus = _write_corpus(tmp_path, {"a.md": _MESSY})

        cli.write_reshaped_corpus(corpus=corpus, out=tmp_path / "reshaped")

        assert (
            not (tmp_path / "reshaped" / "doc000.md").read_text(encoding="utf-8").startswith("---")
        )

    def test_the_returned_digest_is_the_digest_of_what_was_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        """THE IDENTITY OF THE SECOND QUESTION SET. Every record downstream
        carries this digest to say which corpus it answered, so it has to be
        the reshaped text's digest and not the source's.
        """
        corpus = _write_corpus(tmp_path, {"a.md": _MESSY, "b.md": _CLEAN})

        digest = cli.write_reshaped_corpus(corpus=corpus, out=tmp_path / "reshaped")

        bodies = tuple(
            (tmp_path / "reshaped" / f"doc{index:03d}.md").read_text(encoding="utf-8")
            for index in (0, 1)
        )
        assert digest == corpus_digest(bodies)
        assert digest != corpus_digest((_MESSY, _CLEAN))

    def test_a_corpus_that_loses_nothing_still_reshapes(self, tmp_path: pathlib.Path) -> None:
        """A page of pure prose costs the reshaping nothing, and the run must
        not depend on there being something to report as lost.
        """
        corpus = _write_corpus(tmp_path, {"a.md": _CLEAN})

        digest = cli.write_reshaped_corpus(corpus=corpus, out=tmp_path / "reshaped")

        assert digest == corpus_digest((reshape_document(_CLEAN.strip()),))

    def test_the_output_directory_is_created(self, tmp_path: pathlib.Path) -> None:
        corpus = _write_corpus(tmp_path, {"a.md": _CLEAN})
        out = tmp_path / "nested" / "reshaped"

        cli.write_reshaped_corpus(corpus=corpus, out=out)

        assert (out / "doc000.md").is_file()

    def test_a_document_that_cannot_survive_the_gate_stops_the_run(
        self, tmp_path: pathlib.Path
    ) -> None:
        """REFUSED RATHER THAN SKIPPED. A corpus missing one page is a
        different corpus, and a run against it would report a number for a
        text nobody chose.
        """
        corpus = _write_corpus(tmp_path, {"a.md": _CLEAN, "b.md": _SCAFFOLDING_ONLY})

        with pytest.raises(AppError) as raised:
            cli.write_reshaped_corpus(corpus=corpus, out=tmp_path / "reshaped")

        assert raised.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE
        assert "1 of 2 documents" in raised.value.message
        assert "document 1: document_is_not_empty" in raised.value.message

    def test_nothing_is_written_when_a_document_is_refused(self, tmp_path: pathlib.Path) -> None:
        """A half-written corpus directory would be readable by the
        benchmark, and would be the corpus nobody chose.
        """
        corpus = _write_corpus(tmp_path, {"a.md": _CLEAN, "b.md": _SCAFFOLDING_ONLY})
        out = tmp_path / "reshaped"

        with pytest.raises(AppError):
            cli.write_reshaped_corpus(corpus=corpus, out=out)

        assert not out.exists()


class TestMain:
    def test_it_writes_the_corpus_and_exits_zero(self, tmp_path: pathlib.Path) -> None:
        corpus = _write_corpus(tmp_path, {"a.md": _MESSY})
        out = tmp_path / "reshaped"

        code = cli.main(["--corpus", str(corpus), "--out", str(out)])

        assert code == 0
        assert (out / "doc000.md").is_file()

    def test_it_reads_the_process_arguments_when_given_none(self, tmp_path: pathlib.Path) -> None:
        """The argv-less path is how the console script reaches it, so it is
        not the same code path as the one every other test here takes.
        """
        corpus = _write_corpus(tmp_path, {"a.md": _MESSY})
        out = tmp_path / "reshaped"
        saved = sys.argv
        sys.argv = ["modeltrainer-reshape-corpus", "--corpus", str(corpus), "--out", str(out)]
        try:
            code = cli.main()
        finally:
            sys.argv = saved

        assert code == 0
        assert (out / "doc000.md").is_file()

    def test_a_missing_corpus_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--corpus"):
            cli.main(["--out", str(tmp_path / "reshaped")])

    def test_a_missing_out_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Parsed after ``--corpus``, so it needs its own case or the second
        refusal is never reached.
        """
        corpus = _write_corpus(tmp_path, {"a.md": _CLEAN})

        with pytest.raises(ValueError, match="--out"):
            cli.main(["--corpus", str(corpus)])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError):
            cli.main(["--corpus", str(tmp_path), "--plan", "gpt2-wiki-qa"])


class TestTheEntryPoint:
    def test_the_console_entry_point_exits_zero(self, tmp_path: pathlib.Path) -> None:
        corpus = _write_corpus(tmp_path, {"a.md": _MESSY})
        out = tmp_path / "reshaped"
        saved = sys.argv
        sys.argv = ["modeltrainer-reshape-corpus", "--corpus", str(corpus), "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as raised:
                cli.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0
        assert (out / "doc000.md").is_file()

    def test_running_it_as_a_module_actually_reshapes(self, tmp_path: pathlib.Path) -> None:
        """Without the __main__ guard the module imports, runs nothing and
        exits 0 -- which looks exactly like a reshaping that wrote an empty
        corpus.
        """
        module_name = "model_trainer.cli.reshape_corpus"
        corpus = _write_corpus(tmp_path, {"a.md": _MESSY})
        out = tmp_path / "reshaped"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", "--corpus", str(corpus), "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert (out / "doc000.md").is_file()
