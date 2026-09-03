"""The production side of the cartridge measurement's hooks.

``test_cartridge_benchmark`` replaces these with fakes, which is the point of
a hook and also means the real implementations run in no test there. They read
the filesystem and decide what a "document" is, and that decision is a
measurement choice rather than plumbing: a corpus read with its frontmatter
attached trains a cartridge partly on YAML and reports the gain as a gain at
prose.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.core.services.model.cartridge_plans import CARTRIDGE_PLANS

_FRONTMATTER = "---\ntitle: A page\ntags: [x]\n---\n"


class TestReadCorpusDocuments:
    def test_it_reads_markdown_in_filename_order(self, tmp_path: pathlib.Path) -> None:
        """Sorted, because the order decides which windows the stride holds out.

        An unsorted directory listing would make the held-out set depend on
        the filesystem, so two machines would score different text and call
        the difference a result.
        """
        (tmp_path / "b.md").write_text("second body", encoding="utf-8")
        (tmp_path / "a.md").write_text("first body", encoding="utf-8")

        assert cli_hooks.read_corpus_documents(tmp_path) == ("first body", "second body")

    def test_frontmatter_is_removed(self, tmp_path: pathlib.Path) -> None:
        """A cartridge trained over frontmatter learns to predict `tags:`.

        The gain it then reports is real and is partly a gain at YAML, which
        is not the thing anybody asked about.
        """
        (tmp_path / "page.md").write_text(f"{_FRONTMATTER}The body text.", encoding="utf-8")

        assert cli_hooks.read_corpus_documents(tmp_path) == ("The body text.",)

    def test_a_document_without_frontmatter_is_all_body(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "page.md").write_text("Just prose, no fence.", encoding="utf-8")

        assert cli_hooks.read_corpus_documents(tmp_path) == ("Just prose, no fence.",)

    def test_an_unterminated_fence_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Refused rather than treated as body.

        Treating it as body would silently train on the YAML this reader
        exists to remove -- the failure it is meant to prevent, arriving
        through the path meant to prevent it.
        """
        (tmp_path / "broken.md").write_text("---\ntitle: never closed\n", encoding="utf-8")

        with pytest.raises(AppError) as excinfo:
            cli_hooks.read_corpus_documents(tmp_path)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE
        assert "never closes it" in excinfo.value.message

    def test_a_document_that_is_only_frontmatter_is_dropped(self, tmp_path: pathlib.Path) -> None:
        """It contributes no window, so carrying it would put an entry in the
        corpus digest that no measurement can see."""
        (tmp_path / "empty.md").write_text(_FRONTMATTER, encoding="utf-8")
        (tmp_path / "real.md").write_text(f"{_FRONTMATTER}Actual prose.", encoding="utf-8")

        assert cli_hooks.read_corpus_documents(tmp_path) == ("Actual prose.",)

    def test_non_markdown_is_ignored(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "notes.txt").write_text("not part of the corpus", encoding="utf-8")
        (tmp_path / "page.md").write_text("part of it", encoding="utf-8")

        assert cli_hooks.read_corpus_documents(tmp_path) == ("part of it",)

    def test_a_directory_with_no_bodies_is_refused(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "empty.md").write_text(_FRONTMATTER, encoding="utf-8")

        with pytest.raises(AppError) as excinfo:
            cli_hooks.read_corpus_documents(tmp_path)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE
        assert "needs text to train on" in excinfo.value.message

    def test_an_empty_directory_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(AppError) as excinfo:
            cli_hooks.read_corpus_documents(tmp_path)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE


class TestCartridgePlansHook:
    def test_the_default_is_the_declared_table(self) -> None:
        """Identity, not equality: a copy would let the two drift apart."""
        assert cli_hooks.cartridge_plans() is CARTRIDGE_PLANS
