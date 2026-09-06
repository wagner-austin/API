"""Tests for the lint step that runs both wiki gates.

The step exists because the rules were reachable only from pytest.
These tests hold it to the two things that makes it worth having: it
refuses when either gate refuses, and it refuses when BOTH do rather
than stopping at the first.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.wiki_gates import PROJECT_ROOT, main, run


def _page(root: Path, name: str, text: str) -> None:
    """Write one wiki page into a synthetic tree.

    Args:
        root: Project root to build ``wiki/pages`` beneath.
        name: Page file name.
        text: Page markdown.
    """
    pages = root / "wiki" / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    (pages / name).write_text(text, encoding="utf-8")


class TestTheStepRefuses:
    """A gate that cannot fail is not a gate."""

    def test_a_structurally_broken_page_fails_the_step(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Frontmatter the structure rule rejects."""
        _page(tmp_path, "broken.md", "# No frontmatter at all\n")

        code = run(tmp_path)
        out = capsys.readouterr().out

        assert code == 1
        assert "structure violation(s)" in out

    def test_a_claim_naming_a_missing_symbol_fails_the_step(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A claim block binding a symbol no target carries.

        This is the wiki-ahead-of-code shape, reaching the step through
        the claim gate rather than the structure one.
        """
        _page(
            tmp_path,
            "claims.md",
            "---\ntitle: x\ntags: [a]\nconfidence: high\n---\n\n"
            "# Page\n\n```json claims\n"
            '{"claims": [{"id": "x", "code": "tankpit_bot.protocol.commands:NOT_A_SYMBOL",'
            ' "value": 1}]}\n```\n',
        )

        code = run(tmp_path)
        out = capsys.readouterr().out

        assert code == 1
        assert "claim-binding violation(s)" in out

    def test_both_gates_report_rather_than_the_step_stopping_at_the_first(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A page can fail structure AND binding, and a reader should
        learn both in one run instead of coming back for a second."""
        _page(tmp_path, "broken.md", "# No frontmatter at all\n")
        _page(
            tmp_path,
            "claims.md",
            "---\ntitle: x\ntags: [a]\nconfidence: high\n---\n\n"
            "# Page\n\n```json claims\n"
            '{"claims": [{"id": "x", "code": "tankpit_bot.protocol.commands:NOT_A_SYMBOL",'
            ' "value": 1}]}\n```\n',
        )

        code = run(tmp_path)
        out = capsys.readouterr().out

        assert code == 1
        assert "structure violation(s)" in out
        assert "claim-binding violation(s)" in out


class TestTheStepAccepts:
    def test_the_real_package_is_clean(self) -> None:
        """The step's own subject, which is what the lint run checks."""
        assert run() == 0

    def test_the_default_root_is_this_package(self) -> None:
        """A default pointing anywhere else would lint a tree nobody
        edits and pass forever."""
        assert (PROJECT_ROOT / "wiki" / "pages").is_dir()
        assert (PROJECT_ROOT / "pyproject.toml").is_file()


class TestTheEntryPoint:
    def test_main_exits_zero_when_the_package_is_clean(self) -> None:
        """`poetry run tankpit-check-wiki` is the form the Makefile uses,
        so the exit code is the contract."""
        with pytest.raises(SystemExit) as exit_info:
            main()

        assert exit_info.value.code == 0

    def test_running_the_module_directly_reaches_main(self) -> None:
        """`python -m scripts.wiki_gates` is a supported invocation, so
        the __main__ block is exercised rather than left uncovered."""
        sys.modules.pop("scripts.wiki_gates", None)

        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module("scripts.wiki_gates", run_name="__main__")

        assert exit_info.value.code == 0


__all__ = ["TestTheEntryPoint", "TestTheStepAccepts", "TestTheStepRefuses"]
