from __future__ import annotations

from pathlib import Path

from model_trainer.core.services.data.corpus import iter_lines


def test_iter_lines_skips_blank_and_whitespace_only(tmp_path: Path) -> None:
    fp = tmp_path / "mix.txt"
    fp.write_text("\n   \nfoo\n  \t\nbar\n", encoding="utf-8")
    out = list(iter_lines([str(fp)]))
    assert out == ["foo", "bar"]
