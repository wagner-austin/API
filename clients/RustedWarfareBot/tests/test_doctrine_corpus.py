"""Every shipped doctrine file parses, so the corpus cannot rot silently.

The schema requires every knob in every file (absence is a
:class:`~rw_bot.validation.DecodeError` by design -- no defaults, no
fallbacks), which means adding a knob is a mass edit of the whole corpus.
Until 2026-09-12 nothing enforced that the edit was complete: a file the
sweeps did not currently reference could sit unloadable for weeks and fail
the first experiment that reached for it, at match time, on a cluster node.
This suite is the missing layer -- the same move that pinned the shipped
default to its constant, widened to every preset the repository carries,
one named test per file so a refusal points at the exact preset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.policy.doctrine_file import parse_doctrine_lines

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CORPUS = sorted((_PROJECT_ROOT / "doctrines").glob("*.doctrine"))


def test_the_corpus_is_present() -> None:
    """An empty glob would pass a parametrized suite by running nothing."""
    assert len(_CORPUS) > 100


@pytest.mark.parametrize("path", _CORPUS, ids=[path.name for path in _CORPUS])
def test_every_shipped_doctrine_parses(path: Path) -> None:
    """The preset decodes under the current schema and names itself."""
    doctrine = parse_doctrine_lines(path.read_text(encoding="utf-8").splitlines())
    assert doctrine["name"], f"{path.name} decodes to a nameless doctrine"
