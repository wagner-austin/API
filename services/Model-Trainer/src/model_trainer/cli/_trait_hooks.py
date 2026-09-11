"""The filesystem seam the trait-composition sweep needs.

WHY A THIRD HOOKS MODULE, stated because two already exist and a third is the
kind of thing that becomes four without a reason. :mod:`_test_hooks` holds the
seams that need real weights, a real GPU or a real corpus on disk;
:mod:`_measurement_hooks` holds the TABLES. This one is the same role as the
first, for a different AXIS, and it is separate because adding it to
``_test_hooks`` pushed that file to 604 lines against this package's 600-line
ceiling -- the guard said so before the suite ran.

Splitting the trait seam out rather than moving the corpus reader was the
cheaper of the two honest options by a wide margin: ``read_corpus_documents``
has 69 references across 31 files, and moving it would have been a rename
touching every sweep in the package to fix a ceiling this axis pushed past.
The boundary drawn here is the one that already exists in the tree -- the
trait axis has its own contract, its own loader and its own plan table -- so
it has its own seam too, and it is where the axis's next seam goes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from model_trainer.core.contracts.trait_corpus import TraitCorpus
from model_trainer.core.services.model.trait_corpus import load_trait_corpora


class ReadTraitCorporaProto(Protocol):
    """Protocol for reading a roster's authored trait pairs off disk.

    Behind a hook for the reason the corpus reader next door is, and one more
    that is specific to this axis: the committed corpus is the artifact the
    whole measurement's power depends on, so a suite that read it would be
    asserting against numbers that move whenever a pair is added. Tests build
    the corpora they need; the default implementation has its own tests
    against real files in a temporary directory.
    """

    def __call__(self, corpus_dir: Path, traits: Sequence[str], /) -> tuple[TraitCorpus, ...]:
        """Read the named traits' pair sets, in roster order."""
        ...


def _default_read_trait_corpora(
    corpus_dir: Path, traits: Sequence[str], /
) -> tuple[TraitCorpus, ...]:
    """Production implementation - read a roster's trait files.

    Args:
        corpus_dir: Directory holding one JSON file per trait.
        traits: The plan's roster, in order.

    Returns:
        One validated corpus per trait, in roster order.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if the directory holds no
            trait files, a named trait has no file, or a file fails its own
            validation.
    """
    return tuple(load_trait_corpora(corpus_dir, traits))


read_trait_corpora: ReadTraitCorporaProto = _default_read_trait_corpora


__all__ = [
    "ReadTraitCorporaProto",
    "read_trait_corpora",
]
