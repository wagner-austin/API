"""Split a corpus into what a cartridge trains on and what it is asked about.

Separated from the benchmark CLI by ROLE rather than by size. The CLI runs
arms and records numbers; this decides what the question set IS, which is
the decision every arm's score depends on and the one that has twice been
wrong in a way no arm could reveal.

BOTH OF THOSE FAILURES ARE WORTH CARRYING, because neither raised anything:

* The held-out stride counted across the CORPUS rather than within a page,
  so three of twelve real pages were trained on and never examined. Every
  arm was fitted to twelve pages and scored on nine. A short question set is
  indistinguishable from a short corpus, so nothing surfaced it until
  someone asked why a plan permitting 120 items returned 24.
* A term qualified an item from raw training text while the evidence arms
  can only cite SENTENCES, which strip fences, table rows and URLs. That
  admitted items no retrieval arm could answer -- biasing toward the
  cartridge -- and is fixed in `corpus_cloze.build_items` rather than here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from model_trainer.core.contracts.cloze import ClozeItem
from model_trainer.core.contracts.qa_plan import QaPlan
from model_trainer.core.encoding import Encoder
from model_trainer.core.services.model.cartridge_corpus import window_documents
from model_trainer.core.services.model.corpus_cloze import build_items


def build_question_set(
    documents: Sequence[str],
    encoded: Sequence[Sequence[int]],
    encoder: Encoder,
    plan: QaPlan,
) -> tuple[list[ClozeItem], str]:
    """Split a corpus and build items from the half the cartridge will not read.

    The split is by window, not by document, and that is what makes the set
    answerable. Pages here are about different projects, so a document-level
    split would leave held-out terms that appear nowhere in the training text
    and every item would be unanswerable from the corpus. Splitting within
    each page keeps a term learnable from the windows the cartridge trains on
    while testing it in a sentence those windows do not contain.

    The window text is recovered by DECODING the ids rather than by slicing
    the document string, because the split is defined on tokens and a
    character offset cannot name a token boundary.

    Args:
        documents: Document bodies, in the order they were encoded.
        encoded: Token ids for each document.
        encoder: Tokenizer, used to read each window's text back.
        plan: The measurement being run.

    Returns:
        ``(items, training_text)``. A plain pair rather than a named record:
        the two have different types, so nothing can transpose them, and a
        class holding them would carry no behaviour of its own.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` when the corpus cannot
            supply windows, a split, or items.
    """
    owners = window_documents(encoded, window=plan["window"])
    stride = plan["held_out_stride"]
    windows_per_document = Counter(owners)
    held_by_document: dict[int, list[str]] = {}
    training: list[str] = []
    seen_per_document: dict[int, int] = {}
    for owner in owners:
        start = seen_per_document.get(owner, 0)
        seen_per_document[owner] = start + 1
        window = plan["window"]
        text = encoder.decode(list(encoded[owner])[start * window : (start + 1) * window])
        # THE STRIDE COUNTS WITHIN A DOCUMENT, NOT ACROSS THE CORPUS, and the
        # first version of this counted across. That made which pages get
        # tested a function of where their windows happened to land in the
        # global sequence: measured on the twelve public me-wiki pages,
        # THREE OF TWELVE held out nothing at all. Those pages were trained
        # on and never examined, so the cartridge was fitted to twelve pages
        # and scored on nine -- silently, because a short question set looks
        # exactly like a short corpus.
        #
        # A SINGLE-WINDOW DOCUMENT IS TRAINED ON AND NOT TESTED. It cannot be
        # both: holding out its only window would leave its terms absent from
        # the training text, and `build_items` would then correctly refuse
        # every item drawn from it as unanswerable. Training is the useful
        # half -- its terms stay learnable and can serve as other pages'
        # distractors -- so that is the side it goes to.
        if windows_per_document[owner] > 1 and start % stride == 0:
            held_by_document.setdefault(owner, []).append(text)
        else:
            training.append(text)
    held_documents = [
        " ".join(held_by_document.get(document, [])) for document in range(len(documents))
    ]
    training_text = " ".join(training)
    return (
        build_items(
            held_documents,
            training_text,
            distractor_count=plan["distractor_count"],
            max_items=plan["max_items"],
        ),
        training_text,
    )


__all__ = ["build_question_set"]
