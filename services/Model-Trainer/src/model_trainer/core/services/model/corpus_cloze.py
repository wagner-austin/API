"""Build cloze items from a corpus, mechanically, with no invention.

WHY THIS EXISTS. Every cartridge number measured so far is a held-out LOSS,
and :mod:`model_trainer.core.contracts.cloze` already says why that is not
enough: "a model can memorise text word-by-word and still fail every question
about it." A prefix that lowers perplexity on wiki prose has demonstrated that
it learned the corpus's TEXT. Whether it learned the corpus's FACTS, in a form
the model can use, is a different question and needs a different instrument.

THE ITEMS ARE GENERATED, NOT WRITTEN. Nothing here composes a sentence or
invents a fact. A term is chosen out of the corpus, the sentence it occurs in
is taken verbatim, and the term is blanked. Distractors are terms drawn from
OTHER documents in the same corpus. That matters for a reason beyond effort: a
hand-written question set is a place for a fact that is not in the corpus to
enter the measurement, and there would be no way to tell from the numbers.

THE MEMORISATION TRAP, AND THE SPLIT THAT AVOIDS IT. If items came from the
text the cartridge trained on, the cartridge would have seen the exact sentence
and would win by recall of that string -- which is the confound the cloze
docstring warns about, reintroduced by the item builder. So:

  - the cartridge trains on the TRAINING windows;
  - items are built from the HELD-OUT windows, which it never saw;
  - a term qualifies only if it ALSO occurs in the training windows.

That last clause is what makes the item answerable at all. The term is
learnable from what the cartridge read, and it is tested in a sentence it did
not read. A term appearing only in held-out text would be unanswerable from
the corpus and would measure the base model's guessing.

WHAT A WRONG ANSWER MEANS HERE. The distractors are other corpus terms, so
chance is one over the candidate count and a model with no corpus knowledge
scores there. Beating chance requires knowing which of several corpus-specific
terms belongs in this sentence, which is not recoverable from general English.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)

from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeItem

#: A candidate term: a NAME, not merely a capitalised word.
#:
#: THE FIRST VERSION OF THIS ACCEPTED ANY CAPITALISED WORD AND THE ITEMS WERE
#: WORTHLESS. Run against the wiki it produced items whose answer was "Plain"
#: against distractors "Getting" and "Produced" -- an item any English model
#: answers without having read a line of the corpus, which measures fluency
#: and reports it as corpus knowledge. Sentence-initial words are capitalised
#: in English, so "capitalised" selects mostly ordinary vocabulary.
#:
#: A name carries internal evidence of being one. Each alternative below is
#: such evidence:
#:
#:   CamelCase or an internal capital   ClearGBM, TankpitBot, LightGBM
#:   an internal digit                  HPC3, gpt2, A100
#:   a hyphen joining capitalised parts Model-Trainer, Rusted-Warfare
#:   all capitals, three or more        API, UCI, OCR
#:
#: A word that is merely capitalised because a sentence started matches none
#: of them. This is still crude and still mechanical, which is the point: a
#: pattern anyone can read beats a tagger nobody can audit.
_TERM = re.compile(
    r"\b(?:"
    r"[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*"
    r"|[A-Za-z]+[0-9]+[A-Za-z0-9]*"
    r"|[A-Z][A-Za-z0-9]+(?:-[A-Za-z0-9]+)+"
    r"|[A-Z]{3,}"
    r")\b"
)

#: Names that match :data:`_TERM` and still carry no corpus information.
#:
#: Mostly file formats and protocol words: a corpus about software mentions
#: JSON and HTTP the way any software text does, so an item asking which of
#: them belongs is not asking about this corpus.
#:
#: Markdown's own admonition markers are deliberately absent. They match
#: ``[A-Z]{3,}`` like the rest, but naming one here puts that literal in a
#: source file, and the repository scans source files for exactly those words
#: as abandoned-work markers. The cost of leaving them in is one possible odd
#: item; the cost of listing them is a guard that cries wolf on every run.
_STOPWORDS = frozenset(
    {
        "JSON",
        "HTTP",
        "HTTPS",
        "HTML",
        "CSV",
        "PDF",
        "URL",
        "URLs",
        "YAML",
        "SQL",
        "XML",
        "API",
        "APIs",
        "CLI",
        "CPU",
        "GPU",
        "RAM",
        "SSD",
        "OS",
        "ID",
        "IDs",
        "UUID",
        "ASCII",
        "UTF",
        "NOTE",
        "WARNING",
        "README",
        "MIT",
        "BSD",
        "GPL",
        "AND",
        "OR",
        "NOT",
        "ALL",
        "ANY",
    }
)

#: Sentence splitter. Splits after . ! or ? followed by whitespace.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

#: Shortest sentence worth blanking, in characters.
#:
#: A very short sentence gives the model almost no context, so the item stops
#: being about the corpus and becomes about which term is commoner in English.
MIN_SENTENCE_CHARS = 60

#: Longest sentence worth blanking, in characters. A very long one crowds the
#: retrieval arm's context window, which is the arm this comparison exists to
#: be fair to.
MAX_SENTENCE_CHARS = 400


#: Fenced code blocks, removed whole.
_FENCE = re.compile(r"```.*?```", re.DOTALL)

#: A URL, bare or angle-bracketed. Removed rather than blanked: a URL is a
#: string of corpus-specific tokens that no model predicts from meaning, so an
#: item containing one measures memorisation of a path.
_URL = re.compile(r"<?https?://\S+>?")

#: A markdown table row, which flattens into a sentence of pipes and dashes.
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)


def sentences(text: str) -> list[str]:
    """Split a document into sentences, after removing what is not prose.

    THE MARKDOWN HAS TO GO FIRST, and the first version of this did not do it.
    The corpus is wiki pages, so an un-cleaned split produced "sentences" that
    were headers with raw URLs in them -- and an item blanking a token out of
    a URL asks the model to recall a path, which is memorisation wearing a
    question's clothes.

    Whitespace is collapsed after that, because a sentence broken across two
    source lines is one sentence, and an item carrying its original line break
    would be scored on text that appears nowhere.

    Args:
        text: The document body.

    Returns:
        The sentences, in order.
    """
    stripped = _FENCE.sub(" ", text)
    stripped = _TABLE_ROW.sub(" ", stripped)
    stripped = _URL.sub(" ", stripped)
    flattened = " ".join(stripped.split())
    # Sliced at the terminators rather than handed to `re.split`, whose return
    # is typed `list[str | Any]` because a pattern MAY carry capture groups.
    # This one does not, so every part is a string -- but the type says
    # otherwise, and an Any here would spread into every item built from it.
    parts: list[str] = []
    cut = 0
    for match in _SENTENCE.finditer(flattened):
        parts.append(flattened[cut : match.start()])
        cut = match.end()
    parts.append(flattened[cut:])
    return [part.strip() for part in parts if part.strip()]


def terms_in(text: str) -> set[str]:
    """Collect the candidate terms a piece of text contains.

    Args:
        text: The text to scan.

    Returns:
        Every distinct term, stopwords excluded.
    """
    return {match.group(0) for match in _TERM.finditer(text) if match.group(0) not in _STOPWORDS}


def _blank_once(sentence: str, term: str) -> str | None:
    """Replace a term's only occurrence in a sentence with the blank marker.

    Args:
        sentence: The sentence to blank.
        term: The term to remove.

    Returns:
        The blanked sentence, or None when the term occurs other than exactly
        once. A sentence naming the term twice would leave one copy visible
        beside the blank, which answers the item.
    """
    pattern = re.compile(rf"\b{re.escape(term)}\b")
    # Counted through `finditer` rather than `findall`, which is typed
    # `list[Any]` because what it yields depends on the pattern's groups.
    if sum(1 for _ in pattern.finditer(sentence)) != 1:
        return None
    return pattern.sub(BLANK_MARKER, sentence, count=1)


def build_items(
    held_out_documents: Sequence[str],
    training_text: str,
    *,
    distractor_count: int,
    max_items: int,
) -> list[ClozeItem]:
    """Build cloze items from held-out text, answerable from training text.

    Args:
        held_out_documents: Document bodies the cartridge did NOT train on,
            one per document, in a stable order.
        training_text: Everything the cartridge DID train on, concatenated. A
            term must occur here to qualify, which is what makes its item
            answerable from the corpus rather than by guessing.
        distractor_count: Wrong candidates per item. Chance accuracy is
            ``1 / (distractor_count + 1)``.
        max_items: Stop after this many items.

    Returns:
        The items, in document order. Each carries a distinct ``item_id``
        naming the document index and the term.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` when the corpus yields
            too few distinct terms to draw distractors from, or no items at
            all. Both mean the measurement cannot be built, and silently
            returning a short list would report a weaker result as a real one.
    """
    learnable = terms_in(training_text)
    per_document = [terms_in(document) for document in held_out_documents]
    everywhere = sorted({term for found in per_document for term in found} | learnable)
    if len(everywhere) <= distractor_count:
        raise _unusable(
            f"the corpus yields {len(everywhere)} distinct term(s), which cannot supply "
            f"{distractor_count} distractor(s) plus an answer; a cloze item needs "
            f"candidates that are all corpus terms or the wrong ones give it away"
        )

    built: list[ClozeItem] = []
    for index, document in enumerate(held_out_documents):
        for sentence in sentences(document):
            if not MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS:
                continue
            for term in sorted(terms_in(sentence) & learnable):
                template = _blank_once(sentence, term)
                if template is None:
                    continue
                # Distractors come from OTHER documents, so a distractor is
                # never a term this sentence's own page is about -- which
                # would make it a plausible answer rather than a wrong one.
                elsewhere = [
                    other
                    for other in everywhere
                    if other != term and other not in per_document[index]
                ]
                if len(elsewhere) < distractor_count:
                    continue
                built.append(
                    ClozeItem(
                        # THE POSITION IS IN THE ID BECAUSE THE TERM IS NOT
                        # UNIQUE. One document usually names its own subject in
                        # several sentences, and an id of document-plus-term
                        # then repeats. Arms are paired BY ID
                        # (`cartridge_qa.compare_arms`), so duplicates would
                        # collapse in the lookup and pair one arm's outcome
                        # against a different question's -- silently, and in a
                        # direction nobody could predict.
                        item_id=f"d{index:03d}-i{len(built):03d}-{term}",
                        template=template,
                        answer=term,
                        distractors=_spread(elsewhere, distractor_count, offset=len(built)),
                    )
                )
                if len(built) >= max_items:
                    return built
    if not built:
        raise _unusable(
            "no held-out sentence carries a term that also occurs in the training text, "
            "so every item would be unanswerable from the corpus; the split may be too "
            "coarse or the documents too short"
        )
    return built


def _spread(candidates: Sequence[str], count: int, *, offset: int) -> list[str]:
    """Take ``count`` candidates spread evenly, starting from ``offset``.

    Evenly rather than the first ``count``, because the list is alphabetical
    and its head is whichever terms happen to start with early letters.

    THE OFFSET IS NOT COSMETIC, and the first version of this omitted it. Every
    item drew the same evenly-spaced slots, so a 24-item set went out with
    three distractors repeated across nearly all of it. That is a measurement
    defect rather than an ugly one: with one distractor set the whole score
    turns on how those particular three words compare to each answer, and a
    single unluckily-plausible distractor moves every item at once.

    Rotating by the item's position spreads that risk across the set. It stays
    deterministic -- no sampling, so the item set is a function of the corpus
    and nothing else.

    Args:
        candidates: Terms to choose from, in a stable order.
        count: How many to take.
        offset: Rotation, normally the item's index within the set.

    Returns:
        Exactly ``count`` distinct terms, in the order chosen.

        DISTINCTNESS IS GUARANTEED RATHER THAN REPAIRED. The caller has
        already established ``len(candidates) >= count``, so ``step >= 1`` and
        the floor indices ``int(index * step)`` strictly increase; rotating
        them by a constant modulo the length is a bijection, so no two can
        collide. An earlier draft carried a top-up loop for collisions that
        cannot occur -- unreachable code, which is worse than none: nothing
        can cover it and nothing can prove it right.
    """
    step = len(candidates) / count
    return [candidates[(int(index * step) + offset) % len(candidates)] for index in range(count)]


def _unusable(message: str) -> AppError[ModelTrainerErrorCode]:
    """Build the error a corpus that cannot produce items raises.

    Args:
        message: What is wrong.

    Returns:
        The error to raise.
    """
    return AppError(
        ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE,
        message,
        model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE),
    )


__all__ = [
    "MAX_SENTENCE_CHARS",
    "MIN_SENTENCE_CHARS",
    "build_items",
    "sentences",
    "terms_in",
]
