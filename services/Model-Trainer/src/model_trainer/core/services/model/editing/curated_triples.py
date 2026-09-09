"""The hand-curated triples for the twelve public me-wiki pages.

A PYTHON LITERAL RATHER THAN A DATA FILE, and the reason is the bar this arm
is testing. These are curated by hand at a stated precision, they are the
arm's most expensive input, and mypy checks their shape at authoring time --
which is strictly stronger than a decoder checking it at run time. A JSON file
would add a parse boundary, a codec and a validation step to twenty rows that
change only when a human rewrites them.

WHAT ONE ROW IS. One attempt to express, as a subject-relation-object triple,
the association that one held-out question asks about. Attempts are recorded
for every distinct answer in the question set INCLUDING the ones that cannot
be grounded, because the reject rate is the deliverable and a curator who
silently drops the hard cases reports a bar that nothing failed.

HOW THEY WERE MADE. For each distinct answer in the 32-item gpt2 question set,
every training sentence containing that answer was listed, together with the
corpus terms occurring strictly before it in that sentence. Where such a term
existed, it became the subject and the relation was written by hand to join it
to the answer. Where none existed, the nearest attempt was recorded anyway and
the gate rejects it -- see :data:`UNGROUNDABLE_ANSWERS`.

THE COST LINE, which criterion 2 of board task 3fc98ed6 asks for, is not in
this file: it is what :func:`~.grounding.gate_candidates` counts over these
rows, plus the wall-clock of the curation pass, and it belongs in the record
the arm emits rather than in a comment that cannot be re-measured.
"""

from __future__ import annotations

from typing import Final

from model_trainer.core.services.model.editing.grounding import TripleCandidate

_EN_DASH: Final[str] = chr(0x2013)
"""U+2013, spelled by codepoint rather than written literally.

One curated sentence carries an en dash, and the row is judged on matching
that sentence exactly, so the character cannot be replaced with a hyphen. A
literal one is an ambiguous-character lint (`RUF001`), which this repository
does not suppress. Naming the codepoint answers the lint's actual question --
which character was meant -- instead of silencing it.
"""

CORPUS_DIGEST: Final[str] = "e2f23c63558342a2ad195705933aaf0c282bca7e104b30210802eb68c0106295"
"""Which corpus these were curated against.

Pinned because a triple's grounding is a claim about one exact text. A run
whose corpus digest differs from this is curating against something else, and
the arm refuses rather than reporting a reject rate for the wrong reason.
"""

UNGROUNDABLE_ANSWERS: Final[tuple[str, ...]] = (
    "FastAPI",
    "UCI",
    "Microsoft-maintained",
    "JVM",
    "AI",
    "XOR",
    "SCiL",
)
"""Answers for which NO training sentence carries a corpus term before them.

Measured, not judged: seven of the twenty distinct answers in the question
set. Every one of them appears in the training text -- that is why the item
builder allowed it -- but always as the first entity in its sentence, so
there is nothing for a locate-then-edit prompt to be about. This is criterion
4's first failure class, and it is a property of the PROSE rather than of the
curator: an English sentence that introduces its subject at the front leaves
no earlier entity to hang an association on.

The rows for these answers are still present below, carrying the nearest
attempt, so the gate rejects them on the record.
"""

ME_WIKI_PUBLIC_TRIPLES: Final[tuple[TripleCandidate, ...]] = (
    TripleCandidate(
        item_id="t00-FastAPI",
        subject="RQ",
        relation="completes the queue stack whose web framework is",
        object="FastAPI",
        source_document="api-monorepo.md",
        source_sentence="FastAPI + Redis + RQ.",
    ),
    TripleCandidate(
        item_id="t01-RQ",
        subject="FastAPI",
        relation="+ Redis +",
        object="RQ",
        source_document="api-monorepo.md",
        source_sentence="FastAPI + Redis + RQ.",
    ),
    TripleCandidate(
        item_id="t02-ClearGBM",
        subject="XGBoost",
        relation="sits in the same backend list as the from-scratch booster",
        object="ClearGBM",
        source_document="covenant-radar.md",
        source_sentence=(
            "## Twelve model backends behind one interface Seven classifiers — XGBoost, "
            "LightGBM, ClearGBM, logistic regression, random forest, a PyTorch MLP, and a "
            "bidirectional LSTM for temporal bankruptcy sequences — plus **five** regressors, "
            "swappable per request.[^1] ClearGBM being on that list matters: it is the "
            "from-scratch implementation described in [[api-monorepo]], competing against the "
            "production libraries in the same harness."
        ),
    ),
    TripleCandidate(
        item_id="t03-hpc3",
        subject="PROJECTS",
        relation="API tools directory holds the cluster CLI called",
        object="hpc3",
        source_document="hpc3-cli.md",
        source_sentence=(
            "**10,422 lines of source against 10,828 lines of tests.**[^1] [^1]: "
            "`~/PROJECTS/API/tools/hpc3`, counted 2026- #7131]( | Fixed "
            "`test_numeric_split_direction` so it actually tested all parameter combinations | "
            "## Two more open, and one closed without merging Verified against the GitHub API "
            "2026-08-26: **12 PRs authored, 9 merged, 2 open, 1 closed unmerged.** The merged "
            "nine are the table above and that number is unchanged ."
        ),
    ),
    TripleCandidate(
        item_id="t04-UCI",
        subject="Kaggle",
        relation="credit corpora sit beside German Credit from",
        object="UCI",
        source_document="covenant-radar.md",
        source_sentence=(
            "- **40 real public datasets** vendored for training and evaluation — Kaggle "
            "credit/loan/default corpora, UCI German Credit, and Taiwanese, Polish, Chinese "
            "and US bankruptcy datasets.[^2] Not synthetic data, which is the usual shortcut "
            "here."
        ),
    ),
    TripleCandidate(
        item_id="t05-LightGBM",
        subject="XGBoost",
        relation="and the other production booster behind that interface,",
        object="LightGBM",
        source_document="cleargbm.md",
        source_sentence=(
            "## It is not a toy — it competes in its own harness ClearGBM is one of the eleven "
            "model backends in [[covenant-radar]], sitting behind the same interface as "
            "XGBoost and LightGBM and selectable per request."
        ),
    ),
    TripleCandidate(
        item_id="t06-Microsoft-maintained",
        subject="Merged",
        relation=f"Jan{_EN_DASH}Mar 2026 while the project was",
        object="Microsoft-maintained",
        source_document="lightgbm-contributions.md",
        source_sentence=(
            f"Merged Jan{_EN_DASH}Mar 2026 while the project was Microsoft-maintained; "
            "it has since moved to its own org and the old `microsoft/LightGBM` links "
            "redirect."
        ),
    ),
    TripleCandidate(
        item_id="t07-PR",
        subject="LightGBM",
        relation=", 9 merged",
        object="PR",
        source_document="links.md",
        source_sentence=(
            "## Code ## Open source - **LightGBM, 9 merged PRs** — Individual pull requests "
            "are listed in [[lightgbm-contributions]]."
        ),
    ),
    TripleCandidate(
        item_id="t08-GitHub",
        subject="hpc3",
        relation="pull-request counts were verified against the API of",
        object="GitHub",
        source_document="hpc3-cli.md",
        source_sentence=(
            "**10,422 lines of source against 10,828 lines of tests.**[^1] [^1]: "
            "`~/PROJECTS/API/tools/hpc3`, counted 2026- #7131]( | Fixed "
            "`test_numeric_split_direction` so it actually tested all parameter combinations | "
            "## Two more open, and one closed without merging Verified against the GitHub API "
            "2026-08-26: **12 PRs authored, 9 merged, 2 open, 1 closed unmerged.** The merged "
            "nine are the table above and that number is unchanged ."
        ),
    ),
    TripleCandidate(
        item_id="t09-RustedWarfareBot",
        subject="TankpitBot",
        relation="is one of the two game clients, and the other is",
        object="RustedWarfareBot",
        source_document="api-monorepo.md",
        source_sentence=(
            "## The two game clients live here too `clients/TankpitBot` and "
            "`clients/RustedWarfareBot` are in this repository but are not services, and they "
            "are the strongest work in it."
        ),
    ),
    TripleCandidate(
        item_id="t10-TankpitBot",
        subject="PROJECTS",
        relation="API clients directory holds the game client called",
        object="TankpitBot",
        source_document="tankpitbot.md",
        source_sentence=(
            "[^2]: `~/PROJECTS/API/clients/TankpitBot/wiki` — `ls -1 pages/*.md | wc -l` → 74, "
            "`ls -1 hubs/*.md | wc -l` → 6."
        ),
    ),
    TripleCandidate(
        item_id="t11-JVM",
        subject="agent-selftest",
        relation="patches the real jar and lets the bytecode verifier of the",
        object="JVM",
        source_document="rustedwarfarebot.md",
        source_sentence=(
            "**`make agent-selftest` patches the real jar and lets the JVM's own bytecode "
            "verifier check the result**, so a patcher regression or a moved obfuscated class "
            "fails at the build gate instead of inside a live engine."
        ),
    ),
    TripleCandidate(
        item_id="t12-AI",
        subject="Impossible",
        relation="is the hardest rung of the built-in",
        object="AI",
        source_document="rustedwarfarebot.md",
        source_sentence=(
            "## The rest - **The standing goal is a measured 100% win rate against the "
            "built-in AI at Impossible** and every rung below, with any champion match "
            "watchable live."
        ),
    ),
    TripleCandidate(
        item_id="t13-PROJECTS",
        subject="Re-derived",
        relation="from the local clone under the home directory named",
        object="PROJECTS",
        source_document="smaller-public-work.md",
        source_sentence=(
            "Re-derived from the local clone at `~/PROJECTS/swarm` on 2026-08-23; first "
            "counted 2026-08-20 and unchanged since."
        ),
    ),
    TripleCandidate(
        item_id="t14-XOR",
        subject="Katakana",
        relation="appears in the same page as the subtype byte scrambled by",
        object="XOR",
        source_document="smaller-public-work.md",
        source_sentence=(
            "## Kana Pop — a browser game teaching Hiragana, Katakana and Rom E container**, "
            "which identifies its payloads *structurally* — by length and byte pattern — "
            "precisely because the subtype byte is XOR-scrambled per session and cannot be "
            "trusted."
        ),
    ),
    TripleCandidate(
        item_id="t15-SCiL",
        subject="Proceedings",
        relation="of the 2026 conference abbreviated",
        object="SCiL",
        source_document="links.md",
        source_sentence=(
            '## Research - **The published paper** — "Quantifying mutual intelligibility '
            'gradients in Turkic languages using language models," Proceedings of SCiL 2026, '
            "published 2026-06-27: - **HSP 2026**, 39th Annual Conference on Human Sentence "
            "Processing, MIT — - **Tu+11**, 11th Workshop on Turkic and Languages in Contact "
            "with Turkic, MIT — All three are [[turkic-mutual-intelligibility]] — the first is "
            "the publication, the other two are conference acceptances."
        ),
    ),
    TripleCandidate(
        item_id="t16-HSP",
        subject="SCiL",
        relation="2026 is listed beside the sentence-processing conference",
        object="HSP",
        source_document="links.md",
        source_sentence=(
            '## Research - **The published paper** — "Quantifying mutual intelligibility '
            'gradients in Turkic languages using language models," Proceedings of SCiL 2026, '
            "published 2026-06-27: - **HSP 2026**, 39th Annual Conference on Human Sentence "
            "Processing, MIT — - **Tu+11**, 11th Workshop on Turkic and Languages in Contact "
            "with Turkic, MIT — All three are [[turkic-mutual-intelligibility]] — the first is "
            "the publication, the other two are conference acceptances."
        ),
    ),
    TripleCandidate(
        item_id="t17-CE",
        subject="Finnish-as-text",
        relation="excess",
        object="CE",
        source_document="turkic-mutual-intelligibility.md",
        source_sentence=(
            "Finnish-as-text excess CE by listener: tr 2.128, uz 2.199, kk 2.232, az 2.388, "
            "ug 2.479, ky 2.727 — every one above that listener's worst within-Turkic score."
        ),
    ),
    TripleCandidate(
        item_id="t18-LSTM",
        subject="PyTorch",
        relation="MLP sits beside a bidirectional",
        object="LSTM",
        source_document="covenant-radar.md",
        source_sentence=(
            "## Twelve model backends behind one interface Seven classifiers — XGBoost, "
            "LightGBM, ClearGBM, logistic regression, random forest, a PyTorch MLP, and a "
            "bidirectional LSTM for temporal bankruptcy sequences — plus **five** regressors, "
            "swappable per request.[^1] ClearGBM being on that list matters: it is the "
            "from-scratch implementation described in [[api-monorepo]], competing against the "
            "production libraries in the same harness."
        ),
    ),
    TripleCandidate(
        item_id="t19-IPA",
        subject="FastText",
        relation="language identification sits beside deterministic transliteration into",
        object="IPA",
        source_document="links.md",
        source_sentence=(
            "- **Turkic API** — the corpus-construction half of the Turkic research: OSCAR, "
            "Wikipedia and CulturaX streaming, FastText language identification across nine "
            "languages, and deterministic IPA transliteration for seven of them across "
            "Cyrillic, Latin and Arabic scripts."
        ),
    ),
)
"""One curation attempt per distinct answer in the gpt2 question set.

Twenty rows for twenty distinct answers across 32 items -- several items ask
about the same term in different sentences, and one association is what a
weight edit writes, so the rows are per answer rather than per item.
"""


__all__ = [
    "CORPUS_DIGEST",
    "ME_WIKI_PUBLIC_TRIPLES",
    "UNGROUNDABLE_ANSWERS",
]
