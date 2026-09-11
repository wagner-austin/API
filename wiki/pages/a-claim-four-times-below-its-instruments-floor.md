---
title: A claim four times below the floor of the instrument that produced it
tags: [model-trainer, cartridges, statistical-power, experiment-design, retraction, research-registration]
related:
  - "[[model-trainer-trait-composition-instrument]]"
  - "[[code-style-qlora-terminates-worse-conforms-better]]"
source_paths:
  - docs/RESEARCH.md
  - tools/hpc3/runs/qa-full-wiki-gpt2-v52.json
  - tools/hpc3/runs/ledger.jsonl
source_git_blobs:
  "docs/RESEARCH.md": 575670696f235494c5f7c5a9d61f3dcedd208e77
  "tools/hpc3/runs/qa-full-wiki-gpt2-v52.json": fa3fa9a9a37c9530a9b5e7648c0ba329f1feb913
provenance:
  - "`tools/hpc3/runs/ledger.jsonl` is in source_paths but NOT in source_git_blobs, and that is deliberate rather than an omission the all-or-nothing rule missed: `git ls-files --error-unmatch` reports it untracked, so it is an untracked artifact and exempt by nature. Pinning it would also be the wrong instrument even if it were tracked -- it is append-only and grows on every cluster submission, so its hash would go red within hours for reasons unrelated to any claim on this page. Read against API HEAD 15562dbb."
  - "THE 24-ITEM RUN IS NOT A SOURCE_PATH AND CANNOT BE ONE. It is recorded only in the closure of agent-board task 1fc5afed by session opus-weight-injection-0902, which is a self-report on an append-only coordination surface with no artifact behind it and no mechanism for marking itself superseded. It is cited on this page as a second under-powered reading, never as a result."
  - "This page was first written in the `personal` wiki on 2026-09-11 and moved here the same day. The move is the point: `personal` is a pdf-corvis wiki, which pins PDFs, and every factual claim here is about files in this repository. Measured before moving -- 0 of the personal wiki's pages pin a repo file, so its citations of docs/RESEARCH.md were paths in a footnote with nothing detecting a change."
fact_checked: "2026-09-11"
confidence: high
hubs: [services]
---

# A claim four times below the floor of the instrument that produced it

`cartridge_qa_benchmark` reported that a trained KV-cache cartridge beats
lexical, dense and fused retrieval from about 774M parameters. The claim was
withdrawn on 2026-09-09, not because a later run disagreed with it but because
arithmetic on the question set shows **no outcome of that run could ever have
supported it**.[^1]

## The arithmetic, which needed no new measurement

The differences were 0.0521 and 0.0417 in accuracy over a **32-item** question
set. That is 1.7 and 1.3 items.[^1]

The comparison is McNemar's test at alpha 0.05 with the mid-p correction, and
that test's verdict depends only on the pairs where the two arms disagree. The
fewest discordant pairs that can ever reject the null is **five**, so the claim
sat roughly four times below the smallest effect its instrument could
resolve.[^1] A difference of 1.3 items cannot produce five discordant pairs in
the required direction.

**No split of that question set could have supported it.**[^1] That is what
makes this a retraction rather than a disagreement: there was no unlucky sample
and no analysis error, and the measurement was built such that the finding it
reported was unreachable.

## What survives, stated separately because it is a different comparison

At the same rungs, cartridge against the **un-augmented base** moved 8.7 and
8.3 items — above the five-pair floor and above the arm's own seed spread.[^1]

> *Cartridges improve the model over the un-augmented base from 774M* stands.
> *Cartridges beat retrieval* does not, and is withdrawn rather than softened.[^1]

Two claims from one run, one supportable and one not, is the ordinary case.
Retracting the whole run would have discarded a real result; softening the
unsupportable half to "suggests" would have kept an unreachable claim in
circulation wearing a hedge.

## The same question was also answered in the opposite direction

A separate run over 12 pages of the me-wiki corpus, scoring **24** held-out
items, reported the reverse: the cartridge losing to BM25, 0.6389 against
0.8333.[^4] That run's own closure records its cartridge gain as falling inside
its seed spread, so it was candid about half of its result — but 24 items is
further below the five-pair floor than the 32-item set that produced the
retracted claim.

Both runs used me-wiki plans, and **the me-wiki plans as a class are now refused
before they execute**.[^1] The useful observation is not that one of the two
directions was right. It is that an instrument too weak to resolve a difference
will still emit one, with a sign, and the sign is determined by noise. Two
sessions read opposite conclusions off the same apparatus.

## The repair is a refusal, not a better analysis

A plan declares `smallest_effect_of_interest`, `alpha` and `mcnemar_test`, and
`cartridge_qa_power.require_resolvable_question_set` refuses the run **before
the model loads** when the realised question set cannot resolve what the plan
declares.[^1] Three properties carry it:

- **It reads the REALISED count, never the configured cap.** The 32-item set
  came from a plan whose cap said 120. A gate reading the cap would have passed
  it.[^1]
- **It is a falsifiability gate, not a power one.** Passing means some
  attainable outcome supports the declared effect, never that the outcome is
  likely. Classifying an observed result against the discordant count that
  actually occurred is a separate statement.[^1]
- **It refuses rather than warns.** The four me-wiki plans are among the refused
  set, deliberately, since they are the plans that produced the retracted
  claim.[^1]

## Why it went unnoticed for as long as it did

The command carried the programme's headline result while appearing in **0 of
571 committed run documents, 0 lines of the research index, and 0 tracked
artifacts**.[^1] Its records were written to a temporary directory that is
purged, and it ran on a local workstation where nothing requires an image digest
or a staged corpus.

Nothing checked. The `research-registration` guard rule now fails `make lint` on
any entry point that builds a `RunRecord` and is named nowhere in the index.[^1]

## The re-run, and what is not yet known

A replacement measurement is registered: **3,740 items** over the full 841-page
corpus, a resolvable floor of **0.00134** against a declared effect of **0.02**,
across base, cartridge, oracle, BM25, dense, fused, expanded, reranked and
long-context arms, three seeds, on one A100.[^2] Its own `why` field reads
`the-retracted-headline-re-run-on-an-instrument-that-can-resolve-it`, and its
`supersedes` field names the 32-item runs.

**Its results are not in the committed record.** Four jobs were submitted on
2026-09-10 and their artifacts are on cluster storage under `/pub/wagnera3`, so
whether the properly-powered instrument reproduces either direction is, as of
this page's date, unknown here.[^3] The question is open, and that is a
different state from either of the answers previously given for it.

## Scope

- The retraction and the surviving claim are read from the research index's own
  entry, which cites the commits that performed them. The underlying per-item
  outcomes were not re-derived for this page.
- The five-pair floor is a property of McNemar's test at alpha 0.05 with mid-p,
  not a measurement of this repository.
- Nothing here evaluates whether cartridges beat retrieval. It is about what the
  runs to date could and could not establish.

[^1]: `docs/RESEARCH.md`, the `cartridge_qa_benchmark` entry. Records the
    retraction (commits `a98769b5`, `dc5f2408`, dated 2026-09-09), the 0.0521
    and 0.0417 differences over 32 items, the five-discordant-pair floor, the
    surviving 8.7 and 8.3 item gains against base, the
    `require_resolvable_question_set` refusal and its realised-count rule, the
    falsifiability framing, the refusal of the four me-wiki plans, the
    0-of-571 registration gap, and the `research-registration` guard rule
    (`9b011256`, corrected in `bd4dfb97`).
[^2]: `tools/hpc3/runs/qa-full-wiki-gpt2-v52.json` — image
    `f148e0412b16595c53e9df657f22b1a898ff0b744af371e2bb725489f91b87ef`, plan
    `gpt2-full-wiki-qa`, corpus `me-wiki-full-841-pages` digest `dccd3375f54d`,
    3,740 items, resolvable floor 0.00134, declared effect 0.02, seeds 7/8/9,
    A100, and the `why` / `supersedes` fields quoted above.
[^3]: `tools/hpc3/runs/ledger.jsonl` — four rows naming
    `mi.qa-full-wiki-gpt2-v50` (twice), `-v51` and `-v52`, jobs 55898417,
    55898551, 55901956 and 55914185, submitted 2026-09-10, each writing to
    `/pub/wagnera3/mi/cartridge/results/`. No terminal state, exit code or
    result is recorded in those rows. Untracked and therefore unpinned; see
    `provenance:`.
[^4]: Agent-board task `1fc5afed`, closure by session
    `opus-weight-injection-0902`: 24 held-out items over the 12 public me-wiki
    pages, gpt2, three seeds, RTX 3090 Ti — base 0.5417, base+cartridge 0.6389
    with the gain inside its own 0.125 seed spread, base+BM25 0.8333,
    base+oracle 1.0000. A board closure is a session's own report carrying no
    audited artifact; see `provenance:` for why it is not a source path.
