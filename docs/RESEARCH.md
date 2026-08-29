# Research index

**Every body of work on this machine that produces numbers someone compares.**
Read this before auditing, extending, or reproducing any experiment.

This file exists because nothing like it did, and the cost was measured: on
2026-08-28 an audit of provenance across this machine examined four research
surfaces and missed two entirely — LSTM and RustedWarfareBot — because there
are roughly ninety directories under `~/PROJECTS` and no list. One of the
surfaces below is still not registered anywhere a tool can see, and one more
was scoped as an example and never onboarded.

The machine-readable half of this is the `projects` table in the hpc3
workspace documents (`tools/hpc3/runs/hpc3*.json`). Each entry declares how a
project runs on the cluster, its own caps and charge account, and — since
2026-08-28 — `repo`, where its code lives. Anything registered there is
enforced against this file by
`tools/hpc3/tests/test_committed_runs.py`; anything not registered appears
below with that stated, and nothing checks it.

---

## Registered with the hpc3 CLI

These submit through `hpc3-submit` / `hpc3-sweep` / `hpc3-chain`, and every
submission lands in `tools/hpc3/runs/ledger.jsonl` (machine-local, deliberately
untracked — it is state, not configuration).

### `mi` — Model-Trainer probes and benchmarks

- **Repo:** this one, `services/Model-Trainer`
- **Runs:** `model_trainer.cli.{gemm_benchmark, probe_ladder, train_benchmark,
  sdpa_benchmark, known_answer_probe, forward_benchmark, probe_trace, ...}`
- **Produces:** one `RunRecord` JSON per run under `/pub/wagnera3/{bench,gemm,
  sdpa,ladder,trace,...}`
- **Provenance:** `RunRecord` + `RunFingerprint` — image digest, GPU model,
  driver, determinism posture, host, package versions. The only surface here
  that carries all six axes.
- **Scale:** 131 ledger rows, the largest body of cluster work.

### `cleargbm` — ClearGBM benchmarks and covenant-radar optimisation

- **Repo:** this one — `libs/cleargbm`, `libs/cleargbm_rs`, `libs/covenant_ml`,
  `services/covenant-radar-api`
- **Runs:** `scripts.optimize -b cleargbm`, `scripts.benchmark_cleargbm_*`
- **Produces:** `libs/cleargbm/docs/BENCHMARK_MANIFEST_*.json` (41 of them) and
  `services/covenant-radar-api/models/optimization_history.jsonl`
- **Provenance:** partial, in two different ways.
  - The six `benchmark_cleargbm_*` entry points pin BLAS threads and build a
    `RunFingerprint` as of 2026-08-27, but the record shape is
    `BenchmarkManifest`, not `RunRecord`.
  - `optimization_history.jsonl` carries a `RunFingerprint` as of 2026-08-28
    — host, packages and image digest — where before it recorded
    `best_val_auc` and `duration_seconds` and nothing about what produced
    them. The 3,068 rows written before that state `"fingerprint": null`
    explicitly, which reads as "nobody recorded one" rather than "there was
    nothing to record"; a missing key is refused outright.
  - **`scripts/optimize` still pins nothing.** It was not among the six entry
    points that got a pin, so its fingerprint honestly reports the
    determinism stack as `none`. The record is now true; the runs are still
    not reproducible against themselves. Fixing that means pinning before
    numpy loads, which `scripts/optimize/__init__.py` currently prevents by
    importing the world at package import time.
- **Scale:** 108 ledger rows.

### `floor` — cloze floor scoring

- **Repo:** this one, `services/Model-Trainer`
- **Runs:** `modeltrainer-score-baseline --experiment extraction-eval`
- **Produces:** `/pub/wagnera3/floor/results/*.json`
- **Provenance:** `RunRecord`, and its known answers are registered so a
  re-run is checked against an established value rather than merely recorded.
- **Scale:** 7 ledger rows.

### `turkic-lstm` — character-level LSTM for Turkic languages

- **Repo:** `~/PROJECTS/LSTM` (separate Poetry project; depends on
  `platform-core` by git rev, NOT by relative path — see below)
- **Runs:** `runs/sweep-turkic-bases.json`, seven members, one per language
- **Produces:** `/pub/wagnera3/LSTM/checkpoints/<lang>_best.pt`; locally
  `results/*.csv` plus a `RunRecord` sidecar per evaluation
- **Compares:** `zero_shot_excess_ce_*.csv` carries `excess_cross_entropy` —
  one model's cross-entropy minus another's — with confidence intervals,
  across seven languages and six arms (`pilot_a/b/c`, `variant_b`, `v3`,
  `2026-02`, `rebuild_2026-08`). Files named `_forMoldir` and a commit
  crediting a Finnish native reviewer indicate this is bound for publication.
- **Provenance:** a `RunRecord` sidecar as of 2026-08-28. Every
  `zero_shot_eval` run writes `<results>.csv.runrecord.json` beside its CSV:
  experiment `turkic-zero-shot-excess-ce`, the OOV regime as the label, one
  named observation per ordered language pair, a SHA-256 of the CSV as the
  payload digest, and a `RunFingerprint` carrying the host and the resolved
  `torch`/`numpy` versions. It states the card and driver as absent because
  the scoring path genuinely uses neither, and the determinism stack as
  `none` because it pins nothing — both true.

  **The CSVs already in `results/` have no sidecar** and cannot get an honest
  one retroactively — nobody recorded what produced them. Re-running the
  evaluation is what fills the gap for anything going into the paper.
- **Onboarded 2026-08-28**, and the blocker was worth recording. Training had
  never run on the cluster: `slurm/train_base.sub` was a careful, unused
  array job pointing at `/pub/wagnera3/LSTM` and `/pub/wagnera3/envs/lstm`,
  neither of which existed. An earlier version of this page read that script
  as a description of practice and said so — which is the exact failure this
  index exists to prevent, made on its own first day.

  What actually blocked it was one line: `platform-core` was added as
  `{ path = "../API/libs/platform_core" }`, which resolves beside
  `~/PROJECTS/API` and cannot resolve on HPC3, where the monorepo is at
  `/pub/wagnera3/api` — lowercase, case-sensitive filesystem. It is now a git
  dependency pinned by the lock file, which carries no layout assumption and
  records exactly which `platform_core` computed a run's fingerprint.

  Now provisioned and verified: checkout, environment (`torch 2.5.1+cu124`,
  `numpy 2.4.6`), and the v3 corpora staged with cluster-side digest
  verification against `runs/turkic-v3-corpus-digests.txt`. All seven sweep
  members preflight clean, 84 GPU-hours against a declared 84-hour cap.
  `slurm/train_base.sub` is deleted rather than kept beside the new path.
- **The sweep pins the card to an A100.** The hpc3 contract refuses a generic
  `--gres=gpu:1`, so the array job's "whatever is free" placement is gone.
  That trades queue time for arms whose numbers can be subtracted from each
  other, which is the whole point of the exercise.
- **Corpus: `rebuild_2026-08/corpora_clean_v3`, and getting there was the
  sharpest lesson of the day.** The sweep first trained from `corpora_clean`,
  because `slurm/train_base.sub` named it. That was wrong. There are THREE
  generations, and the directory names say nothing about which is current:

  | directory | budget | binding | status |
  |---|---|---|---|
  | `corpora_clean_2026-02/` | 10,215,670 | Uyghur | superseded |
  | `corpora_clean/` | 12,642,807 | Uzbek | superseded |
  | `rebuild_2026-08/corpora_clean_v3/` | **11,658,775** | **Uzbek** | **current** |

  `overleaf-tu-paper/LM_MI_LSA_template.tex` states 11,658,775 with Uzbek
  binding — v3, and `train_v3.log` used it too. Meanwhile
  `turkic-transliteration/docs/tu-proceedings-datasets-section.tex` still
  describes the 2026-02 build; that draft section is stale relative to the
  paper it feeds, and now carries a banner saying so.

  Fixed: the sweep points at v3, v3 is staged and digest-verified on the
  cluster, and the base copy staged in error was removed. `LSTM/CORPORA.md`
  is the marker that would have prevented the mistake and now exists.

- **v3's transliteration inputs are fully accounted for.** Its manifest
  records eight digests and all eight match `turkic-transliteration` today:
  seven `*_ipa.rules` by file digest, and `symbol_map` by TABLE digest —
  `corpus/clean.py` hashes the parsed rows re-encoded as JSON, not the CSV.
  Its seven siblings are file digests, so comparing the CSV's hash and
  concluding the map drifted is a mistake someone will make. It was made
  here on 2026-08-28 and asserted in three places before being caught.
  Reproduce with `read_symbol_map()`: 18 rows, `9a3b98c8…`.

  `corpora_clean/` records no rule digests at all — that is the real gap of
  the three. For `corpora_clean_2026-02/`, the draft section states that the
  producing script "is not in either repository" and used a classifier never
  wired into the released package; that is the author's open item, concerns
  raw-corpus filtering upstream of cleaning, and is **not established** to
  apply to `corpora_raw_v3`.

  The obvious suspect was checked and cleared: the 2026-08-12 `U+02A6`
  ligature merge landed before every corpus here. Zero `U+02A6` in any
  Kyrgyz file; 19,421 merged forms in v3.

- **Corpora come from the engine, not a corpus repo.**
  `~/PROJECTS/turkic-transliteration` holds `src/turkic_translit/rules/*.rules`
  and the cleaner; its `data/` is empty. The corpora are its output and live
  in LSTM.

- **Nothing in the ledger yet.** Preflight admits; no job has been submitted.

---

## Not registered anywhere

Real research, producing numbers that get compared, reachable by no tool.

### RustedWarfareBot — system identification against an obfuscated binary

- **Repo:** this one, `clients/RustedWarfareBot`
- **Produces:** `runs/*.log` (seeded: `aa-s12345`, `aa-s1337`),
  `sweeps/*.txt` (`aggression`, `army-mix`, `antiair`, `aa-cover`, ...),
  `models/fleetdoom.ndjson`
- **Compares:** seeded runs across parameter sweeps against a stated goal —
  "100% win rate against the built-in AI at Impossible and every rung below,
  measured".
- **Provenance:** its own notion — the README says it "pins every claim to the
  build it was measured on", which is the right instinct and a different
  vocabulary from `RunFingerprint`.
- **Runs locally**, not on the cluster.

### `sirius` — declared as an example, never run

- **Repo:** none. **Confirmed 2026-08-29, and the answer is negative** — the
  entry above used to say "unconfirmed … consistent with being the
  destination, but nothing states the link and it should be confirmed before
  being relied on." It has now been checked, and the link does not hold:
  `~/PROJECTS/metabolomics-dashboard` contains **zero** occurrences of
  `sirius` or `zodiac` in any `.py`, `.R`, `.Rmd`, `.md` or `.json` file. It
  assigns formulas with **MFAssignR in R** (`run_stage1.R`,
  `run_mfassignr.Rmd`, `stage1_state.RData`). `cho_formulas_assigned.csv` is
  MFAssignR's output, not SIRIUS's.
- **Not on the cluster either.** `/pub/wagnera3/envs/sirius` and
  `/pub/wagnera3/sirius` do not exist, `sirius` is not on `PATH`, and no
  `sirius` module is available.
- **Status: deliberately NOT onboarded, and this is the finding rather than a
  task left undone.** Registering it would declare a project whose
  environment does not exist, running a tool the repo it names does not use.
  That is the failure `c38fcc52` documents on this index's own first day —
  a script read as practice and asserted — reproduced on purpose. The
  `sirius` entries in `tools/hpc3/README.md` and
  `examples/chain-sirius-zodiac.json` are ILLUSTRATIONS of the chain shape
  and nothing more; they are kept because the shape is worth showing, and
  they are named here so nobody mistakes them for a registration.

---

## The shared record, and who uses it

`platform_core.run_record.RunRecord` is the one shape a research run is meant
to emit: an experiment name, a label, named observations, a payload digest,
and a `RunFingerprint` saying what produced them. `platform_core.comparability`
then decides whether two of them may be subtracted.

Its consumers, as of 2026-08-29:

- **Model-Trainer's CLIs** — the original adopters.
- **`covenant_ml` benchmarking** — emits one beside every manifest it writes
  (`benchmark_run_record`). The manifest holds the per-seed detail; the
  record holds the claim, in the vocabulary `compare_run_records` checks. It
  had a fingerprint and its own record shape until now, which is why nothing
  could read its numbers beside another experiment's.
- **LSTM** — `char_lstm.provenance` writes a `.runrecord.json` beside every
  results CSV.

Still outstanding, stated precisely:

- **`covenant-radar-api`'s optimisation history** carries an explicit
  three-state `fingerprint` per row but no `RunRecord`. Its entry point now
  pins the BLAS thread count (2026-08-29), so future rows are at least
  reproducible against each other; the 3,068 rows written before that are
  not, and their `fingerprint: null` says so.
- **RustedWarfareBot** has neither.

THE CLAIM THAT USED TO STAND HERE WAS FALSE. This paragraph said "LSTM and
RustedWarfareBot cannot adopt it at all, because `platform_core` is not
installable outside this monorepo." Both halves are wrong: LSTM adopted it on
2026-08-28 via a git dependency pinned by its lock file, and RustedWarfareBot
is IN this monorepo — its sibling `clients/TankpitBot` already declares
`platform-core = { path = "../../libs/platform_core", develop = true }`. The
obstacle was never installability.

That is the gap this index exists to make visible rather than to hide, and
the sentence above is what happens when it is described from memory instead
of checked.

## Adding a research project

1. Add an entry to `projects` in a workspace document under
   `tools/hpc3/runs/` — resources, `budget`, and `repo`.
2. Add a section here. `test_committed_runs.py` fails if a registered project
   is missing from this file.
3. Emit `RunRecord`s from whatever produces the numbers.
4. Submit through the hpc3 CLI rather than a hand-written `sbatch` script, so
   the run lands in the ledger and `hpc3-trace` can answer "which job produced
   this artifact".
