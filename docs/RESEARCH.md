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

---

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
  `numpy 2.4.6`), and corpora staged with cluster-side digest verification
  against `runs/turkic-base-corpus-digests.txt`. All seven sweep members
  preflight clean, 84 GPU-hours projected against a declared 84-hour cap.
  `slurm/train_base.sub` is deleted rather than kept beside the new path.
- **The sweep pins the card to an A100.** The hpc3 contract refuses a generic
  `--gres=gpu:1`, so the array job's "whatever is free" placement is gone.
  That trades queue time for arms whose numbers can be subtracted from each
  other, which is the whole point of the exercise.
- **Corpus caveat, and it is the sharper of the two open questions.** The
  sweep trains from `corpora_clean`, the base corpora `train_base.sub` named.
  Recent local training (`train_v3.log`) used
  `rebuild_2026-08/corpora_clean_v3` instead, which stages separately to
  `/pub/wagnera3/mi/corpora`. Pointing the sweep at v3 is a one-line change
  per member — but it is a research decision, and the two sets differ in
  provenance as well as content:

  | | `corpora_clean` (base) | `corpora_clean_v3` |
  |---|---|---|
  | equalized char budget | 12,642,807 | 11,658,775 |
  | records cleaning params | yes | yes |
  | records **which rules built it** | **no** | yes (8 digests) |

  Neither repository stores corpora — `~/PROJECTS/turkic-transliteration` is
  the *engine* (`src/turkic_translit/rules/*.rules` plus the cleaner); its
  `data/` is empty. The corpora are the engine's output, and live in LSTM.

  Checked 2026-08-28 against the engine as it stands: v3's seven `.rules`
  digests still match exactly, and its `symbol_map` digest matches **neither**
  version in that repo's history, in any line-ending form. So **v3 is not
  byte-reproducible from what is in version control**, and the base set
  records nothing to reproduce from at all.

  The known-risky map change — merging the `U+02A6` ligature for Kyrgyz,
  committed 2026-08-12 with the note that "corpora published before then
  carry it" — is *not* the discrepancy. Both Kyrgyz files were checked
  directly: zero `U+02A6` in either, 19,374 merged forms in the base set and
  19,421 in v3.
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

- **Repo:** unconfirmed. `examples/chain-sirius-zodiac.json` runs the SIRIUS
  `formula` and `zodiac` subcommands, which are metabolomics tools;
  `~/PROJECTS/metabolomics-dashboard` contains `cho_formulas_assigned.csv`,
  which is consistent with being the destination, but nothing states the link
  and it should be confirmed before being relied on.
- **Status:** appears only in `tools/hpc3/examples/`, with zero ledger rows.
  A second PI's work (accounts are per-PI; see the account comment in LSTM's
  `slurm/train_base.sub`) that was scoped and never onboarded.

---

## The shared record, and who uses it

`platform_core.run_record.RunRecord` is the one shape a research run is meant
to emit: an experiment name, a label, named observations, a payload digest,
and a `RunFingerprint` saying what produced them. `platform_core.comparability`
then decides whether two of them may be subtracted.

Its consumers today are Model-Trainer's CLIs and nothing else. `covenant_ml`
benchmarking has a fingerprint but its own record shape; `covenant-radar-api`'s
optimisation history has neither; LSTM and RustedWarfareBot cannot adopt it at
all, because `platform_core` is not installable outside this monorepo.

That is the gap this index exists to make visible rather than to hide.

## Adding a research project

1. Add an entry to `projects` in a workspace document under
   `tools/hpc3/runs/` — resources, `budget`, and `repo`.
2. Add a section here. `test_committed_runs.py` fails if a registered project
   is missing from this file.
3. Emit `RunRecord`s from whatever produces the numbers.
4. Submit through the hpc3 CLI rather than a hand-written `sbatch` script, so
   the run lands in the ledger and `hpc3-trace` can answer "which job produced
   this artifact".
