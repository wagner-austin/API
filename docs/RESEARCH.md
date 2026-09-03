# Research index

**Every body of work on this machine that produces numbers someone compares.**
Read this before auditing, extending, or reproducing any experiment.

This file exists because nothing like it did, and the cost was measured: on
2026-08-28 an audit of provenance across this machine examined four research
surfaces and missed two entirely — LSTM and RustedWarfareBot — because there
are roughly ninety directories under `~/PROJECTS` and no list. Both have since
been onboarded — LSTM as `turkic-lstm` on 2026-08-28, RustedWarfareBot as
`rusted` on 2026-08-29 — leaving one entry below that was scoped as an example
and deliberately never registered.

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

What each project declares is RENDERED from the workspace documents below, not
retyped here. Every hand-written restatement of these numbers in this file has
been wrong at least once — see the `rusted` entry's correction — so the table
is generated and `hpc3-research-index` fails when it drifts.

<!-- generated: hpc3-projects. Do not edit by hand. -->

Rendered from `tools/hpc3/runs/hpc3*.json`. Regenerate with `hpc3-research-index --write`.

| project | partition | gpu | cpus | mem GiB | minutes | image | deterministic | ckpt steps |
|---|---|---|---|---|---|---|---|---|
| `cleargbm` | free | cpu | 4 | 16 | 60 | `0a525f532a9e` | yes | 0 |
| `floor` | free-gpu | `A100` x1 | 8 | 32 | 60 | `df841c661b9e` | yes | 0 |
| `mi` | free-gpu | `A100` x1 | 8 | 64 | 240 | `55651342e15d` | yes | 500 |
| `rusted` | free | cpu | 4 | 2 | 100 | `b1eaaa2e5a43` | yes | 0 |
| `tankpit` | free | cpu | 2 | 2 | 60 | `0cfdd5592a1a` | yes | 0 |
| `turkic-lstm` | free-gpu | `A100` x1 | 4 | 16 | 150 | `6e034383e300` | no | 27344 |

<!-- /generated: hpc3-projects -->

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
    `RunFingerprint` as of 2026-08-27. **This entry said until 2026-09-03
    that the record shape was `BenchmarkManifest` and not `RunRecord`. That
    is wrong.** `benchmarking/provenance.py` has carried
    `benchmark_run_record`, `benchmark_observations` and `benchmark_label`
    since the fingerprint landed, and writes both: the manifest holds the
    per-seed detail, the record holds the claim, and neither contains the
    other. A session acting on the old sentence rewrote a module that already
    existed before reading the file; the duplicate was reverted in
    `5e53cf13`.
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
  members preflight clean.

  **This paragraph read "84 GPU-hours against a declared 84-hour cap" until
  2026-09-03, and no configuration ever said that.** `hpc3-turkic-lstm.json`
  allocates `minutes` per member and declares
  `budget.self_imposed_gpu_hours: 36.0`. Eighty-four is seven times twelve,
  which is a number nobody measured — the same shape of mistake as reading
  `slurm/train_base.sub` as a description of practice, recorded above.

  What the local runs actually took, from `LSTM/train_v3.log` and
  `train_v3_lane2.log` on 2026-08-15: seven languages in two lanes sharing
  one consumer GPU, 00:29 to 05:25, so **under five hours wall-clock and
  roughly ten GPU-hours in total**. Per-language figures from those logs are
  upper bounds rather than measurements, because the two lanes advanced in
  lockstep and each interval is bounded by the slower of the pair; the
  largest such interval is 2h02m.

  The per-member limit was raised from 90 to 150 minutes on 2026-09-03. Ninety
  sat below the 2h02m upper bound already observed on a shared consumer card,
  which is a limit set under the measurement rather than over it.

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

### `rusted` — RustedWarfareBot, system identification against an obfuscated binary

- **Repo:** this one, `clients/RustedWarfareBot`
- **Runs:** `rw_bot.harness.campaign_match`, one scheduled job per match, via
  a campaign document emitted by `scripts.campaign_doc`
- **Produces:** `runs/*.log` (seeded: `aa-s12345`, `aa-s1337`),
  `sweeps/*.txt` (`aggression`, `army-mix`, `antiair`, `aa-cover`, ...),
  `models/fleetdoom.ndjson`; on the cluster, one scorecard per match under
  `/pub/wagnera3/rusted/runs/sweeps/<batch>/`
- **Compares:** seeded runs across parameter sweeps against a stated goal —
  "100% win rate against the built-in AI at Impossible and every rung below,
  measured".
- **Provenance:** `rw_bot.provenance`, since 2026-08-29 — a `RunFingerprint`
  per sweep and a `RunRecord` per arm, in the shared vocabulary. The README's
  instinct ("pins every claim to the build it was measured on") was right and
  is now executed rather than described. Its observations are the arm's win
  rate with the counts beside it — three wins from three and thirty from
  thirty are both 1.0, and only one is evidence — plus extractor drops,
  median worth, unengageable targets and intercepts.
- **Sizing, read off `runs/hpc3-rusted.json` rather than described:** four
  CPUs, 2 GB, 100 minutes on `free`, `requeue` on, `deterministic` on,
  `checkpoint_steps: 0`. The zero is honest because the per-match scorecard
  IS the checkpoint: a preempted match costs one match.
- **Declares an image**, `/pub/wagnera3/rusted/images/v4/rusted.sif` pinned
  by sha256 `b1eaaa2e`, binding `/pub/wagnera3`, with `env_path` `/opt/env`
  inside it.
- **Not yet submitted, and what is missing is the staged game tree**, not the
  image. Nothing has been run against the cluster to see how that failure
  presents, so no claim is made here about which command reports it first.
- **This entry disagreed with the registry until 2026-09-02**, claiming one
  CPU, 45 minutes and no image at all, against a workspace document committed
  seven minutes earlier in `b81c7f91` that declared four CPUs, 100 minutes
  and an sha256-pinned image. The prose was corrected against the registry,
  which this file's own preamble names as the machine-readable half. Worth
  keeping as a worked example: `test_committed_runs.py` passed throughout,
  because it asserts that every registered project APPEARS here and that
  declared repo paths exist, and nothing compares a sizing sentence against
  the numbers it describes. Presence is enforced; agreement is not.

---

### `tankpit` — TankpitBot, the sim as a measurable opponent

- **Repo:** this one, `clients/TankpitBot`
- **Runs:** `tankpit-sim-run` — the production `Bot` playing a timed session
  against `sim/server.py` on real field terrain, no browser and no network
- **Produces:** `runs/sim/sim-<stamp>.capture_session.json` and
  `.world.json`, plus the probe event stream; `tankpit-feature-rows` reshapes
  an events artifact into one tick-indexed row per decision
- **Compares:** sessions across doctrines and world parameters — the bot's
  own policy against itself, which is what the tick corpus is a design matrix
  for.
- **Provenance:** `RunRecord` since 2026-09-02, on the feature-row
  derivation. **Its honest limit is stated rather than papered over:** the
  record describes the DERIVATION and identifies the live run only by a
  digest of its events artifact, because an events record carries no build
  stamp, commit or version — nothing recorded what produced the 539 archived
  runs and no fingerprint written now can claim it. Stamping the build at
  emission time is filed separately.
- **Sizing, measured 2026-09-02 rather than guessed:** a 150-round practice
  session ran 144 s wall and peaked at 26 MiB of Python allocation on the
  workstation. Declared 2 CPUs, 2 GB, 60 minutes on `free` — the wall clock
  is roughly twenty times the measured session so a slower node and a longer
  soak both fit, while staying far under the partition's 72-hour cap.
- **`deterministic: true` is a measurement, not an assumption.** Two
  independent sessions with the same named layout and population seed
  produced a byte-identical `world.json`
  (`0bc360232d812984b403783c631e2f01…`, 60 rounds, 2026-09-02), and the same
  digest again from two SEPARATE PROCESS invocations. That is what lets the
  project declare `checkpoint_steps: 0` honestly under the
  `requeue AND (checkpoints OR deterministic)` clause.

  **The evidence ladder, stated so nobody reads it as more than it is:**
  same-process replay ✓, cross-process on one host ✓, **cross-node on the
  cluster ✓ — measured 2026-09-03, and on one pair only.** Jobs `55715577`
  and `55718398` produced a byte-identical `world.json`
  (`673447d2e720812d…`) across two different NODES (`hpc3-15-23`,
  `hpc3-15-25`), two different IMAGES (`b838e0242ecc`, `0cfdd5592a1a`) and
  the code change between them. That is a stronger result than the flag
  needed and a weaker one than `rusted`'s: this is a single pair, where
  rusted's panel was twelve seeds across two arms. One pair cannot see an
  intermittent divergence, which is precisely the failure rusted found.

  **Image `b838e0242ecc` (v1) was deleted on 2026-09-03** as a superseded
  120 MB artifact, so this measurement now stands on what is RECORDED
  rather than on what can be re-run: the ledger row for `55715577` carries
  the digest and both `world.json` files are still on disk, but
  re-executing that half of the pair is no longer possible. A future
  cross-node panel should be built from images that still exist rather
  than extended from this pair.

  `rusted` is the standing warning here: its
  panel found cross-invocation replay "achievable on this runtime and does
  not always happen", with two members bit-exact across nodes 40 minutes
  apart while their counterparts moved (`9ae66117`), and it declared
  `deterministic: false` until that was resolved. The mechanism differs —
  that was a JVM game engine, this is pure integer Python with a tick-paced
  clock and no wall-time input to outcomes — which is why the flag is `true`
  here rather than deferred. It is a mechanism argument plus a same-host
  measurement, NOT cluster evidence, and the first cluster runs should
  re-measure it before anything is subtracted across nodes.

  Note also that `free` is `PreemptMode=CANCEL`, so `requeue` is inert on it;
  the flag's live consequence here is the comparability axis, not restart
  behaviour.
- **The stamp is no longer an input to the world.** It selected the practice
  layout AND the container-population seed until 2026-09-02, so an array
  whose tasks stamp themselves varied the room and the larder along with
  whatever it meant to vary — that cost a retracted saturation table.
  `--layout` and `--population-seed` now state the world, and omitting
  either under `--sweep` or a set `SLURM_ARRAY_TASK_ID` is refused rather
  than defaulted.
- **Registered but NOT YET RUNNABLE, and this is the honest state rather
  than an oversight.** `/pub/wagnera3/envs/tankpit` does not exist. The
  monorepo IS staged at `/pub/wagnera3/api` (at commit `80221ea`, behind this
  tree), the cluster's system Python is 3.9 where this package needs 3.11,
  and there is no Poetry on the login node. `hpc3-preflight` reports the
  missing environment, which is the correct refusal; nothing has been
  submitted. The remaining work is provisioning, not registration.
- **It ships an image, and the image is self-contained.** v2 at
  `/pub/wagnera3/tankpit/images/v2/tankpit.sif`, sha256 `0cfdd5592a1a…`,
  127 MB, `env_path` `/opt/env`, built from commit `bccf5afa`.

  The first registration declared `image: null`, reasoning that four of five
  projects run from a directory environment and this payload is "pure
  Python". That was a popularity argument and it was wrong: `rusted` is also
  CPU-only on `free` with `requeue` and `deterministic`, and it carries an
  image. The image answers both real blockers — the cluster's system Python
  is 3.9 where this needs 3.11, and a directory environment reads its
  payload from the mutable `/pub/wagnera3/api` checkout.

- **v1 ran, and the three submissions it took are the useful record.** The
  distribution did not carry its own data: the XOR key was read four parents
  above its module (site-packages after an install) and the field minimaps
  by bare CWD-relative names. **The two failed by different mechanisms**, so
  fixing one did not fix the other — `55715554` died on the GIF, and
  `55715564` then died on the key *even though the key had just been staged
  beside the GIF*. `55715577` completed only with the assets staged, a
  working directory set, and `TANKPIT_XOR_KEY_FILE` passed per run.

  Fixed at the packaging layer rather than the image layer (`bccf5afa`,
  [[packaged-data-assets]]): the assets ship inside `tankpit_bot.data` and
  `tankpit_bot.resources` addresses them through `importlib.resources`. The
  checkout-relative constant, the environment override, the CWD candidate
  list and the container's COPY/ENV were deleted rather than kept — an
  override is a second answer to a question that must have one, and it is
  what let one defect grow two independent workarounds.

  **The proof is the run document.** v2's is a bare command line, where v1's
  needed a shell, a working directory and an environment variable to find
  files the wheel should always have carried.
- **Bootstrapping the first image needed a step the documented flow does not
  cover.** `hpc3-image-capture` probes an existing environment over SSH at
  `env_path`, and the four-command flow starts with capture — so a project
  with no environment has nothing to capture from. A bootstrap environment
  was built by hand first (`/pub/wagnera3/envs/tankpit`, Python 3.11.16 taken
  from the interpreter inside `envs/cleargbm`, since the module system offers
  3.8, 3.10 and 3.14 but no 3.11). It is disposable now the image exists.

---

## Not registered anywhere

Real research, producing numbers that get compared, reachable by no tool.

### `code-style` — QLoRA on this monorepo, scored by this monorepo's own guards

- **Repo:** this one. Corpus emitter `tools/code-corpus`, instrument
  `tools/code-style-eval`, training through `services/Model-Trainer`'s
  `hf_lm` backend.
- **Runs:** local on the 3090 Ti, not on the cluster. The instrument IS this
  repository's checkers — `ruff`, `mypy` and `monorepo_guards` scoped per
  item through `scripts/guard.py --root` — so running it inside an image
  would measure that image's copy of the rules rather than the repo's. That
  is a reason to keep it local, not an omission.
- **Produces:** `code-corpus-v1.jsonl` plus its holdout; a QLoRA adapter
  (NF4 storage, bfloat16 compute, double quantization); per-arm generation
  manifests recording whether each completion terminated or hit the token
  budget; per-arm outcome JSONL, one row per item per checker;
  `comparison.json` with a paired 2x2 table and McNemar mid-p; and
  `perplexity.json`.
- **Compares:** base Qwen2.5-Coder-1.5B against the same model with the
  adapter attached, on the same held-out files, two ways — token-level
  perplexity masked to the continuation, and guard-pass rate. Per-item
  outcomes throughout, so the contrast is paired rather than two rates
  subtracted.
- **Provenance:** a `RunRecord` sidecar beside `comparison.json` since
  2026-09-02, in the shared vocabulary. Three of its six fingerprint axes are
  empty BY CONSTRUCTION and say so: scoring is CPU work outside any image and
  pins no determinism, so image, GPU and driver are unknown, and the
  determinism posture is recorded as explicitly unpinned. Its packages axis
  carries `ruff`, `mypy` and `platform-core`, because a guard-pass rate is
  those checkers' verdict and a ruff release that adds a rule moves the
  number with nothing about the models having changed. Observations carry
  counts beside rates.
- **What it does NOT carry, and this is the honest half.** The generation
  configuration
  (base model, adapter, decoding parameters) is covered only indirectly, by
  the comparison's payload digest over the outcome files. And the corpus was
  emitted while both source repositories were dirty, which the manifest
  flags; no run built on it is citable until it is re-emitted clean.
- **A claim in this entry was wrong for one day, and the correction is kept.**
  It said the training run emitted no `RunRecord` and recorded only a
  determinism posture. Training in fact captured a full `RunFingerprint` all
  along, written into the manifest beside the weights by the same
  `capture_run_fingerprint` the benchmarks use: the 2026-09-01 adapter names
  an RTX 3090 Ti on driver 591.86 with `torch 2.6.0+cu124`. The real gaps
  were narrower — the manifest is not a `RunRecord`, so nothing could compare
  it against another experiment, and its package axis named `numpy`, `torch`
  and `transformers` while a QLoRA run's arithmetic is decided by `peft` and
  `bitsandbytes`, neither of which was recorded. Both are closed as of
  `92183bbd`. The lesson is the one this file already carries twice: check
  the artifact before writing what it contains.

- **Results as of 2026-09-02, stated with their limits.** Perplexity moved
  2.8327 to 1.9631 with 392 of 392 held-out items improving, and train/holdout
  overlap was checked to be zero by path AND by content — the content check
  is the one that matters here, because `scripts/guard.py` is byte-identical
  across all 41 packages. Guard-pass showed NO detectable difference across
  three sweeps (mid-p 0.84 on the last), and the combined rate sits near 2%,
  which is a floor where the metric has almost no power to move. The first
  two sweeps were void for reasons recorded on the board: a token budget that
  truncated 83% of completions, and before that an unscoped guard invocation
  that gave every item the same verdict.
- **Not novel, and the task spec that says otherwise is wrong.** A systematic
  search found the core already published: ContextCov (arXiv 2603.00822)
  compiles a repository's written conventions into executable AST and
  architectural checks, and per-repository LoRA is the upper-bound baseline
  in Code2LoRA (arXiv 2606.06492). The papers are on the personal wiki under
  `computational-linguistics`. The defensible claim is fitness for purpose —
  no public benchmark scores THIS repo's conventions — not originality.

---

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
- **RustedWarfareBot** — `rw_bot.provenance` builds one per sweep arm.
  Its fingerprint's load-bearing axis is neither a card nor a library: it is
  the **game**, recorded in the packages axis as three digests. The project
  already knew the build decides everything — every wiki page pins
  `game_version` "because the jar is obfuscated and class names change
  silently between releases" — but that pin is a hand-maintained string on
  documentation, and silent renaming is exactly the case a maintained label
  notices last. The digests are read off the bytes that ran. Two arms
  measured against different builds now refuse to subtract.

  It carried only the first of the three until 2026-08-29, and the other two
  were found by asking what the jar digest does NOT cover:

  - `rusted-warfare` — SHA-256 of `game-lib.jar`, the engine's code.
  - `rusted-warfare-jvm` — the bundled runtime's own `JAVA_VERSION` followed
    by a digest of its whole tree. The two platforms ship **different major
    versions** — Java 8 in the Linux depot, Java 13 in the Windows one — so
    this is not a formality. The host axis separates those two today by
    accident, because the operating systems differ; two Linux runs either
    side of a depot that bumped its bundled JRE fingerprinted identically.
  - `rusted-warfare-assets` — a digest of `assets/`, the maps, mods and unit
    definitions the simulation reads. The project had already lost a batch
    family to this exact gap: a map missing from a clone sent the engine to
    its boot sandbox and voided every scorecard, with the jar digest matching
    throughout.

  The tree digests are `rw_bot.tree_identity`, and they are deliberately
  reproducible with coreutils alone — `find … | LC_ALL=C sort | xargs
  sha256sum --text | sha256sum` — because a record only one package can
  verify is a record nobody checks.

Still outstanding, stated precisely:

- **`covenant-radar-api`'s optimisation history** carries an explicit
  three-state `fingerprint` per row but no `RunRecord`. Its entry point now
  pins the BLAS thread count (2026-08-29), so future rows are at least
  reproducible against each other; the 3,068 rows written before that are
  not, and their `fingerprint: null` says so.

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
