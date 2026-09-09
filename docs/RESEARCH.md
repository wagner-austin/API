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

| project | partition | gpu | cpus | mem GiB | minutes | image | deterministic | resumes |
|---|---|---|---|---|---|---|---|---|
| `cleargbm` | free | cpu | 4 | 16 | 60 | `0a525f532a9e` | yes | no |
| `code-style` | free-gpu | `A100` x1 | 8 | 32 | 240 | `5dfd78a7eb14` | yes | no |
| `floor` | free-gpu | `A100` x1 | 8 | 32 | 60 | `df841c661b9e` | yes | no |
| `mi` | free-gpu | `A100` x1 | 8 | 64 | 240 | `55651342e15d` | yes | no |
| `mi-cu128` | free-gpu32 | `RTX6000` x1 | 8 | 64 | 55 | `6d9ba0baac40` | yes | no |
| `rusted` | free | cpu | 4 | 2 | 100 | `b1eaaa2e5a43` | yes | no |
| `tankpit` | free | cpu | 2 | 2 | 60 | `0cfdd5592a1a` | yes | no |
| `turkic-lstm` | free-gpu | `A100` x1 | 4 | 16 | 150 | `6e034383e300` | no | yes |

<!-- /generated: hpc3-projects -->

### `mi` — Model-Trainer probes and benchmarks

- **Repo:** this one, `services/Model-Trainer`
- **Runs:** every entry point below, named in full. THE LIST USED TO END IN AN
  ELLIPSIS, and that ellipsis was the defect: prose that says "and others"
  cannot be checked, so a command could produce compared numbers for weeks
  while appearing nowhere. `cartridge_qa_benchmark` did exactly that — it
  carried the cartridge programme's headline result with 0 committed run
  documents, 0 lines here and 0 tracked artifacts, and the claim was retracted
  on 2026-09-09 for sitting four times below what its instrument could
  resolve. `score_baseline`, with more committed run documents than any other
  entry point in the workspace, was equally absent. The
  `research-registration` guard rule now fails `make lint` on any entry point
  that builds a `RunRecord` and is not named here, so this list cannot go
  stale silently again.

  `model_trainer.cli.{cartridge_base_lora_sweep, cartridge_benchmark,
  cartridge_companion_sweep, cartridge_composition_sweep,
  cartridge_content_lora_sweep, cartridge_diverse_companion_sweep,
  cartridge_headroom, cartridge_qa_benchmark, cartridge_solo_grid,
  cartridge_solo_seeds, cartridge_varied_companion_sweep, continuations,
  forward_benchmark, gemm_benchmark, gemm_probe, known_answer_probe,
  known_answer_registry, legacy_gemm_probe, probe_ladder, probe_trace,
  score_baseline, score_run, sdpa_benchmark, sdpa_probe, train_benchmark,
  train_step_probe, triple_edit_benchmark}`

  Naming a command here records that it produces a comparable number. It does
  NOT assert that the number is adequately powered — that is the separate
  minimum-detectable-effect sweep's question, and `cartridge_qa_benchmark` is
  the standing proof that a surface can be registered and still be running an
  instrument too weak for what it reports.
- **Produces:** one `RunRecord` JSON per run under `/pub/wagnera3/{bench,gemm,
  sdpa,ladder,trace,...}`
- **Provenance:** `RunRecord` + `RunFingerprint` — image digest, GPU model,
  driver, determinism posture, host, package versions. The only surface here
  that carries all six axes.
- **Scale:** 131 ledger rows, the largest body of cluster work.

#### `cartridge_benchmark` — cartridge capacity and composition on a real base

Added 2026-09-03. Measures a trained key-value prefix against a real
pretrained model over a real corpus, replicated across seeds. It exists
because the cartridge strategy's unit tests measure a two-layer, two-head
model with random weights, and three of the conclusions drawn from that model
did not survive a real one — the tiny model stops gaining at ~8 slots and
then loses, while gpt2 is still gaining at 512 with no saturation point in
range; composition retains ~59% rather than ~25%; and an untrained prefix
goes from harmless to −0.7612 on held-out text.

- **Command:** `python -m model_trainer.cli.cartridge_benchmark --plan
  gpt2-wiki --corpus <dir> --second-corpus <dir> --device cuda --controls
  none --out <file>`
- **Needs, on a compute node:** `HF_HOME=/pub/wagnera3/hf
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`, the same prefix `floor` and the
  `mi` training runs already use, plus both corpora staged under a bound
  path. Verified to need no network: the local run that produced the numbers
  above ran under `HF_HUB_OFFLINE=1`.
- **Second corpus is required and must be UNRELATED to the first.** Composing
  two cartridges trained on two halves of one corpus measured 94% retention,
  and the number was an artifact — each half already predicted the other.
  Against an unrelated corpus the same code reports 59%.
- **Belongs to `mi`, and should override `mi`'s size.** `mi` is the right
  project by definition — Model-Trainer probes and benchmarks out of
  `services/Model-Trainer` — and a project here is a resource/budget/image
  profile, not a topic, so this needs no project of its own. But `mi`
  defaults to 240 minutes and 64 GiB, and one `gpt2-wiki` plan is ~10
  minutes over a 124M base with a prefix of at most 512 slots. Booking the
  default is not free: `free-gpu` is preemptible, so a 240-minute window
  exposes a 10-minute job to preemption it never needed to risk. State
  `minutes` and `mem_gb` in the run document; overrides are validated
  exactly as a hand-authored spec is.
- **`--controls` is required and has no default.** It names which cross-card
  controls the run applies (`none` / `split-k` / `attention` / `both`, from
  `core/services/model/control_arms.py`), and the fingerprint records the
  ones that were applied. Required for the same reason `--second-corpus` is:
  the two arms compute different numbers, so a record whose posture was
  guessed names a condition it may not have run under. Every number reported
  on this page was measured under `none`.
- **One run document is committed:** `tools/hpc3/runs/cartridge-gpt2-wiki-v100-v32.json`,
  against image `65762bbd4d30` (v32) — the first `mi` image built from a
  commit containing this command. It pins a V100 rather than `mi`'s default
  A100, and the reason is a measured incident rather than a preference — the
  A100-pinned attempt sat PENDING five hours on a partition whose A30s and
  V100s were free, and `hpc3.contracts.gpu_supply` carries the account and
  the preflight rule that now refuses it. Until v32 existed
  there was deliberately no document, because the then-registered image
  (`55651342e15d`, v23) predated the command and a run naming it would have
  asserted something untrue.
- **Read the spread, not just the mean.** Every arm reports one, and the
  sweep's separations are judged against the largest spread among the *sweep*
  arms — not against every arm, because the composition arm trains two
  cartridges over a doubled prefix and is noisier for reasons that say
  nothing about the sweep.
- **A plan is reproducible from its seeds.** It was not until 2026-09-03:
  training drew dropout from a process-wide RNG nothing seeded, so two runs
  of one plan disagreed. That was first written up here as GPU contention,
  because the two runs happened to differ in machine load — a coincidence
  with a plausible story attached. Whether contention affects these numbers
  is still unmeasured.

#### `cartridge_composition_sweep` — retention versus compartment count

Added 2026-09-04 (board task `a67d6038`). Measures what a trained cartridge
retains as more independently trained cartridges are composed in front of
it, at N in {2, 4, 8} under two slot policies (fixed 64 per cartridge;
fixed 512 total), with three built-in controls: an untrained-composed arm
per configuration (structure-versus-content attribution), a cross-gain arm
per other corpus (each foreign cartridge scored alone on the primary
held-out text, the relatedness detector), and the fixed-policy alone arms
being the same configuration at every N (exact agreement is a free
replication check, and it held).

- **Command:** `python -m model_trainer.cli.cartridge_composition_sweep
  --plan gpt2-compartments --corpus <dir> --other-corpora <d1,d2,...>
  --device cuda --out <file>`. Comma-joined Windows-form paths from Git
  Bash — MSYS converts only the first `/c/...` in an argument, and the
  mangled remainder reaches Python as a directory that globs empty.
- **Result, measured 2026-09-04 on the 3090 Ti, driver 591.86, offline:**
  the compartmental limit is TWO. Clean-roster retention (all cross-gains
  negative): fixed-64 goes 62.8% at n2 to −45.4% at n4 to −7.0% at n8;
  budget-512 goes 44.3% to +14.4% to −7.0%. Replicated across three roster
  rotations: n2 sits at 59–73% whichever corpus partners (identity moves
  it ±7pp), n4 is negative in every fixed-policy roster tested, n8 erases
  the gain under both policies. The v2 record is bit-identical across two
  processes (sha256 `aa61330b9692…`), and an earlier pair agreed on 90 of
  90 shared observations across a record-shape change.
- **Attribution:** at n2 the cost is structural — noise slots alone retain
  41% and trained content adds about twenty points back; by n4 content
  interference crosses over and trained strangers cost more than noise.
- **Two artifacts this run caught in its own first roster:** tech-wiki and
  hpc3-wiki cartridges predict me-wiki text sight-unseen (+0.18, +0.41
  cross-gain) because the operator's narrative wiki shares their
  vocabulary, which inflated the first n8 reading to +27% where the clean
  number is negative. Relatedness between compartments is measured by the
  cross-gain arm, never assumed. And a hostile cross-gain does not predict
  composition damage: the most hostile-alone corpus (−0.42) composed as
  benignly as the friendliest at n2.
- **No run document is committed, deliberately:** the registered `mi` image
  predates this command, so a committed run naming it would assert something
  untrue. (`cartridge_benchmark` was in the same position until image v32;
  its document is now committed against that digest.) The follow-on —
  composition-aware training, per the
  ICAE multi-span finding — belongs to board task `292c3272`.

#### `cartridge_companion_sweep` — composition-aware training moves the ceiling

Added 2026-09-04 (board task `bc29dc3e`), and it answers the sweep above:
the two-compartment ceiling is a property of naive training, not of
composition. Every cartridge in a grid cell is trained with a frozen
companion present at a swept per-step probability, two companion kinds
(fresh noise; a plain-trained cartridge on a corpus HELD OUT from every
composition partner, refused by the CLI if it overlaps), and the same arms
and controls as the baseline sweep.

- **Command:** `python -m model_trainer.cli.cartridge_companion_sweep
  --plan gpt2-companions --corpus <dir> --other-corpora <d1,d2,d3>
  --companion-corpus <dir> --device cuda --out <file>`
- **Result, measured 2026-09-04 on the 3090 Ti, bit-identical across two
  full-grid processes (record sha256 `9e87e816…`):** the trained-companion
  recipe at p=0.5 puts four-compartment retention at **+44.6%** where the
  naive baseline was **−45.4%** — a +0.78 swing on the composed mean
  against a 0.049 floor, from the grid's tightest cell — while
  two-compartment retention rises 62.8% → 78.3% and the solo cost is four
  hundredths of gain. Content-companionship beats noise-companionship on
  every axis. The overdose endpoint is real in both kinds: p=1.0 training
  destroys solo performance (noise −0.68; trained −0.32, whose composed
  arms then BEAT its alone arm — a cartridge adapted to company).
- **The retention observation is conditional by design:** absent where a
  cell's alone arm did not improve on the base, because a ratio against a
  non-gain has no reading; the raw arm means always carry the verdict.
  That rule exists because the p=1.0 collapse is a real cell every full
  grid hits, and the first version of the CLI died on it.
- **The n8 extension ran ON THE CLUSTER 2026-09-04** (board task
  `684492dd`, plan `gpt2-companions-n8` added in `acaedbb3`, image v34
  `cdd1341b…` built from that commit with the plan asserted by the image's
  own smoke, job 55753007, V100, run document
  `tools/hpc3/runs/cartridge-companions-n8-v100-v34.json`, 0 SU): **the
  recipe survives seven-stranger deployment after single-companion
  training.** Trained-p0.5 puts eight-compartment retention at **+26.5%**
  (composed +0.2243, spread 0.0562) where the naive baseline was −7.0%
  (composed −0.062) — a +0.29 swing, ~5× the cell floor — at the same
  four-hundredths solo cost. Trained-p0.25 reads +22.2%. Noise
  companionship's n4 break-even VANISHES at n8 (−4.4/−4.6%,
  indistinguishable from naive): content is the load-bearing ingredient,
  and at n8 the trained cells' composed arms match their untrained-composed
  controls (+0.02) — content interference erased, while noise-trained
  cartridges lose −0.39/−0.46 to real content. plant-eco (the seventh
  partner) measured +0.05 cross-gain against a 0.048 spread — clean,
  verified in-run. Dose ordering (p0.5 > p0.25, trained > noise)
  replicates from the n2/n4 grid.
- **The retention observation is conditional by design:** absent where a
  cell's alone arm did not improve on the base, because a ratio against a
  non-gain has no reading; the raw arm means always carry the verdict.
  That rule exists because the p=1.0 collapse is a real cell every full
  grid hits, and the first version of the CLI died on it.
- **The V100 replication ran 2026-09-05 and EVERY VERDICT SURVIVES THE
  CARD** (job 55773639 on v36, run document
  `tools/hpc3/runs/cartridge-companions-n2n4-v100-v36.json`, 0 SU): the
  full grid — both kinds, all three probabilities including the p=1.0
  collapse — reads trained-p0.5 at 78.0%/41.4% retention (vs the 3090 Ti's
  78.3%/44.6%), with dose and kind orderings identical, the overdose
  endpoint replicated in both kinds, and every p<1 alone arm agreeing
  across cards to within a hundredth. Cross-card deltas are largest
  exactly in the collapse cells, where the spreads are too.
- **Open extensions, filed rather than implied:** the budget slot policy
  under companioned training; the scale rung (`gpt2-medium` on the
  cluster, a 7B base on A100 — cheap for cartridges, since only slots
  carry optimizer state). Varied-count exposure was measured the same day
  and REFUTED at its target — see the next subsection.

#### `cartridge_varied_companion_sweep` — varied-count exposure, refuted at its target

Added 2026-09-04 (board task `7815a0fd`), attacking the n4→n8 retention
decay: every cartridge trains beside a DRAWN number (uniform 1..K when
present) of frozen seed-variant companions from one held-out corpus, the
pool's first member byte-identical to the recorded single companion so the
records compare as supersets.

- **Command:** `python -m model_trainer.cli.cartridge_varied_companion_sweep
  --plan gpt2-companions-varied --corpus <dir> --other-corpora <d1..d7>
  --companion-corpus <dir> --device cuda --out <file>`
- **Result, measured 2026-09-04 on HPC3** (jobs 55759514/55761217 on one
  V100, image v35 `4e02f3b0…` from `0aacbd24`, records bit-identical,
  sha256 `1fd6bb9d…`, run documents
  `tools/hpc3/runs/cartridge-companions-varied-v100-v35{,-twin}.json`,
  0 SU): **the hypothesis is refuted at its target.** At trained-kind,
  p=0.5, K=3: n8 retention is **+18.3%** against the single-companion
  +26.5% — the composed difference (−0.073) sits at ~1× the cell spreads,
  so at best flat, plausibly worse. n4 improves modestly to **+51.0%**
  (vs 44.6%) with the composed spread collapsing 0.049 → 0.010, the
  tightest composed cell in the program, at a higher solo cost (−0.063
  vs −0.044).
- **The mechanism reading is the finding:** count-invariance WAS learned —
  the untrained-composed controls (+0.60 at n4, +0.30 at n8) sit far above
  every earlier grid's noise-composition arms — yet real strangers still
  interfere at both counts. The count-decay is CONTENT interference, not
  count shock, confirming the n8 finding from the opposite direction:
  content-companionship is load-bearing, and a pool of three same-corpus
  companions cannot teach content diversity.
- **The motivated follow-on is a content-DIVERSE pool** (companions drawn
  from several held-out corpora), deliberately excluded from this cell as
  a second variable; this record is its baseline. It ran the same night —
  next subsection.

#### `cartridge_diverse_companion_sweep` — the best recipe, and the decay's cause settled

Added 2026-09-05 (board task `d2c03dd4`), the named lever after the
varied-count refutation: the pool's three members each train on a
DIFFERENT held-out corpus (epi — the recorded companion, byte-identical
by the shared seed formula — plus metabolomics and atmospheric-chemistry),
and a NEW companion-cross instrument scores every pool member alone on
the primary held-out, so companion leakage is measured in-record where
every earlier grid assumed it.

- **Command:** `python -m model_trainer.cli.cartridge_diverse_companion_sweep
  --plan gpt2-companions-diverse --corpus <dir> --other-corpora <d1..d7>
  --companion-corpora <c1,c2,c3> --device cuda --out <file>`
- **Result, measured 2026-09-05 on HPC3** (job 55772675 + twin 55773234,
  V100, image v36 `0401aa9b…` from `d894c6b2`, run documents
  `tools/hpc3/runs/cartridge-companions-diverse-v100-v36{,-twin}.json`,
  0 SU): **the program's best recipe at both counts.** n4 retention
  **+55.5%** (composed +0.4608; vs single-companion 44.6% at ~1.4× floor)
  and n8 retention **+28.0%** (composed +0.2323) — decisively above the
  same-content pool's 18.3% (~1.4× floor) and a within-floor TIE with the
  single companion's 26.5%, stated as a tie. Solo cost −0.060. All three
  companion-cross arms NEGATIVE (epi −0.36, metabolomics −0.11,
  atmospheric-chemistry −0.04): the pool is measured clean.
- **The mechanism verdict that settles the arc:** with the diverse pool
  the n8 composed arm EQUALS its untrained-composed control (+0.2323 vs
  +0.2431, within a third of a spread) — content interference at n8 is
  fully trained away, so the residual n4→n8 decay is STRUCTURAL slot
  dilution (seven 64-slot strangers are 448 foreign positions against 64
  own), which no companionship recipe can remove because the cost no
  longer has a content component. Three arms (n8 extension, varied
  refutation, diverse recovery) converge on this from three directions.
- **Companionship is exhausted as an n8 lever; the follow-ons are
  capacity-side:** the budget slot policy under diverse-companioned
  training, base-side composition LoRA (teach the BASE to read crowded
  prefixes, fixing every cartridge at once), and the scale rung — larger
  bases where 448 foreign slots are a smaller fraction of attention.
- **The scale rung ran 2026-09-05 and INVERTS the softer-ceiling
  hypothesis** (plan `gpt2-medium-companions-diverse`, added `e5476201`
  with the differs-only-in-the-base contract pinned by test and by the
  v37 image's own smoke; job 55776517 + twin 55786853, V100, image v37
  `2adee62f…`, run documents
  `tools/hpc3/runs/cartridge-medium-diverse-v100-v37{,-twin}.json`,
  0 SU): on a base three times the size under the identical recipe,
  **n4 transfers near-exactly (+54.1% vs gpt2's +55.5%) and n8
  COLLAPSES (−86.6% vs +28.0%)**. The controls attribute it: medium's
  n8 untrained-composed arm is itself negative (−0.29 vs gpt2's +0.24) —
  the 24-layer base's structural tolerance for a 512-slot foreign prefix
  is far worse than the 12-layer base's before content enters — and
  composed sits another 0.42 below that, so the recipe's content-erasure
  did not transfer either. The schedule is not the confound: the same
  schedule learns the solo cartridge (+0.81) and composes four
  compartments (+0.44) on medium; only the crowded-prefix regime fails.
  Depth compounds prefix interference; scale alone COSTS many-compartment
  composition rather than buying it. **The 7B path runs through
  base-side adaptation** (composition LoRA), now the only standing n8
  lever at scale; n4 deployment is scale-robust at ~55% on both bases.

#### `cartridge_base_lora_sweep` — the base learns the crowd, and the two levers stack

Added 2026-09-05 (board task `6c752568`), the arm the whole cartridge-side
arc pointed at: a rank-8 LoRA on the base's attention (`c_attn`) trains to
do language modeling behind a DRAWN number (uniform 1..8) of frozen
composed cartridges from the three held-out pool corpora, then the
recorded grid's own measurement functions run with the adapted base
underneath — `train_on`, the loop that trained every recorded cartridge,
drives the LoRA unchanged, with only which side learns switched. Two cell
families: plain-trained cartridges on the adapted base (is base-side
alone enough?) and diverse-companioned cartridges on it (do the levers
stack?). Companion-cross and per-family floors carried over; every alone
arm prices the LoRA itself.

- **Command:** `python -m model_trainer.cli.cartridge_base_lora_sweep
  --plan gpt2-base-lora --corpus <dir> --other-corpora <d1..d7>
  --pool-corpora <c1,c2,c3> --device cuda --out <file>`
- **gpt2 result, measured 2026-09-05 on HPC3** (job 55787810 + twin
  55788364 BIT-IDENTICAL ACROSS V100 NODES, sha256 `efad0a93…`, image
  v38 `13bd47e9…` from `e1bc2009`, run documents
  `tools/hpc3/runs/cartridge-base-lora-v100-v38{,-twin}.json`, 0 SU):
  **the levers attack different components and stack to program bests.**
  LoRA + plain cartridges repairs the STRUCTURAL catastrophe (n4
  −45.4% → −6.9%; noise-composition controls leap n4 −0.12 → +0.28) but
  plain cartridges still bleed content interference — base-side is
  necessary, not sufficient. LoRA + diverse cartridges: **n4 +58.1%, n8
  +33.3%** (vs diverse-alone 55.5%/28.0%; the n8 gain is ~2× the 0.027
  floor, the program's tightest family). Solo cost ≈ zero for plain
  cartridges, a solo GAIN for diverse ones. Full n8 ladder: naive −7.0 →
  lora-plain −5.6 → same-content 18.3 → single 26.5 → diverse 28.0 →
  **lora+diverse 33.3**.
- **gpt2-medium result, measured 2026-09-05/06** (plan
  `gpt2-medium-base-lora` differing only in the base, pinned by test and
  by the v39 image's smoke; job 55790169, image v39 `0bd7983b…` from
  `f7f696a8`, run document
  `tools/hpc3/runs/cartridge-medium-base-lora-v100-v39{,-twin}.json`;
  twin 55798416 ran on hpc3-gpu-18-02 against the original's
  hpc3-gpu-17-02 and the two records are byte-identical — sha256
  `372cee59da557974421fc8a6e38ba67cf53ee3b51a488e4dfc1848e29eabef77`
  both, a CROSS-NODE certificate, the strongest class): **a split
  decision that completes the mechanism map.** The structural repair TRANSFERS to depth — medium's
  n8 noise-composition control flips −0.29 → +0.42, a +0.71 swing on the
  exact quantity the scale rung measured as the collapse's structural
  half — and n4 sets a new medium best (**+59.3%** vs 54.1% without the
  LoRA). But medium's n8 with real content still collapses (−79.4% vs
  −86.6%; composed −0.62 sits 1.04 BELOW the repaired noise control,
  where gpt2-small's gap is 0.15) with a 0.53 cell floor — seed-chaotic,
  near a bifurcation. **Depth amplifies content interference in a way
  neither current lever touches**; structure is solved at both scales.
- **Operating points this fixes for the serving design:** base-LoRA +
  diverse cartridges at up to FOUR simultaneous compartments is
  scale-robust and best-in-program (~58–59% on both bases); eight is
  deliverable on the 12-layer base (+33.3%) and NOT on the 24-layer base,
  where a ≤4-compartment scope router remains the honest deployment.
- **Open, filed rather than implied:** a content-side lever for depth
  (more diverse voices in medium's pool, or content-aware LoRA
  objectives) with the medium record as baseline; the budget slot policy
  under the stacked recipe; the 7B rung once a content-at-depth lever
  exists.

#### `cartridge_content_lora_sweep` — crowd-invariance distillation closes the content gap at depth

The content lever the previous subsection filed, run the same day (board
task `a85fbabe`, operator-directed). The LM objective cannot close the
content gap even in principle — it rewards reading past a crowd's
SHAPE, not ignoring what the crowd SAYS — so the lever changes the
objective and nothing else: same LoRA, same pool, same grid, same seeds,
plan rows pinned equal to `gpt2-medium-base-lora` field for field by
test and by image smoke.

- **The objective** (`cartridge_content_lora.py`, commit `7288bc8f`):
  per step, draw a roster of frozen pool cartridges and a TARGET member,
  take the window from the target's own corpus, and minimise the KL from
  the plain base's predictions behind the target ALONE to the adapted
  base's predictions behind the full roster. The target is drawn over
  every roster position (no positional shortcut; every compartment stays
  live, matching serving semantics) and counts draw 1..8 (a count of one
  distils "the LoRA does not disturb the alone case"). Teacher is a
  second frozen base instance — PEFT injects adapters into the wrapped
  module tree, so one loaded base cannot play both roles.
- **Result, measured 2026-09-06 on HPC3** (job 55801429, V100, 103 min,
  image v40 `798d234a…` from `7288bc8f`, run document
  `tools/hpc3/runs/cartridge-medium-content-lora-v100-v40{,-twin}.json`;
  twin 55801941 reproduced the record byte for byte, sha256
  `9abd901a1346d854f1085a553ffee3002f3d60ad3773a02b4c1d3731ec0ca34e`
  both — a SAME-NODE certificate, the queue having placed both runs on
  hpc3-gpu-16-01; the pipeline's cross-node determinism is separately
  established by the arc's four cross-node certificates): **the depth
  collapse is repaired, and then some.**
  Diverse n8 at medium: composed +0.3090 against alone +0.8113 —
  **+38.1% retention where the LM objective recorded −79.4%**, clearing
  the 0.1625 family floor by 1.9× and BEATING gpt2-small's own best n8
  (+33.3%). Diverse n4 sets the program record at **+63.2%** (was
  +59.3%). Plain cartridges flip positive at both counts (n4 −11.5% →
  +30.1%, n8 −48.4% → +15.8%). The content gap — composed below its own
  noise-composition control — shrinks from **1.04 to 0.30** at n8. And
  the alone arms RISE (+0.8503/+0.8113 vs +0.8058/+0.7817): the
  invariance objective is free at solo, as the count-1 anchor was built
  to guarantee. KL converged 290 → 153 → 129 over three epochs;
  companion-cross stays negative (no leakage).
- **What this settles for the serving design:** with crowd-invariance
  distillation on the base, EIGHT simultaneous compartments are
  deliverable at depth (+38.1%), and four compartments at +63.2% is the
  best cell the program has produced on any base. The ≤4 scope router
  for deep bases is no longer forced by the measurements. The two
  named interference mechanisms — structural and content — now each
  have a working lever, and both levers live base-side.
- **gpt2-small anchor, measured 2026-09-07** (job 55806539 + twin
  55806826 on hpc3-gpu-18-01 and hpc3-gpu-16-00, ~40 min each, records
  CROSS-NODE BIT-IDENTICAL sha256
  `9abfbdd4a053bc7c16b7af3451f2307603b4b9c27c96b5ade201a4d6091eaad6`,
  same v40 image as the medium record so the comparison isolates exactly
  the base; run documents
  `tools/hpc3/runs/cartridge-content-lora-v100-v40{,-twin}.json`):
  **crowd-invariance beats the LM objective at both scales, and the n4
  ceiling is scale-invariant.** Diverse n4 +63.3% (vs +58.1% under the
  LM objective) — matching medium's +63.2% to a tenth of a point.
  Diverse n8 +49.6% (vs +33.3%): the objective wins even where nothing
  was collapsing. Plain flips positive at both counts (+17.3%/+17.4%),
  alone arms rise again (+0.9053/+0.8584), KL converges 224 → 150 → 135.
  Depth still prices n8 (49.6% at 12 layers vs 38.1% at 24) but no
  longer breaks it.
- **The 1.5B rung (gpt2-xl, 48 layers), BOTH objectives, measured
  2026-09-07** (jobs 55808450 LM / 55808466 invariance, ~3.5 h each on
  A30-24GB after the first content attempt 55807973 hit CUDA OOM on
  V100-16GB by 50 MiB at the KL step — two 1.5B fp32 models — so both
  runs moved cards together, `gpu_pinned_because` declared; image v41
  `f38bc982…` from `40c55fa5`, run documents
  `tools/hpc3/runs/cartridge-xl-{base,content}-lora-a30-v41{,-twin}.json`;
  twins 55809977/55809982 BYTE-IDENTICAL, sha256 `cff9f3ce…` (LM) and
  `6269120d…` (invariance) — SAME-NODE certificates, the queue having
  placed all four runs on hpc3-gpu-l54-09; cross-node determinism is
  separately established by the arc's five cross-node certificates):
  **the count penalty vanishes at 1.5B, so the 24-layer n8 collapse is a
  MID-DEPTH VALLEY, not a depth law.** Under the LM objective diverse
  retention reads n4 +54.8%, n8 +54.8%; under invariance +52.8%/+52.8%
  — n8 EQUALS n4 to a tenth of a point under both objectives (composed
  means differ by 0.0004/0.0001 against floors of 0.019/0.046; each
  record's own separation flag reads 0.0), where 24 layers priced n8 at
  −79.4% (LM) and 12 layers at +33.3%. The two objectives TIE on
  diverse at 1.5B (composed means 0.4452-0.4456 vs 0.4367-0.4368,
  within both floors — stated as a tie); the invariance objective's
  remaining margin is plain cartridges, positive at both counts
  (+22.5%/+5.8%) where the LM objective leaves them negative
  (−11.5%/−27.9%), and the ladder-wide fact that it collapses NOWHERE.
  The n8 content gap (composed below its untrained-composed control)
  reads ~0.21 under BOTH objectives at 48 layers — depth's content
  amplification at 24 layers (1.04) does not extrapolate. KL converged
  148 → 121 → 110; LM loss 2.81 → 2.67 → 2.60; every companion-cross
  and diverse-cross arm stays negative in both records (pools measured
  clean).
- **The ladder verdict** (acceptance of the scale-ladder arm): diverse
  retention n4/n8 across 12 → 24 → 48 layers reads 58.1/33.3 →
  59.3/−79.4 → 54.8/54.8 under the LM objective and 63.3/49.6 →
  63.2/38.1 → 52.8/52.8 under crowd-invariance. Four-compartment
  serving is scale-robust at +53-63% everywhere; the n8 question is
  depth-shaped, worst at 24 layers, gone at 48; crowd-invariance is the
  only objective that never collapses on any rung and the only one that
  keeps plain cartridges positive everywhere it was measured.
- **The 7B rung (Pythia-6.9B, GPT-NeoX, NF4), BOTH objectives, measured
  2026-09-07** (board task `af35fc20`; jobs 55810964 LM 3h04m /
  55810966 invariance 3h21m on A30-24GB; image v42 `270d8197…` from
  `09f5a4bb`, 44/44 smokes green including the dtype-boundary and
  architecture-policy self-asserts; the NF4-forced boundary cast — fp32
  master slots cast to the model's compute dtype at `layer_blocks`,
  read off the embedding weight — committed in `44f95b52` with the fp32
  path pinned byte-unchanged by test AND by image smoke; model identity
  content-pinned in the run documents, snapshot `c0e3eee3` + both
  safetensors shard sha256s): **the recipe does not survive the
  architecture jump AS-IS — and the failure is the measurement's
  PRECONDITION, not composition.** The solo cartridge gain nearly
  vanishes: alone arms read +0.068 diverse / −0.055 plain (LM) and
  +0.211 / +0.229 (invariance) against ~0.81 on every GPT-2 rung, with
  per-seed spans (−0.19..+0.19 LM, −0.09..+0.40 invariance) the same
  order as the means, so retention ratios on these records are division
  artifacts and composition (negative everywhere, both objectives)
  cannot be attributed. The adaptation halves work normally — LM loss
  2.89 → 2.74, KL 98 → 91, every cross arm negative (pools clean) — so
  what failed to transfer is specifically the 64-slot cartridge's solo
  gain under GPT-2-tuned hyperparameters on a 4-bit 4096-dim base.
  Candidate mechanisms filed, deliberately unattributed: base headroom
  (decidable by an absolute base-loss measurement on the held-out
  corpora), NF4/bf16 slot-gradient precision (decidable by an
  unquantized bf16 rung), hyperparameter scaling, and the measured
  large floors.
- **NF4 determinism is CERTIFIED, and the one divergence is explained
  to the byte:** the LM pair (55810964 + twin 55811523, both placed on
  hpc3-gpu-l54-09) is BYTE-IDENTICAL, sha256 `6ef4b9c9…` — a same-node
  certificate. The invariance pair (55810966 on hpc3-gpu-k54-01 + twin
  55811542 on hpc3-gpu-l54-09) diverges in record sha256 (`254d0126…`
  vs `8c48e82a…`) while ALL 198 OBSERVATIONS ARE BIT-EQUAL: the sole
  differing field in either file is `fingerprint/host/logical_cores`
  (32 vs 64 — the two A30 nodes' CPU counts), the fingerprint doing
  its job. That makes the invariance pair a CROSS-NODE
  observation-identity certificate — the stronger of the two — and
  4-bit dequant + bf16 compute + the boundary cast reproduce
  bit-for-bit in every measured quantity across nodes.
- **The headroom measurement, run the same day (board task `afee6162`,
  `cartridge_headroom` CLI, commit `77e12c3e`, image v44 `9e98d0a7…`;
  jobs 55812858 + twin 55812861, four minutes each, records
  BYTE-IDENTICAL sha256 `7668c51d…`, same-node hpc3-gpu-k54-01; the
  first v43 pair died in 19 s on a device-placement bug a cpu-only
  suite cannot see as a crash — fixed and pinned by a recording fake
  in `77e12c3e`): **headroom is the dominant mechanism of the 7B
  solo-gain collapse.** Plain-base held-out loss on the primary corpus
  runs 4.57 (gpt2) → 4.27 (medium) → 4.11 (xl) → **3.66
  (pythia-6.9b/NF4)**, and pythia sits 0.34-0.90 nats below every
  GPT-2 base on ALL eleven corpora — so the 7B's plain base already
  predicts at the level the smaller bases reach only WITH a cartridge
  (gpt2's adapted level ≈ 4.57 − 0.81 ≈ 3.76). Most of the recorded
  ~0.81-nat gain was never available at 7B. The remaining distance
  down to xl's adapted level (3.66 − ~3.30 ≈ 0.36 nats) matches the
  best measured 7B per-seed gain (+0.40) within the cells' floors —
  CONSISTENT with a family-common adapted floor, stated as consistency
  and not proof. What headroom does not explain is reliability: seeds
  7/8 reach ~0.4 while seed 9 goes negative in the same cell, a
  training-variance component headroom cannot produce. Tokenizer
  caveat carried in-record: pythia tokenizes the same text ~7% shorter
  (3328 vs 3584 held-out tokens on the primary), so its per-token
  losses are mildly INFLATED relative to gpt2's at equal compression —
  the headroom reading is conservative, not flattered.
- **The nine-seed reliability measurement, run the same day (board
  task `b89cd348`, `cartridge_solo_seeds` CLI, commit `4fc8dfbb`,
  image v45 `567cb42d…`; 7B pair jobs 55813508 + twin 55813516,
  ~18 min each, records BYTE-IDENTICAL sha256 `353ab575…`, same-node
  hpc3-gpu-k54-01; xl control pair 55818092 + twin 55813528 on
  DIFFERENT nodes (l54-09 / l54-07) with ALL 14 observations bit-equal
  and record shas differing solely in `fingerprint/host/logical_cores`
  64 vs 32 — a cross-node observation-identity certificate, matching
  the 7B invariance pair's; solo cartridges trained AND scored behind
  the PLAIN base at exactly the recorded knobs, read through the
  sweeps' own plan hook): **the family's tightness was never luck, and
  7B/NF4 cartridge training is genuinely low and unreliable.**
  gpt2-xl, nine seeds: mean +0.827, spread 0.079, every draw in
  [+0.776, +0.855]. Pythia-6.9B/NF4, the same nine seeds: mean
  **+0.160**, spread 0.320 — twice its own mean — worst seed −0.045
  (one hard failure in nine), best +0.275. The sweep records' ~0.4
  per-seed readings are explained, not contradicted: the sweeps'
  measurement pool trains and serves its cartridges behind the
  LoRA-ADAPTED base (verified in `cartridge_base_lora_sweep`'s
  `_MeasurementPoolProvider`), so those gains carried the adaptation's
  contribution; behind the plain base the cartridge alone delivers
  ~0.16. This also sharpens the headroom verdict: pythia base 3.66
  minus the measured 0.16 lands at ~3.50, still ABOVE xl's
  cartridge-adapted ~3.28 — the collapse is NOT purely
  headroom-to-a-common-floor, and the training-side deficit
  (GPT-2-tuned hyperparameters on 4096-dim KV geometry, or NF4
  gradient quality) is real and is what any recovery rung must fix.
- **The bf16 precision control (board task `c4b9a01b`, the loader's
  third declared state landed in `f4447989`, image v46 `bd6ca365…`;
  jobs 55833896 + twin 55833931, ~9 min each — half NF4's wall clock —
  records BYTE-IDENTICAL sha256 `445e345f…` ACROSS NODES gpu-24-07 and
  gpu-l54-07, the arc's first full cross-node byte identity on a 7B
  record): **NF4 is exonerated — the 7B training deficit is not the
  4-bit weights.** Same nine seeds, same knobs, quantization removed:
  mean +0.102, spread 0.248, two negative draws (vs NF4's +0.160,
  0.320, one negative; failures do not co-occur by seed, so failure is
  regime-level, not seed-intrinsic). Paired per-seed, bf16 − NF4 reads
  −0.058 ± 0.032 (t = −1.80, df 8, not significant; MDE 0.075 nats):
  whatever NF4 costs or saves is bounded far below the 0.67-nat
  family-vs-7B deficit. By elimination among the filed mechanisms, the
  training-side deficit is the HYPERPARAMETER/ARCHITECTURE mismatch —
  GPT-2-tuned lr/slots/epochs on 4096-dim KV geometry — riding on the
  attributed headroom component.
- **Minimum-detectable-effect rows for this program's published null
  claims** (adopting the machine-wide MDE standard, 2026-09-08; all
  computed from per-seed rows already in the records, no new runs):
  the 1.5B n8-equals-n4 tie holds with numbers attached — LM observed
  +0.0004 against an MDE of 0.019, invariance +0.0001 against 0.068 —
  and the objectives' diverse-n4 tie at 1.5B holds (−0.008 against an
  MDE of 0.047). ONE CLAIM IS CORRECTED: at 1.5B diverse n8 the paired
  test RESOLVES what the range-based floor could not — invariance sits
  −0.0088 ± 0.0014 below the LM objective (t ≈ −6.5), so "tie" was the
  wrong word. The difference is ~1% of the alone gain and changes no
  operating decision, but it is a resolved small LM advantage, not a
  tie, and the floor instrument's blindness to paired effects is
  exactly why these rows now exist.
- **The hyperparameter rung (board task `47d5f8c6`, operator-directed;
  `cartridge_solo_grid` CLI, commit `103bdaf7`, image v47 `290794b0…`
  from `fcb39991`; jobs 55841983 + twin 55841997, 1h08m each, records
  BYTE-IDENTICAL sha256 `bc18d701…`, same-node hpc3-gpu-k54-05; the
  in-grid anchor cell lr0.01×c64 reproduces the certified `445e345f`
  bf16 record BIT-FOR-BIT, seed for seed, so the grid reads):
  **REFUTED in the measured ranges — the recorded knobs already sit at
  the grid's maximum, and neither learning rate nor capacity recovers
  the 7B solo gain.** The lr bracket is well-formed around 0.01: at
  0.001 every seed is negative at both slot counts (means −0.17/−0.34),
  at 0.03 training diverges catastrophically (means −0.63 to −1.67,
  spreads up to 2.98, nine of nine negative), and 0.003 is
  indistinguishable-to-worse (−0.039 vs anchor, t −1.04). Capacity
  NEVER helps: 256 slots ≤ 64 slots at every learning rate, and at
  lr 0.003 significantly worse (−0.134 vs anchor, t −5.19). The
  winning cell IS the anchor — the recorded knobs — whose deployable
  NF4 form is already certified (`353ab575`), so acceptance's
  winner-under-NF4 leg is the existing record, not a new run. Standing
  verdict after three eliminations (headroom measured, NF4 exonerated,
  lr/slots refuted): at 12 epochs, ~0.10-0.16 nats IS the 7B solo
  regime for KV-prefix cartridge training on this corpus, and the
  ~0.36-nat residual to the family's adapted floor is not reachable by
  any knob measured so far. Bounds and MDEs, per the MDE standard
  (added after the closure audit correctly found them absent from this
  entry): the failure counts at n=9 carry exact 95% Clopper-Pearson
  intervals STATED AS BOUNDS — the anchor's 2/9 bounds the failure
  rate to [2.8%, 60.0%], the 9/9-negative cells to [66.4%, 100%], and
  even a 0/9 cell (none exists) would only bound it below 33.6% — so
  no cell's reliability claim at this n is tighter than these
  intervals allow. Paired-versus-anchor MDEs from the per-seed rows:
  0.087 nats (lr 0.003 × c64), 0.091 (lr 0.01 × c256), 0.060
  (lr 0.003 × c256) — the two indistinguishable cells could have hidden
  effects up to those sizes, an order below the 0.36-nat recovery the
  rung was hunting.
- **The epochs line closes the training axes, and the conclusion is
  about the METHOD (board task `e03cd293`; declared-cells refactor
  commit `fe692719` — `SoloGridCell`/`SoloCellSet` carry knobs AND
  recorded observation tokens as data, `cell_set_for` refuses
  undeclared selectors; image v48 `dd04cb08…`, 48 smokes; jobs
  55848106 + twin 55848109, ~55 min each, records BYTE-IDENTICAL
  sha256 `114acee2…`, same A30 pool; the 12-epoch cell reuses
  `ANCHOR_TOKEN` so its nine rows reproduce BOTH the `bc18d701` grid
  and the `445e345f` solo certificates BIT-FOR-BIT, seed for seed —
  the line reads):** **exposure does not recover the 7B solo gain;
  it destroys it.** Doubling epochs to 24 halves the mean (+0.046 vs
  +0.102; paired −0.055 vs anchor, t −2.14, under the 2.306 crit but
  with MDE 0.060 — any recovery ≥ 0.060 nats would have been seen,
  and none was). Quadrupling to 48 is catastrophic: mean −0.240,
  NINE OF NINE seeds negative (Clopper-Pearson 95% failure bound
  [66.4%, 100%]), paired −0.342 vs anchor (t −5.55, MDE 0.142). The
  per-cell negative counts bound as: e12 2/9 → [2.8%, 60.0%], e24
  1/9 → [0.3%, 48.2%], e48 9/9 → [66.4%, 100%]. The dose-response is
  monotone DOWN from the recorded knob, which now sits at the maximum
  of every training axis measured — learning rate, slots, epochs —
  with precision exonerated (bf16−NF4 null at MDE 0.075) and ~0.4
  nats of headroom demonstrably present. **METHOD-LEVEL CONCLUSION,
  stated as filed:** KV-prefix cartridge capacity does not transfer
  into pythia-6.9b at this scale under naive solo training. The
  ~0.10-0.16-nat regime is a property of the method on this
  architecture, not of any tuning knob; the ~0.36-nat residual to
  the family's adapted floor is unreachable by training-axis search.
  A 7B composition rung is therefore not merely unjustified but
  moot until the method itself changes (companioned/composition-aware
  training at 7B is the one measured lever left standing at smaller
  scales, and it is a different experiment, not a knob).
- **The companioned recipe at 7B closes the method question, and the
  answer is that nothing about KV-prefix cartridges transfers to this
  architecture (board task `68a96413`; precision-through-plans refactor
  commit `fd1f242a` — `precision_selector` declared per plan row,
  resolved through the one `resolve_precision` chokepoint, and the
  diverse sweep grew an in-record `naive-solo` arm; image v49
  `0a38233c…`, 49 smokes; jobs 55858713 + twin 55858759, ~2h07m each,
  records BYTE-IDENTICAL sha256 `c7fcd094…`; the naive arm reproduces
  the certified `445e345f` solo record BIT-FOR-BIT, all nine seeds —
  the record reads, and every companioned arm pairs per-seed against
  its own in-record baseline):** two eliminations in one record. THE
  SOLO AXIS: companioned-alone minus naive-solo is −0.0016 ± sd 0.1215
  (t −0.04, MDE 0.093) — composition-aware training moves the 7B solo
  gain by NOTHING; the ~0.10-regime survives the method change
  untouched (+0.1002 vs +0.1018; 1/9 negative → CP95 [0.3%, 48.2%]).
  THE COMPOSITION AXIS, where the recipe earned its name at smaller
  scales, INVERTS: n4 composed −0.216 (NINE OF NINE negative, CP
  [66.4%, 100%], in-record retention −216% against the medium record's
  +44.6% under the same recipe), n8 −0.284 (9/9), and the content
  effect flips sign — at n4 trained companions cost MORE than noise
  slots (paired −0.106 vs untrained-composed, t −3.38, MDE 0.072),
  the exact opposite of the +20-point content rescue the recipe bought
  at gpt2 scale. The companion-cross arms say why: a foreign 64-slot
  cartridge ALONE costs 0.75-1.0 nats on the primary held-out at 7B,
  an order beyond any gpt2-family cross effect — foreign KV-prefix
  content is actively toxic to this architecture's attention.
  **STANDING VERDICT, the 7B story complete:** training axes (lr,
  slots, epochs) eliminated at their maxima; precision exonerated;
  headroom present; and now the one measured method lever eliminated
  on both of its axes. KV-prefix cartridges do not transfer to
  pythia-6.9b — not the tuning, not the training, not the method. A
  compartmental design at this scale needs a different mechanism
  (the base-side LoRA lever and retrieval are the measured
  candidates), not a better cartridge recipe.
- **Records archive in-repo (2026-09-09, prompted by board 9d34f1bb):**
  every cartridge RunRecord whose sha256 is quoted here or in the
  api-codebase wiki — 38 cluster-produced, 8 austinpc-local, plus this
  arm's pair — is committed under
  `services/Model-Trainer/results/cartridge/` with `-text` protection,
  so every byte-identity certificate is verifiable by diffing two
  files in any checkout. "Produces on `/pub`" above remains where runs
  WRITE; this directory is where cited records LIVE. Archiving each
  arm's records is a standing closure step from this arm forward.
- **Open, filed rather than implied:** the mechanism of the mid-depth
  valley at gpt2 scales; the remaining 0.30 content gap at medium n8;
  the unswept pool size K=3 (registered as the sweep family's
  frozen-by-copy knob per the 2026-09-09 registration audit — a
  gpt2-scale question now, since no 7B rung remains to spend it on);
  and the 7B compartmental design itself, which after this record is a
  base-adaptation or retrieval question, not a cartridge one.

#### `cartridge_qa_benchmark` — can the model USE what the cartridge carries, and does it beat retrieval

Registered 2026-09-09 (board task `e3c833f7`), and the registration is the
first thing worth recording: this command produced the cartridge programme's
most-cited result while appearing in **0 of 571 committed run documents, 0
lines of this file, and 0 tracked artifacts**. Its records were written to a
temporary directory that is purged. It ran on the operator's local 3090 Ti,
where nothing requires an image digest or a staged corpus, and the reason it
was never registered is that nothing checked. The `research-registration`
guard rule (`9b011256`, corrected in `bd4dfb97`) now fails `make lint` on any
entry point that builds a `RunRecord` and is named nowhere here.

- **Command:** `python -m model_trainer.cli.cartridge_qa_benchmark --plan
  <name> --corpus <dir> --device cuda --controls <arm> --out <file>`
- **What it measures, and why it is not the loss benchmark.**
  `cartridge_benchmark` reports held-out loss; a model can memorise text
  word-by-word and still fail every question about it. This scores a
  multiple-choice question set built from held-out windows, over six arms:
  base alone, cartridge, BM25 retrieval, dense retrieval, reciprocal-rank
  fusion, and an oracle. `QA_EXPERIMENT` differs from `CARTRIDGE_EXPERIMENT`
  so the comparability layer refuses to difference the two.

- **THE HEADLINE WAS RETRACTED THE DAY THIS SECTION WAS WRITTEN**
  (`a98769b5`, `dc5f2408`). "The cartridge arm beats lexical, dense and fused
  retrieval from ~774M" rested on differences of 0.0521 and 0.0417 accuracy
  over **32 items** — 1.7 and 1.3 items. Under McNemar at alpha 0.05 with
  mid-p, the fewest disagreements that can ever reject is 5, so the claim sat
  roughly four times below the floor of the instrument that produced it. No
  split of that question set could have supported it.

  **What survives:** cartridge versus BASE moved 8.7 and 8.3 items at those
  rungs, above the floor and above its own seed spread. *"Cartridges improve
  the model over the un-augmented base from 774M"* stands. *"Cartridges beat
  retrieval"* does not, and is withdrawn rather than softened.

- **The refusal that now prevents the repeat.** A plan declares
  `smallest_effect_of_interest`, `alpha` and `mcnemar_test`, and
  `cartridge_qa_power.require_resolvable_question_set` refuses the run —
  before the model loads — when the REALISED question set cannot resolve what
  the plan declares. Against the realised count, never `max_items`: the
  32-item set came from a plan whose cap said 120. Every plan declares 0.05,
  which is the effect size this literature reports (WRAP +0.020, arXiv
  2401.16380; this machine's extraction ablation +0.061 for a removed 7:1
  dilution, +0.029 for permuted copies, +0.004 for hub-slug markers, which
  was noise). That needs 100 items, so **the four me-wiki plans are refused
  before they run** — deliberately, since they are the plans that produced
  the retracted claim.

  It is a FALSIFIABILITY gate, not a power one: passing means some attainable
  outcome supports the declared effect, never that the outcome is likely.
  Classifying an observed result against the discordant count that actually
  occurred is a separate statement.

- **Two axes added at registration, both previously absent.**
  `pythia-6.9b-api-wiki-qa` — the ladder had stopped at gpt2-xl, one rung
  past its own ~774M crossing, while the cartridge sweeps have run this base
  on an A30 since image v36; every field but `model_id` is the ladder's,
  `max_seq_len` included, so scale is not confounded with the retriever's
  budget. And `gpt2-large-api-wiki-qa-slots-{32,64,128,256}` — `num_slots`
  was 128 in every plan while the programme's stated mechanism for why a
  cartridge should lose to a retriever is that its slot budget is fixed and
  an index is not. That is a claim about a curve, and it had been measured at
  one point. All four cells hold `max_seq_len` at 768 rather than each
  cell's own `1024 - num_slots`, or the smallest cartridge would also carry
  the largest evidence budget.

- **The two gaps filed at registration are closed** (`baca6369`,
  `643757bd`). BM25's `K1`, `B` and `RETRIEVED_CHUNKS` were module constants,
  so "the cartridge beats BM25" named one arbitrary point in a parameter
  space; they are now `bm25_k1`, `bm25_b` and `retrieved_chunks` on the plan,
  carried on the index so a record reports the retriever that actually ran.
  And a **long-context arm** exists: `long_context_items` hands the model the
  corpus whole and lets the window truncate it.

  **Read its coverage before its accuracy.** The arm reports
  `long_context_corpus_fraction`, and it has to. `with_evidence` keeps the
  OPENING of the evidence and drops the rest — its docstring claimed the
  opposite until 2026-09-09 — so where the corpus overflows the window this
  arm is not "the corpus in context" but "the first few per cent of it,
  chosen by document order". At the current rungs the me-wiki corpus is
  15,602 tokens against a 896-token budget, so the arm carries about 6% and a
  cartridge beating it has beaten almost nothing. It becomes the honest
  long-context baseline only as that fraction approaches 1, which is an
  argument for the bigger bases and the bigger corpora rather than for the
  arm's absence.

- **Still open.** No reranker and no query-expansion arm, so the search side
  is parameterised but not yet widened. And **no run document is committed,
  deliberately**: no registered image carries the post-`cdb84e12` code, so a
  committed run naming one would assert something untrue — the same position
  `cartridge_composition_sweep` held until image v32.

### `mi-cu128` — the Blackwell determinism baseline

Registered 2026-09-04 (board task `9e4db632`, commit `3400be03`); the full
battery ran and closed the same day. First-hand section by the owner; the
onboarding miss that briefly left this file red — a registered project whose
name appeared nowhere here — was mine, and another session bridged it.

- **Repo:** this one, `services/Model-Trainer` + `clients/OrderedKernels`;
  workspace `tools/hpc3/runs/hpc3-mi-cu128.json`, spec
  `tools/hpc3/specs/abl-cu128-image.json`
- **Entry points:** `ordered_kernels.cli.{attn_probe, gemm_probe, score,
  train_step}`. Added 2026-09-09 by the `research-registration` guard rule,
  which found four commands here that build a `RunRecord` and were named
  nowhere in this file. The section discussed `ordered_kernels`' work in prose
  and named none of the commands producing it, which is the same defect the
  `mi` entry's ellipsis had. Names are qualified because the guard matches
  qualified paths: `gemm_probe` alone is ambiguous between this project and
  `model_trainer.cli.gemm_probe`, and they are different measurements.
- **Why it exists:** the `RTX6000` GRES on `free-gpu32` is 96 GB RTX PRO 6000
  Blackwell hardware, `sm_120`, which the cu124 image cannot drive — its CUDA
  runtime enumerates zero devices there. The blocker was software, not
  billing: those nodes are free (metered at zero, verified 2026-08-31).
- **Stack:** torch 2.7.1+cu128 / CUDA 12.8 / NVRTC 12.8.61, image
  `6d9ba0baac40` (cu128-v1, build 55749663, 38/38 smokes). Wheels built at
  `e007e999`, the SAME commit as the cu124 line's v33 image, and
  `transformers`/`cupy`/`numpy` held identical — so the two lines differ only
  in the CUDA stack.
- **Runs:** 23 jobs, all `COMPLETED`, 2026-09-04 (55751296–55752044): arch
  probes, gemm all five arms, train_step owned+ordered through xl, sdpa
  probe, attn_probe, floor150 rank1, and the full 2,627-item fully-owned
  score on both the RTX PRO 6000 (4:04) and the L40S (3:14). Records under
  `/pub/wagnera3/{gemm,train,attn,floor}/cu128-v1/`.
- **Verdict, within-stack:** ordered/owned records bit-identical across the
  Blackwell card and the L40S — gemm 186/186, train 1,283/1,283, attention
  stages 140/140, full-set outcomes one payload (`sha256:e964e46b…`),
  accuracy 1,374/2,627 on both. Vendor cuBLAS still speaks per-card dialects
  (18/93 digests shared), so the arms-versus-answers contrast survives on
  sm_120.
- **Verdict, cross-stack (measured, not assumed):** the L40S's cu128 records
  equal its cu124 corpus records for every arm tried — ordered gemm 186/186,
  ordered train 1,283/1,283, floor150 rank1 payload equal, vendor cublas
  186/186 — and the full-set outcomes file is byte-identical to the Windows
  cu124 one. **Cross-stack comparability remains a per-jump measurement, not
  a property:** it held for 2.6.0+cu124 → 2.7.1+cu128; the next toolchain
  starts unproven. Full narrative: personal wiki,
  `a-loss-agrees-where-the-computation-does-not` footnote 22 (`e2be2a3`).
- **What the zeros exclude — power audit, board `ab5b9882`, 2026-09-08.**
  Every verdict above is a ZERO-FAILURE identity claim, so neither a
  t-based MDE nor McNemar applies: there is no spread to divide by, and
  feeding `sd=0` into `t_crit*sd/sqrt(n)` returns 0 and reads as perfect
  power. The instrument is the exact one-sided Clopper-Pearson bound
  `1 - 0.05^(1/n)` — the largest per-comparison divergence rate that could
  still have produced zero observed divergences:

  | claim | comparisons | 95% UB | expected divergences in a 2,627-item run |
  |---|---|---|---|
  | ordered train_step, cross-card | 1,283 | 0.233% | 6.1 |
  | full-set outcomes digest, cross-card | 2,627 | 0.114% | 3.0 |
  | attention stages, cross-card | 140 | 2.117% | 55.6 |
  | ordered gemm, cross-card | 93 | 3.170% | 83.3 |
  | pooled over the four | 4,143 | 0.072% | 1.9 |

  The digest row is the TIGHTEST, not the weakest: a sha256 over 2,627
  items is a conjunction, so one differing item breaks it and it evidences
  2,627 agreements rather than one.
  **Stated threshold of practical interest: one divergence in a full
  2,627-item scoring run, 0.0381%** — the scale this project operates at.
  Against it every row is **NOT TESTED**, pooled included: the data are
  consistent with a rate that would put a few divergent items in a full
  run. Against a coarser 1% bar the two large rows are TESTED and the gemm
  and attention rows are not. The verdicts are not wrong; what they
  exclude is narrower than "bit-identical" reads.
- **`186` and `93` count different UNITS of the same battery, and both are
  correct — settled by a cluster read 2026-09-09.** This entry says
  cross-card `gemm 186/186`; the personal-wiki narrative says the cards
  "agree on all 93 ordered ones". Both files hold **186 observations over
  93 distinct shapes**, each shape carrying two summaries of one computed
  matrix, `<shape>|digest48` and `<shape>|sum`; the observation names are
  identical across the two cards, so the pairing is exact and total. So
  186/186 is observation-level agreement and 93/93 is shape-level
  agreement — one battery, two units, no conflict. Read directly from
  `/pub/wagnera3/gemm/cu128-v1/{rtx6000,l40s}-ordered.json` by
  @opus-qft-code-0903, who had the cluster connection this audit did not.
  **The bound above uses 93, and non-independence is the reason, not
  caution.** `digest48` and `sum` are two views of the SAME matrix: a
  kernel that changes a result moves both, and one that does not moves
  neither. They cannot diverge separately, so counting 186 independent
  trials would inflate the denominator with duplicate evidence and halve
  a bound the data does not support halving (1.598% against the correct
  3.170%). **Whenever either number is quoted, name its unit** — "186/186
  observations across 93 shapes", "93 shapes (186 observations)" —
  because two accurate totals over one battery, stated without their
  units, read as a disagreement to everyone who did not run them.
- **The run-level axis is n=1, and it is a different question.**
  Bit-identity across 2,627 items says nothing about whether a FRESH pair
  of runs reproduces it — that is the axis determinism pins address, and
  it was observed once per arm, so its 95% bound is 95% and vacuous. The
  cross-card question here is well evidenced; the repeat question is not
  evidenced at all, and pinned determinism is not a substitute for
  measuring it.
- **The cross-boundary term is MEASURED, and large where the owned kernels
  are not used.** Same card pair, same pins, same battery, same hour:
  vendor cuBLAS gemm digests agree on 18 of 93, so **80.6% diverge**. The
  ordered/owned arms' bit-identity is a property of THOSE KERNELS, not of
  the card pair, and transfers to no arm calling vendor cuBLAS. Any future
  comparison crossing this boundary on a vendor arm must carry that term;
  a p-value from replicate spread cannot see it.
- **Two toolchain defects banked, fixed at root:** torch 2.7's SDPA
  eligibility APIs initialise CUDA even for a CPU-device probe, fatal on
  driverless build nodes (`d6363b9b`); and `ordered_kernels`' gemm/bench
  CLIs read shape-table hooks from the module `5bea978c` had moved them out
  of — a cluster-only crash the tests missed because faking a nonexistent
  attribute silently creates it (`8fdeca63`). The pinned image bridges the
  old hook name via `run_ordered_gemm_probe.py` beside the gemm artifacts.

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
- **Power:** audited 2026-09-09, board `1e4ab572`. The benchmark family's
  verdicts are stated as per-seed WIN COUNTS over five seeds, which is a sign
  test whose best attainable two-sided p is 0.0625 — **no outcome rejects at
  0.05**, so "leads" and "ties" are the same verdict on that instrument. The
  manifests carry per-seed `r_squared` and `auc_roc`, so the paired instrument
  runs on data already on disk, and it splits the standing scoreboard:
  `weather_tmax` and `metab_confidence` are genuine adequately-powered nulls;
  `rw_value`'s claimed ClearGBM lead is a powered NULL at one point of R²;
  `voc_match_quality`, `financial_distress` and the `us_binary` head-to-head
  are NOT TESTED. **42 additional seed-runs make the whole board conclusive
  and 13 of them cover three of the four corpora** — counts from
  `required_replicates` in `platform_core.minimum_detectable_effect`.
  **The root cause is a timing constant that crossed into quality work:**
  `DEFAULT_SEEDS = (42, 43, 44)` in `covenant_ml`'s `benchmarking/factory.py`
  is documented as reproducing "the workload the ClearGBM PERFORMANCE work is
  tuned against", beside `DEFAULT_REPEATS` and `DEFAULT_WARMUPS` — wall-clock
  knobs. Three seeds stabilises a timing median; inherited by a QUALITY arm it
  instead sets what can be concluded, and it is the bare `MIN_REPLICATES` floor
  the power module accepts. It is an overridable CLI default rather than a
  value stranded in each plan, and **the override was used and did not help**:
  the p6 and binning runs all passed seeds 42–46 and still reach four NOT
  TESTED verdicts out of six, because `--seeds` takes a count and not an effect
  size. That is the argument for deriving the count instead of picking it. **A separate provenance gap sits under all of it:
  none of the ten p6/binning quality manifests carries a `fingerprint` — they
  predate the 2026-08-27 entry-point pin, so the numbers the standing rests on
  were produced by nothing recorded.**
  Two claims cannot be tested from disk at all because only means and
  across-fold sds were published: the rw_matches CV ties, and
  `scale_pos_weight`'s "+1.3 AUC points". The fix for both is one column of
  per-fold numbers. Details in the `Power:` sections of
  `BENCHMARK_RESULTS_2026-08-25_count_aware_binning.md`,
  `BENCHMARK_RESULTS_2026-08-24_p6_farm_and_rw_value.md`,
  `BENCHMARK_RESULTS_2026-08-22_scale_pos_weight.md` and
  `BENCHMARK_RESULTS_2026-08-22_knob_closure.md`.
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
  `results/*.csv` plus a `RunRecord` sidecar per evaluation, and as of
  2026-09-03 a second sidecar per *training* run beside each checkpoint
- **Compares:** `zero_shot_excess_ce_*.csv` carries `excess_cross_entropy` —
  one model's cross-entropy minus another's — with confidence intervals,
  across seven languages and nine arms (`pilot_a/b/c`, `variant_b`, `v3`,
  `2026-02`, `rebuild_2026-08`, and as of 2026-09-04 `v5` and `v6`). Files
  named `_forMoldir` and a commit crediting a Finnish native reviewer indicate
  this is bound for publication.

  **Arms before v4 are not subtractable from arms after it.** v4 corrected
  ‹щ› from ɕː to ʃː in Kazakh, Kyrgyz and Uzbek-Cyrillic; the segment appears
  in no evaluation snippet, so the error reached those models as readers and
  not as targets, moving Kyrgyz-as-reader by 0.22 while every other language
  stayed inside ±0.02. It was carrying about two thirds of one of the draft's
  reported asymmetries. Measured and cited on the personal wiki at
  `transcription-error-inflated-a-directional-asymmetry`.

  **`v5` and `v6` also carry a second output**, `<results>_asymmetry.csv`, 21
  unordered pairs against the CSV's 49 ordered ones. It bootstraps the
  DIFFERENCE between the two directions directly rather than asking whether
  two separately-bootstrapped intervals overlap. That matters: non-overlapping
  intervals imply a difference, overlapping ones imply nothing, and the
  az↔tr asymmetry that the overlap test called lost at v6 survives the
  difference test at `+0.2253 [+0.0629, +0.3918]`.
- **Eight languages as of 2026-09-03, seven of them scored.** Russian was
  added as a second non-Turkic control — Finnish is the agglutinative control,
  Russian the contact language the Cyrillic corpora borrow from. It is a base
  (`best_val_loss` 1.1579525100506294, vocabulary 31, three epochs) and not a matrix member,
  because no human intelligibility ratings were collected for it, so it has
  neither a row nor a column to score against. `char_lstm.corpora` splits the
  two sets explicitly as `LANGS` (8) and `PERCEPTION_LANGS` (7).
- **Provenance:** a `RunRecord` sidecar as of 2026-08-28. Every
  `zero_shot_eval` run writes `<results>.csv.runrecord.json` beside its CSV:
  experiment `turkic-zero-shot-excess-ce`, the OOV regime as the label, one
  named observation per ordered language pair, a SHA-256 of the CSV as the
  payload digest, and a `RunFingerprint` carrying the host and the resolved
  `torch`/`numpy` versions. It states the card and driver as absent because
  the scoring path genuinely uses neither, and the determinism stack as
  `none` because it pins nothing — both true.

  **Training writes one too, as of 2026-09-03** (`char_lstm.training_record`,
  experiment `turkic-char-lstm-base-training`). Every completed run writes
  `<lang>_best.pt.runrecord.json` beside the checkpoint, labelled with the
  corpus rather than the language: the generation directory plus the first 12
  hex of the corpus SHA-256. The digest is there because five of the seven v4
  corpora are byte-identical to v3 and two are not, so the directory name
  alone would report a corpus change that did not happen and miss one that
  did. Unlike the scoring fingerprint this one carries the card and the
  driver, because training uses both — reusing the scoring fingerprint would
  have recorded something false, which is worse than recording nothing.

  Its determinism stack also reads `none`, and there that is a statement
  about configuration, not about outcome. Measured 2026-09-03: the `tr` base
  trained twice from seed 1234 on one RTX 3090 Ti ten hours apart, no flags
  set, produced byte-for-byte identical 3,736,656-byte checkpoints and a
  `best_val_loss` agreeing to every digit. It still records `none`, because
  reproducing once is not the same as having asked for reproducibility. And
  it does not generalise: GPT-2 on this same card and torch build diverges
  from its own seed, so this is a fact about a 933,535-parameter model, not
  about CUDA.

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
  | `rebuild_2026-08/corpora_clean_v3/` | 11,658,775 | Uzbek | superseded, and wrong |
  | `rebuild_2026-09/corpora_clean_v4/` | 11,658,775 | Uzbek | seven languages |
  | `rebuild_2026-09/corpora_clean_v5/` | 11,658,775 | Uzbek | eight, mixed provenance |
  | `rebuild_2026-09/corpora_clean_v6/` | **11,658,775** | **Uzbek** | **current** |

  **The last three rows are from 2026-09-03 and 2026-09-04**, and the budget
  has not moved across any of them, which is why the paper's corpus sentence
  survives all four generations. `LSTM/CORPORA.md` carries the per-generation
  detail; the three facts that belong in an index are these.

  **v3 is not merely superseded, it is wrong**, for the ‹щ› reason recorded
  above. Anything computed on it can be compared with the earlier arms and
  with nothing after it.

  **v6 is the first generation where every language's manifest pins its own
  `output_sha256`** and names the normaliser that produced it — a
  `NormalizationRecord` carrying the Unicode version, the format-character
  category stripped, and a digest of the fold table. Before that, a corpus
  could be re-cleaned under a changed normaliser with nothing recording it.

  **Five of the eight v6 corpora came out byte-identical to v5 through a
  fresh download**, which is the evidence that the pipeline reproduces end to
  end. The three that did not — fi, tr, kk — are archives that no longer
  reproduce from their own recorded parameters: 0, 1 and 10 lines of 10,000
  match a fresh pull. A line count cannot see that, because all three have
  exactly the 10,000 lines their manifests claim, and a line count is how
  this was first (wrongly) concluded to affect kk alone. Only those three
  were retrained; the other five checkpoints are reused, which is sound only
  because training here reproduces bitwise (below).

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

  **What the one pair excludes — power audit, board `1e4ab572`,
  2026-09-09.** This is a ZERO-FAILURE claim, so no spread exists to
  divide by and the t-based MDE does not apply; the instrument is the
  exact one-sided Clopper-Pearson bound `1 - 0.05^(1/n)`. The answer
  depends entirely on which axis `n` counts, and the two readings differ
  by three orders of magnitude:

  | axis | n | 95% UB | the question it answers |
  |---|---|---|---|
  | per-ELEMENT, inside the one pair | 9,003 | 0.033% | would a divergence in any single world element have been seen |
  | per-NODE-PAIR | 1 | 95.0% | would a divergence between two DIFFERENT nodes have been seen |

  `world.json` carries 8,996 mines plus tanks, containers and ferries, so
  the digest is a conjunction over ~9,000 elements and is a very sensitive
  detector — the same reason `mi-cu128`'s digest row is its tightest, not
  its weakest. **But the elements are not independent replicates of the
  thing this flag claims.** A node-specific divergence mechanism — CPU
  stepping, libm, image contents — moves many elements together or none,
  so 9,003 elements is one thorough observation of ONE pair, not 9,003
  trials of the pair-level question. `deterministic: true` licenses
  subtracting measurements ACROSS NODES, so it rides on the pair axis.

  **Stated threshold of practical interest: one divergent pair in twelve,
  8.3%** — `rusted`'s panel width, the smallest cross-node panel anyone on
  this machine has actually run. Against it this claim is **NOT TESTED**,
  by a factor of eleven. The prose above is already honest that one pair
  cannot see an intermittent divergence; 95% is what that sentence is
  worth as a number.

  Three boundaries were crossed simultaneously — node, image, and the code
  change between the jobs — so the zero is a joint result over all three
  and **no single term is separately measured.** A future panel should
  vary one axis at a time, or it will inherit the same confound at greater
  cost.

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
- **Power, beyond the determinism flag:** the same 2026-09-09 audit covered
  this project's other zero-failure claims, all recorded on the client wiki
  rather than restated here. `make audit`'s gate (`_passed`,
  `validate/audit.py:141`) tests a RATE with no floor on `n`, so a perfect
  record needs n ≥ 19 to clear its 0.85 floor: `dual-hit` 6/6 and `missile`
  6/6 pass at 100% carrying no evidence they exceed it, and `walk` 204/232
  sits too close to separate — six of the nine claims ARE tested, some
  overwhelmingly. The larder probe's `3/3` bounds its refusal rate at 63.2%
  and the 3 is a default argument rather than a stopping rule. The
  divergence soaks are one scripted scenario each, so their per-scenario
  bound is 95% and their weight comes from the negative control that proves
  the detector fires, not from the round count. Pages:
  `clients/TankpitBot/wiki/pages/physics-module-roadmap.md` and
  `larder-plan.md`.
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

### `code-style` — QLoRA on this monorepo, scored by this monorepo's own guards

- **Repo:** this one. Corpus emitter `tools/code-corpus`, training and
  generation through `services/Model-Trainer`, scoring through
  `tools/code-style-eval`.
- **Entry points:** `code_style_eval.cli.compare`, the console script
  `code-style-eval-compare`, which emits `comparison.json` and its
  `RunRecord`. Added 2026-09-09 by the `research-registration` guard rule.
  This entry already described the ARTIFACT and the PACKAGE and named no
  command, so a mechanical reader found nothing while a human reader found
  everything — and the rule's first version passed this module anyway,
  because it matched the bare stem `compare` against the word "compare" used
  in prose two paragraphs down. Both halves of that are why the names here
  are qualified.
- **Runs:** `runs/code-style-run-train.json`, then
  `runs/code-style-run-gen-base.json` and `-candidate.json`. The order is not
  enforced by a dependency: the generation arms load the adapter the training
  run saved, so submitting one before the other finishes fails on a missing
  metadata file rather than on anything subtler.
- **Both commands read a committed document** rather than a configuration's
  worth of flags — `runs/code-style-qlora-v1.json` for the training payload
  and `runs/code-style-gen-v1-{base,candidate}.json` for the two arms — so
  the arms provably differ in one field. Thirteen flags would be thirteen
  chances to type one differently between two arms that must differ in
  exactly one thing, which is the same argument `modeltrainer-cluster-train`
  already makes for its payload.
- **Scoring stays off the cluster, and that is a finding rather than a gap.**
  The instrument IS this repository's checkers — `ruff`, `mypy` and
  `monorepo_guards` scoped per item through `scripts/guard.py --root` — so
  running it inside an image would measure that image's copy of the rules
  rather than the repo's. The GPU half is what the cluster is for; the CPU
  half is what the repository is for, and they meet at a directory of files
  and a manifest whose shape is `platform_core.continuation_task`.
- **Produces:** `code-corpus-v1.jsonl` plus its holdout; a QLoRA adapter
  (NF4 storage, bfloat16 compute, double quantization); per-arm directories of
  generated files with a manifest recording whether each completion terminated
  or hit the token budget; per-arm outcome JSONL, one row per item per checker;
  `comparison.json` with a paired 2x2 table and McNemar mid-p; and
  `perplexity.json`.
- **Compares:** the QLoRA adapter against **its own base** — the same weights
  under the same NF4 quantization, with nothing attached — on the same
  held-out files, two ways: token-level perplexity masked to the continuation,
  and guard-pass rate. Per-item outcomes throughout, so the contrast is paired
  rather than two rates subtracted. The control is deliberately NOT
  `load_prepared_hf_lm_from_hub`, which loads unquantized on purpose;
  comparing an NF4 adapter against bfloat16 weights would measure two changes
  at once.
- **Provenance:** a `RunRecord` from each of the three stages as of
  2026-09-03. Generation's carries the decoding parameters' effect through a
  digest over which items finished, and reads its package axis from the
  ARTIFACT's metadata rather than from the model that loaded — so both arms
  name the same libraries, including `peft`, which the base arm does not
  execute but which decides what the candidate is. Scoring's still has three
  empty fingerprint axes BY CONSTRUCTION, and says so: it is CPU work outside
  any image and pins no determinism.
- **What it does NOT carry, and this is the honest half.** The corpus was
  emitted while both source repositories were dirty, which the manifest flags;
  no run built on it is citable until it is re-emitted clean. And **every
  number this project has produced so far was produced locally, before
  registration.** Registration makes the next run reproducible; it does not
  reach backwards.
- **A preemption costs the whole run, and the workspace now says so.** This
  project declared `checkpoint_steps: 500`, copied from `mi`. Nothing honours
  it: `HPC3_CHECKPOINT_STEPS` is exported by the sbatch wrapper and read by no
  payload in this monorepo, and Model-Trainer checkpoints at EPOCH boundaries
  only — "every other completed epoch publishes the rolling checkpoint so an
  interruption costs at most one epoch". This payload declares
  `num_epochs: 1`, so one epoch is the whole run and the only checkpoint is
  the one written after training finishes. The declaration is now `0`, which
  the contract defines as "none".
  What protects the run is the other half of the same rule: `deterministic`
  replay, where "requeue alone IS protection … the whole run is a checkpoint
  at step zero". On `free-gpu`, whose `PreemptMode` is `CANCEL`, `--requeue`
  is inert, so in practice the protection is that a preempted run is
  resubmitted by hand and replays. `mi` carries the same unhonoured `500`
  against the same trainer.
- **Results as of 2026-09-02, stated with their limits.** Perplexity moved
  2.8327 to 1.9631 with 392 of 392 held-out items improving, and train/holdout
  overlap was checked to be zero by path AND by content — the content check
  is the one that matters here, because `scripts/guard.py` is byte-identical
  across all 41 packages. Guard-pass showed NO detectable difference across
  three sweeps (mid-p 0.84 on the last), and the combined rate sits near 2%,
  which is a floor where the metric has almost no power to move.
  **NOT TESTED, and for two strata not testable** — the counts in the
  previous sentence were crossed between strata, which mattered because
  the discordant count is exactly what decides falsifiability. Corrected
  against the four `comparison.json` files 2026-09-08: the 226-item
  all-scored stratum gave **6** discordant pairs (3v3, mid-p 0.844); the
  **5** discordant pairs and the 5.6% rate behind the power figure of 0.21
  belong to the 90-item finished-in-both-arms stratum, not to the 226.
  McNemar conditions on the *d* discordant pairs, so the smallest p a
  stratum can produce is `2*0.5^d` at a *d*:0 split — 0.0625 at *d*=5,
  above α=0.05. **The finished (n=90, d=5) and clean-import (n=49, d=5)
  strata could not have rejected under any outcome**, and their reported
  `exact_p` of 1.0 could not have been anything else. Against a stated
  threshold of +5 pp absolute guard-pass, all three strata are NOT TESTED;
  the classification is insensitive to that threshold, since every
  stratum's minimum detectable effect already equals or exceeds its own
  base pass rate. Roughly 800 items reach power 0.73. Full derivation and
  the strata table: `wiki/pages/code-style-guard-pass-instrument-limits.md`. The first two sweeps were void for reasons recorded on
  the board: a token budget that truncated 83% of completions, and before that
  an unscoped guard invocation that gave every item the same verdict.
- **Not novel, and the task spec that says otherwise is wrong.** A systematic
  search found the core already published: ContextCov (arXiv 2603.00822)
  compiles a repository's written conventions into executable AST and
  architectural checks, and per-repository LoRA is the upper-bound baseline
  in Code2LoRA (arXiv 2606.06492). The papers are on the personal wiki under
  `computational-linguistics`. The defensible claim is fitness for purpose —
  no public benchmark scores THIS repo's conventions — not originality.
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
- **A second claim was wrong for a week.** This entry said the pipeline
  existed end to end. Training and generation were scratchpad scripts with
  absolute Windows paths compiled into them, and the generator had
  hand-rolled a model loader `load_prepared_hf_lm_from_handle` had already
  provided for months. Both are closed as of `5bea978c`; the training half
  needed no new code at all, only a payload document.

---

## Not registered anywhere

Real research, producing numbers that get compared, reachable by no tool.

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

**A project is registered once it is reproducible, not before.** Registration
requires an image digest, and producing one is most of the work — so the first
four steps happen before anything is written down here.

```bash
# 0. The first environment. NOT `module load python`: the cluster's python
#    modules are 2.7/3.8/3.10/3.14 and everything here needs 3.11, which lives
#    behind miniconda3. Bootstrap refuses to hand back an environment whose
#    interpreter belongs to another project.
hpc3-bootstrap --config runs/hpc3-<name>.json --project <name>     --env-path /pub/wagnera3/envs/<name> --python 3.11

# 1-3. Turn that environment into a pinned image, and get its digest.
hpc3-image-capture --config … --env-path /pub/wagnera3/envs/<name> --out specs/<name>-image.json
hpc3-image --spec specs/<name>-image.json --out-dir … --image-name <name>
#    scp the rendered files AND the first-party wheels into the image directory
hpc3-image-build --config … --project <name> --name … --image-dir … --image-name <name>
```

**STEPS 0-3 ASSUME THE PROJECT NEEDS ITS OWN IMAGE, AND MOST DO NOT.** A
project whose workload is an existing package's CLI needs that package's
image REBUILT from a newer commit, not a new image. `code-style` was
registered on 2026-09-03 by rebuilding `specs/abl-image.json` -- the image
`mi` already uses -- against fresh first-party wheels, and it declares the
result. The recurring job is a version bump; the four-step flow above is the
first-image case, which happens once per image family and not once per
project.

For a version bump: build the five wheels, edit `git_commit` in the spec
(which is what every version bump has actually done -- see
`tools/hpc3/wiki/pages/capture-source-drift.md`), render, stage, build.
`hpc3-image-capture` will NOT help: it probes the image, so re-running it
reproduces the environment the last image sealed rather than the repository's
current state.

**A rename in any package that spec names is a change to the image recipe.**
`required_symbols` and `smoke_commands` cite Python module paths, and
`tools/hpc3/tests/test_committed_specs.py` re-checks them; run `make check`
in `tools/hpc3` after moving anything across a module boundary, rather than
learning it from a `%post` failure twenty-five minutes into a build.

Then, and only then:

4. **Add the `projects` entry** to `tools/hpc3/runs/hpc3-<name>.json` —
   resources, the built image's path and `sha256`, `env_path` (the in-image
   prefix, normally `/opt/env`), `pinned_packages`, `budget` and `repo`. The
   filename is not free: `test_committed_runs.py` requires
   `hpc3-<name>.json` to declare exactly the project `<name>`, and requires
   that no project is declared by two workspaces.
5. **Add a section here.** `test_committed_runs.py` fails if a registered
   project's name does not appear in this file. This is the one step nothing
   can generate — it is where you say what the project measures and what its
   provenance does not cover.
6. **`hpc3-research-index --write`** to regenerate the table above. The
   committed block is checked, so a stale one fails.
7. **Emit `RunRecord`s** from whatever produces the numbers.
8. **Submit through the hpc3 CLI** rather than a hand-written `sbatch` script,
   so the run lands in the ledger and `hpc3-trace` can answer "which job
   produced this artifact".

**What this list used to omit, and what it cost.** Steps 0–3 were absent
entirely, so the first environment was improvised each time — one of them
(`envs/tankpit`) is a venv whose interpreter is a symlink into another
project's environment, which nothing records and which breaks the day that
project is cleaned up. Step 6 was absent, and until 2026-09-03 registration
ALSO meant editing two hardcoded project lists inside
`test_committed_runs.py`; both were met as surprise red tests rather than as
steps. Those lists are now derived invariants, so a seventh project needs no
test edit at all.
