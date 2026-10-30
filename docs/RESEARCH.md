# Research index

**Every body of work on this machine that produces numbers someone compares.**
This is the registration directory. Read the linked wiki pages for methods,
results, limitations and corrections; use package READMEs for commands.

Start with the [API wiki](../wiki/index.md) or the
[HPC3 wiki](../tools/hpc3/wiki/index.md). The
[WRDS research entry](../wiki/pages/wrds-equity-research-protocol.md)
connects study design, extraction, admission, queueing, evaluation and readiness.

Workspace documents (`tools/hpc3/runs/hpc3*.json`) own project resources and
images. `hpc3-research-index --check` verifies the generated table and counts
tracked artifacts against the reviewed markers below. A marker records a
review of that many files, not a successful experiment or a live job count.
The `research-registration` guard requires qualified CLI names for entry points
that produce `RunRecord`s. Registration does not establish statistical power.

Historical investigation notes are retained at
[API 4a145beef](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md).
Each project links its exact range. These are dated evidence, including
retractions and missing artifacts, rather than a second maintained index.
The focused wiki records own subsequent corrections. The snapshot is also
available offline with `git show 4a145beef:docs/RESEARCH.md`.

## Registered with the hpc3 CLI

Submit through `hpc3-submit`, `hpc3-sweep` or `hpc3-chain`. The machine-local,
untracked `tools/hpc3/runs/ledger.jsonl` records submissions; query the live
controller for job state. The table below describes registered defaults,
not available capacity, installed commands or a completed run.

<!-- generated: hpc3-projects. Do not edit by hand. -->

Rendered from `tools/hpc3/runs/hpc3*.json`. Regenerate with `hpc3-research-index --write`.

| project | partition | gpu | cpus | mem GiB | minutes | image | deterministic | resumes |
|---|---|---|---|---|---|---|---|---|
| `cartridge-qa` | free-gpu | `A100` x1 | 8 | 64 | 720 | `890e444560d4` | yes | yes |
| `cleargbm` | free | cpu | 4 | 16 | 60 | `0a525f532a9e` | yes | no |
| `code-style` | free-gpu | `A100` x1 | 8 | 32 | 240 | `5dfd78a7eb14` | yes | no |
| `floor` | free-gpu | `A100` x1 | 8 | 32 | 60 | `df841c661b9e` | yes | no |
| `mi` | free-gpu | `A100` x1 | 8 | 64 | 240 | `55651342e15d` | yes | no |
| `mi-cu128` | free-gpu32 | `RTX6000` x1 | 8 | 64 | 55 | `6d9ba0baac40` | yes | no |
| `rusted` | free | cpu | 4 | 2 | 180 | `97a80bdeb16d` | yes | no |
| `tankpit` | free | cpu | 2 | 2 | 60 | `0cfdd5592a1a` | yes | no |
| `turkic-lstm` | free-gpu | `A100` x1 | 4 | 16 | 150 | `6e034383e300` | no | yes |

<!-- /generated: hpc3-projects -->

### `mi` — Model-Trainer probes and benchmarks

<!-- reviewed: tools/hpc3/runs/extraction-ablation-e2be2a3-scores/*.json = 44 -->
<!-- reviewed: tools/hpc3/artifacts/gemm-*/*.json = 58 -->
<!-- reviewed: tools/hpc3/artifacts/train-*/*.json = 30 -->
<!-- reviewed: tools/hpc3/artifacts/trace-*/*.json = 12 -->

Code: [Model-Trainer](../services/Model-Trainer/README.md).
Runs:

  `model_trainer.cli.{cartridge_base_lora_sweep, cartridge_benchmark,
  cartridge_companion_sweep, cartridge_composition_sweep,
  cartridge_content_lora_sweep, cartridge_diverse_companion_sweep,
  cartridge_headroom, cartridge_solo_grid,
  cartridge_solo_seeds, cartridge_trait_repair_sweep, cartridge_trait_sweep,
  cartridge_varied_companion_sweep, continuations,
  forward_benchmark, gemm_benchmark, gemm_probe, known_answer_probe,
  known_answer_registry, legacy_gemm_probe, probe_ladder, probe_trace,
  score_baseline, score_run, sdpa_benchmark, sdpa_probe, train_benchmark,
  train_step_probe, triple_edit_benchmark}`

| Question | Authoritative record |
|---|---|
| Service, cluster and CLI entry points | [Service architecture](../wiki/pages/model-trainer-service-architecture.md) |
| Fingerprints and comparable observations | [RunRecord provenance](../wiki/pages/model-trainer-run-record-provenance.md) |
| Corpus adaptation programme | [Programme directory](../wiki/pages/corpus-attachment-program.md) |
| Naive composition, leakage and controls | [Composition ceiling](../wiki/pages/model-trainer-composition-ceiling.md) |
| Companions, LoRA, scale and the negative 7B result | [Companioned training](../wiki/pages/model-trainer-companioned-training-recipe.md) |
| Trait expression, coherence, pilot and seed requirements | [Trait instrument](../wiki/pages/model-trainer-trait-composition-instrument.md) |
| Numerical uncertainty | [Noise floor](../wiki/pages/model-trainer-noise-floor-is-a-range.md) |
| Image contents and refusal before submission | [Image freshness](../tools/hpc3/wiki/pages/image-freshness.md) |

The registered image does not establish that every command above is installed.
Use the run's frozen image and preflight. Three-seed pilots are not the full
trait grid; null results remain bounded by the recorded detectable effects.
[Historical measurements and corrections, lines 77–1014](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L77-L1014)
include the determinism campaign and the four-scale marker-effect correction.

### `cartridge-qa` — can the model USE what the cartridge carries, and does it beat retrieval

Code: [Model-Trainer](../services/Model-Trainer/README.md).
Runs: `model_trainer.cli.cartridge_qa_benchmark`.

- [Question-set construction](../wiki/pages/model-trainer-cartridge-question-set.md)
- [Controlled axes](../wiki/pages/model-trainer-question-set-axes.md)
- [Retraction and resolution limits](../wiki/pages/a-claim-four-times-below-its-instruments-floor.md)
- [Historical protocols, submission failures and v56 staging](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L1015-L1216)

The 32-item claim that cartridges beat retrieval was withdrawn. Passing the
realised-item gate establishes falsifiability, not power or a successful run.
Window changes compare within-cell gaps because they change the question set.

### `mi-cu128` — the Blackwell determinism baseline

<!-- reviewed: tools/hpc3/runs/*-cu128-*.json = 24 -->

Code: [Ordered kernels](../clients/OrderedKernels/README.md).
Runs: `ordered_kernels.cli.{attn_probe, gemm_probe, score, train_step}`.

- [Determinism posture](../tools/hpc3/wiki/pages/determinism-posture.md)
- [Image provenance](../tools/hpc3/wiki/pages/image-ledger-lessons.md)
- [Historical Blackwell battery and power audit](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L1217-L1405)

The September battery's 186 observations are not 186 independent shapes.
Two of its 23 arms used an untracked bridge outside the frozen image; the
historical record names the bridge digests and later fix. A one-repeat arm
cannot establish repeatability, and shape-level bounds are not node-level bounds.

### `cleargbm` — ClearGBM benchmarks and covenant-radar optimisation

<!-- reviewed: libs/cleargbm/docs/BENCHMARK_MANIFEST_*.json = 41 -->

Code: `libs/cleargbm`, `libs/cleargbm_rs`, `libs/covenant_ml`,
`services/covenant-radar-api`. Runs: `scripts.optimize`,
`scripts.benchmark_cleargbm_*`, `scripts.benchmark_paired_evaluation`,
`scripts.benchmark_feature_ablation`, `scripts.research_proposals`,
`scripts.research_trial`; isolated proposals enter through
`fleet.cli.research_propose`.

- [Optimisation paths and the discarded determinism pin](../wiki/pages/covenant-radar-optuna-optimisation.md)
- [Paired comparisons, ablation and numerical contracts](../wiki/pages/evaluation-numerical-contracts.md)
- [WRDS study and queue entry](../wiki/pages/wrds-equity-research-protocol.md)
- [WRDS baseline readiness](../wiki/pages/wrds-baseline-readiness.md)
- [Historical benchmark power audit and provenance corrections](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L1406-L1530)

The substantive baseline has not run: the measured October 28–29 jobs and
October 30 installed-file recheck concern the older one-security diagnostic.
Evaluator-owned `research_proposals select` and
`research_trial --ledger --final-manifest` enforce one-shot final scoring.
The readiness page distinguishes implemented controls from deployment rehearsal.

Benchmark manifests and shared `RunRecord`s already coexist; do not recreate
the record writer. The older 3,068 optimisation rows explicitly recorded
`fingerprint: null`. The optimiser wiki corrects this index's former claim
that the applied CPU pin also reached its fingerprint.

Market studies use the same evaluation and provenance layer:
[market and forecast research](../libs/covenant_ml/docs/MARKET_RESEARCH.md),
[execution rehearsal](../libs/covenant_ml/docs/EXECUTION_REHEARSAL.md),
[demo preflight](../libs/covenant_ml/docs/KALSHI_DEMO.md) and
[demo order trial](../libs/covenant_ml/docs/KALSHI_DEMO_ORDER.md).
Their [historical acquisition and demo record](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L1531-L1607)
distinguishes pending observations, synthetic rehearsals and demo fills from profit.

### `floor` — cloze floor scoring

<!-- reviewed: tools/hpc3/artifacts/floor*/*.json = 113 -->
<!-- reviewed: tools/hpc3/runs/floor*.json = 100 -->

Code: [Model-Trainer](../services/Model-Trainer/README.md).
Runs: `model_trainer.cli.score_baseline` (`modeltrainer-score-baseline`).

- [One scorer and decision-level digests](../wiki/pages/model-trainer-cloze-scoring-path.md)
- [Known-answer registry](../wiki/pages/model-trainer-known-answer-registry.md)
- [Comparability and experiment identity](../wiki/pages/model-trainer-run-record-provenance.md)
- [Historical card measurements and uncertainty bounds](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L1608-L1762)

Agreement on item decisions does not establish floating-point identity or
universal hardware invariance. The historical per-item and per-card bounds
answer different questions. Compare records under the same experiment name.

### `turkic-lstm` — character-level LSTM for Turkic languages

Code: separate `~/PROJECTS/LSTM` repository, not present in this checkout.
Runs: `char_lstm` training and evaluation, including `zero_shot_eval`.

- [Personal wiki methodology](https://github.com/wagner-austin/wiki/blob/main/pages/cross-entropy-as-mi-proxy-lstm-methodology.md)
- [Single-card reproducibility](https://github.com/wagner-austin/wiki/blob/main/pages/seeded-lstm-training-reproduces-bitwise-on-one-card.md)
- [Historical corpus versions, missing sidecars and listener bootstrap](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L1763-L2002)

Results already on disk have no sidecar where none was emitted; later code
cannot reconstruct one honestly. Corpus/transcription versions define distinct
experiments. Single-card bit identity does not imply cross-card identity.

### `rusted` — RustedWarfareBot, system identification against an obfuscated binary

Code: [RustedWarfareBot](../clients/RustedWarfareBot/README.md).
Runs: `rw_bot.harness.campaign_match`, one job per match, using campaign
records emitted by `scripts.campaign_doc`.

- [Client wiki](../clients/RustedWarfareBot/wiki/index.md)
- [Policy determinism](../clients/RustedWarfareBot/wiki/pages/policy-determinism.md)
- [Power and MDE ledger](../clients/RustedWarfareBot/wiki/pages/power-audit-mde-ledger.md)
- [Historical registration corrections](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L2003-L2066)

`rw_bot.provenance` records each arm. Engine, bundled JVM and assets all
contribute to build identity; a jar digest alone does not identify a game.
Use the generated resource table and live ledger rather than hand-copied counts.

### `tankpit` — TankpitBot, the sim as a measurable opponent

Code: [TankpitBot](../clients/TankpitBot/README.md).
Runs: `tankpit-sim-run`; `tankpit-feature-rows` derives tick rows and a record.

- [Client wiki](../clients/TankpitBot/wiki/index.md)
- [Feature corpus provenance](../clients/TankpitBot/wiki/pages/feature-corpus-provenance.md)
- [World parameterisation](../clients/TankpitBot/wiki/pages/sim-world-parameterization.md)
- [Packaged assets](../clients/TankpitBot/wiki/pages/packaged-data-assets.md)
- [Historical counts, lost run documents and one-pair limits](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L2067-L2310)

The headless sim is registered, not live browser play. Derivation provenance
cannot recover missing build stamps from old events. The single cross-node
pair changed node, image and code together; its 9,003 elements are not 9,003
independent node pairs. The deleted first image prevents replay of that half.

### `code-style` — QLoRA on this monorepo, scored by this monorepo's own guards

<!-- reviewed: tools/code-style-eval/runs/*/comparison.json = 6 -->

Code: [Code-style evaluator](../tools/code-style-eval/README.md).
Runs: `code_style_eval.cli.compare` (`code-style-eval-compare`).

- [Measured termination and conformance](../wiki/pages/code-style-qlora-terminates-worse-conforms-better.md)
- [Historical v1/v2 power analysis and replay limits](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L2311-L2464)

Training, generation and scoring use shared records. Corpus identity and
matching base quantisation matter. The wider v2 comparison does not
retroactively give the v1 five-disagreement experiment adequate power.

## Not registered anywhere

### `sirius` — declared as an example, never run

The example workspace was deliberately never registered. The metabolomics
workflow uses MFAssignR; the historical SIRIUS examples are not completed jobs.
[Scope decision](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L2465-L2494).

## The shared record, and who uses it

`hpc3.cli.execution_inspect` produces a shared record from a completed durable
attempt's verified native execution evidence. It reports pending accounting
without claiming a completed measurement. See the existing
[installed-controller verification](../tools/hpc3/wiki/pages/wrds-installed-controller-verification.md).

Use [RunRecord and RunFingerprint](../wiki/pages/model-trainer-run-record-provenance.md),
implemented in `libs/platform_core`, and each project's existing writer.
The [historical adoption corrections](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L2495-L2562)
withdraw the false claim that LSTM and RustedWarfareBot could not adopt it.
The optimiser's actual recording boundary is documented in its wiki page above.

## Adding a research project

1. Register the project in `tools/hpc3/runs/` through the existing HPC3 CLI.
2. Add its exact project heading and qualified producing commands here.
3. Emit the shared `RunRecord` and fingerprint through the existing writer.
4. Submit through the existing queue and retain the run's artifacts and identity.

Follow the [HPC3 command reference](../tools/hpc3/README.md),
[image build flow](../tools/hpc3/wiki/pages/image-build-flow.md) and
[submission rules](../tools/hpc3/wiki/pages/submission-rules.md).
Regenerate the table with `hpc3-research-index --write`; update reviewed
markers only after reading the changed evidence. Place research detail in its
existing wiki, link it here, and audit its citations. The
[earlier onboarding rationale](https://github.com/wagner-austin/API/blob/4a145beef16d445e9f786da8ebcce9e3d2f5c2c8/docs/RESEARCH.md#L2563-L2647)
remains available as a dated record.
