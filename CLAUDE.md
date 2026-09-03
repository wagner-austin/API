# API monorepo — session rules

Loaded automatically into every session under `~/PROJECTS/API/`. The global
`~/.claude/CLAUDE.md` rules apply on top. This file routes; it does not
duplicate. Where it names a canonical document, read that document rather
than trusting a summary here.

Typed Python monorepo: ML training, NLP, media services, quant-ML risk
modeling. Strict mypy (no `Any`, no `cast`, no `type: ignore`), 100%
statement + branch coverage, FastAPI + RQ + Redis + Kafka.

## Research: read the index before producing a number

**`docs/RESEARCH.md` is the canonical index of every body of work on this
machine that produces numbers someone compares** — including the two that
live outside this repository. Read it before auditing, extending, or
reproducing any experiment. It states, per project, what provenance the
project carries and what it does not; the honest gaps are the point of the
file, not an oversight in it.

It is enforced, not decorative: the machine-readable half is the `projects`
table in the hpc3 workspace documents (`tools/hpc3/runs/hpc3*.json`), and
`tools/hpc3/tests/test_committed_runs.py` fails when a registered project is
missing from `RESEARCH.md`. Registered today: `mi` (Model-Trainer probes),
`cleargbm`, `floor` (cloze floor scoring), `turkic-lstm` (`~/PROJECTS/LSTM`),
`rusted` (RustedWarfareBot). `sirius` is deliberately NOT registered and the
file explains why — do not "fix" it by registering it.

**Adding a research project** — the procedure is at the bottom of
`RESEARCH.md`. In one line: register it in a `tools/hpc3/runs/` workspace
document, add its section to `RESEARCH.md`, emit `RunRecord`s from whatever
produces the numbers, and submit through the hpc3 CLI so the run lands in the
ledger and `hpc3-trace` can answer "which job produced this artifact".

**`platform_core.run_record.RunRecord` is the one shape a research run
emits**, and `platform_core.comparability` decides whether two of them may be
subtracted. Do not invent a second record shape — that is what made
`covenant_ml`'s numbers unreadable beside another experiment's until it
adopted the shared one.

## Which wiki a finding goes in

Five wikis carry this machine's ML/HPC knowledge and they are **not**
interchangeable. Route by what the claim is evidence *about*, not by what
project you happen to be sitting in. All five are searchable through the
corvis MCP tools (`wiki_search_query`, `wiki_search_list_wikis`).

| slug | tree | takes |
|---|---|---|
| `tech` | `~/PROJECTS/tech-wiki` | Third-party library and spec internals, read from primary source. LightGBM's histogram path, GPT-2's transposed `c_proj`, MEMIT's residual spread. Contract: `html-sha256` — every claim pinned to a sha256'd capture under `sources/`. |
| `personal` | `~/PROJECTS/wiki` | The published literature and this wiki's own measurements. Papers get a page with an archived PDF; so do determinism measurements taken on this stack. Contract: `pdf-corvis`. |
| `api-codebase` | `wiki/` (this repo) | What THIS repo's code does and why — the ClearGBM program record, service architecture, cross-service context. Contract: `code-paths`. |
| `hpc3` | `tools/hpc3/wiki/` | Cluster facts and the hpc3 package's design record: partitions, billing, images, staging, triage, budgets. Contract: `code-paths`. |
| `me` | `~/PROJECTS/me-wiki` | Recruiter-facing narrative only. Never the place a technical finding lands first. Contract: `external-urls`. |

The clients keep their own `code-paths` wikis under `clients/<name>/wiki/`
(`tankpitbot`, `navprobe`, `rustedwarfarebot`) — that is why the `Clients`
hub in `wiki/index.md` stays thin by design.

**The boundary that gets confused:** a fact about LightGBM's C++ goes to
`tech`; a fact about ClearGBM's reimplementation of it goes to `api-codebase`;
the paper that introduced the algorithm goes to `personal`; how the benchmark
was scheduled goes to `hpc3`. One finding can legitimately produce pages in
several — link them, do not restate them.

**Full routing procedure, including how to drive an existing research project
rather than start a parallel one: the `/ml-research` skill.**

## Running work on the cluster

`tools/hpc3` is the only supported path to HPC3. Do not hand-write an
`sbatch` script — a hand-written job lands in no ledger, and `hpc3-trace`
cannot answer which job produced an artifact. The package encodes what the
cluster cost to learn as rules that run; a job that would break one cannot be
constructed.

```
hpc3-preflight   would it start?
hpc3-submit      start it
hpc3-watch       what is it doing, what did it cost
hpc3-triage      is anything wrong that looks fine?
hpc3-sweep       many members, one sbatch call
hpc3-chain       stages, each after the last
hpc3-trace       which job trained this?
hpc3-cancel      stop it, and say what actually stopped
```

`tools/hpc3/README.md` is the command reference. **The reasoning — every
rule's incident, every measured fact's measurement — is in
`tools/hpc3/wiki/`, and new incident narrative goes THERE, not into the
README** (the README reached 1,045 lines absorbing it, which is why the wiki
exists).

## Documentation goes in a wiki, not in a new `docs/` file

The repo carries loose planning markdown that predates the wikis —
`docs/covenant-radar-plan.md`, `docs/model-trainer-refactor.md`,
`docs/char_lstm_integration_plan.md`, `services/Model-Trainer/docs/*.md`.
They are history. **Do not add to that pattern.** A durable finding goes to
the wiki page its contract fits; a plan you are executing goes to the agent
task board (`task_*` tools); a command reference goes to the package README.

`docs/RESEARCH.md` is the standing exception — it is an enforced index, not a
plan doc.

## Wiki page discipline (both `code-paths` wikis in this repo)

Enforced by `packages/wiki-check` in the MCPs workspace via `wiki_audit_run` /
`wiki_audit_page`. Six rules are specific to this contract, five of them fatal:
`source-path-exists`, `source-path-line-anchor`, `git-blob-hash-pin`,
`claude-md-anchor-exists`, `memory-file-exists` (errors) and
`code-citation-symbol-at-line` (warning). ~40 universal rules run too.

- **Every page declares `source_paths:`** — repo paths relative to that
  wiki's `workspaceRoot`, which must resolve at audit time.
- **Every `source_paths:` entry gets a `source_git_blobs:` pin.**
  `source-path-exists` only proves the path still resolves, which stays true
  across a total rewrite of the file. `git-blob-hash-pin` is the only rule
  that catches drift, and it fires ONLY on pinned paths. An unpinned citation
  is a claim nothing will ever re-check.
- **Evidence that is not a repo path goes in `provenance:`**, not in
  `source_paths:` — probe job ids, `sshare` readings, cluster paths under
  `/pub`, and citations that genuinely point outside the wiki's
  `workspaceRoot`.
- **YAML must parse.** `related: [[a]], [[b]]` is a syntax error; write
  `related: ["[[a]]", "[[b]]"]`. All 21 hpc3 pages carried the broken form
  until 2026-09-02 and nothing caught it, because the wiki was unregistered
  and therefore never parsed by anything.
- **Bump the page count in `index.md`** when adding a page; the
  `enumeration-count` rule checks it.

## Conventions

- **Per-package `make check`** (`check: lint | test`) — run it in the package
  you touched. Read the OUTPUT, not the exit code, unless you appended
  `; exit $LASTEXITCODE`: `powershell -Command "& { make check }"` exits 0
  even when make fails.
- **`libs/monorepo_guards`** is the static-analysis framework — 34 rules,
  Python + Rust, configured centrally in `monorepo-guards.toml`. Guards apply
  to `src`, `scripts`, `tests`. `forbid_pyi = true`; `allow_print_in_tests =
  false`.
- **No `dataclass` under any `src/`** — use `TypedDict`, especially for
  anything deserialized from an external source. Enforced by
  `dataclass_ban_segments`.
- **Strict typing**: no `Any`, no `cast`, no `type: ignore`, no `.pyi`.
- **100% statement + branch coverage.** A task is not done until green
  `make check` output is in the conversation.
- Fleet: `make infra`, then `make up-<service>`. `make status`, `make logs`.
  Service port map is in `README.md` and `wiki/pages/service-port-map.md`.
