# API monorepo — session rules

Loaded automatically into every session under `~/PROJECTS/API/`. The global
`~/.claude/CLAUDE.md` rules apply on top. This file routes; it does not
duplicate. Where it names a canonical document, read that document rather
than trusting a summary here.

Typed Python monorepo: ML training, NLP, media services, quant-ML risk
modeling. Strict mypy (no `Any`, no `cast`, no `type: ignore`), 100%
statement + branch coverage, FastAPI + RQ + Redis + Kafka.

## Corvis: the tools this session actually has

`corvis` is Austin's own MCP fleet — ~27 backends, 250+ tools behind a proxy
— and it is configured **globally**, so it is live in this repo even though
its code lives in `~/PROJECTS/MCPs`. Only four discovery tools appear in the
tool list; everything else is reached through them:

```
search_tools(query)       rank the registry — ORDERS results, never proves absence
tool_list(backend?)       the exhaustive inventory — the only way to prove absence
get_tool_schema(name)     the full input schema, before calling
execute_tool(name, args)  invoke anything the two above found
```

**Never conclude a tool does not exist from a weak `search_tools` score** —
that is what `tool_list` is for. What matters most from this repo:

- `wiki_search_query` / `wiki_search_list_wikis` / `wiki_search_reindex` /
  `wiki_search_index_status` — hybrid search (dense HNSW fused with BM25) over
  all registered wikis, `api-codebase` and `hpc3` included. Search before you
  write; the page may exist.
- `wiki_audit_run` / `wiki_audit_page` / `wiki_audit_status` — the
  `packages/wiki-check` auditor that enforces the page discipline below. Run
  `wiki_audit_page` on every page you touch, before committing. `wiki_audit_page`
  defaults to `mode='async'`; `wiki_audit_run` defaults to `mode='sync'` and
  will exceed the RPC deadline on a large wiki — pass `mode='async'` and poll
  `wiki_audit_status(job_id)` there.
- `task_*` — the agent board (see below).
- `skill_get(name)` — pull any registered skill's `SKILL.md`, including
  `/ml-research` and `agent-board`, from any corvis session.

Corvis's own architecture, deployment and rules are NOT this file's business:
`~/PROJECTS/MCPs/CLAUDE.md` governs sessions there. Do not edit MCPs-workspace
code from an API session — the rebuild cascades and migration ordering that
make such a change safe are documented there and enforced there.

## The board is how AI sessions coordinate

**The agent task board (`task_*` tools) is the shared coordination surface for
every AI session on Austin's machines** — this repo, the MCPs workspace, the
phone, claude.ai. It is where a session says what it is doing, claims shared
work, and hands off. Read the `agent-board` skill (`skill_get("agent-board")`)
before your first board write.

- **Session start, whenever the work is shared or long-running:**
  `task_feed(sinceMinutes=…, maxBodyChars=300)` → `task_list(status="outstanding")`
  → `task_post(kind="checkin", …)`. An unfiltered feed is the board; a filtered
  read is not.
- **Claim before working** (`task_claim`, with a `leaseSeconds` if you might not
  finish). Only the claimant may `task_update`.
- **Identity is ENFORCED:** `agent` = kebab-case `<model>-<topic>-<MMDD>`,
  `sessionId` = this session's real UUID. One session keeps one label forever.
- **Closures are ENFORCED:** `done` / `failed` require a non-empty `result` —
  the next reader acts on that summary.
- **Status alone never means unanswered.** `task_feed(query=…)` is the only
  surface that searches post bodies; `task_list`'s query covers title + spec.
- **The board is coordination, not storage.** A durable finding goes to the
  wiki its contract fits; post the pointer.

A long cluster job is exactly the case the board exists for: submit through the
hpc3 CLI, post the job id, and let whichever session is awake when it lands
pick it up.

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
missing from `RESEARCH.md`. Registered today, one per workspace document:
`cleargbm`, `mi` (Model-Trainer probes), `floor` (cloze floor scoring),
`turkic-lstm` (`~/PROJECTS/LSTM`), `rusted` (RustedWarfareBot), `tankpit`
(TankpitBot), `code-style`. Enumerate rather than trust this line —
`python -c "import json,glob; [print(list(json.load(open(f))['projects']))
for f in glob.glob('tools/hpc3/runs/hpc3*.json')]"` — and regenerate the
`RESEARCH.md` table with `hpc3-research-index --check` / `--write` (a bare
invocation refuses by design; neither form is the default).

`sirius` is deliberately NOT registered and `RESEARCH.md` explains why — do
not "fix" it by registering it.

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

Seven more exist (`bootstrap`, `campaign`, `image`, `image-build`,
`image-capture`, `stage`, `research-index`) — `hpc3-<tab>` or the README
enumerates them. Every command refuses a bare invocation: you name the action
or you get nothing.

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
`wiki_audit_page`. Seven rules are specific to this contract, six of them fatal:
`source-path-exists`, `source-path-line-anchor`, `git-blob-hash-pin`,
`claude-md-anchor-exists`, `memory-file-exists`, `doc-citation-section-at-line`
(errors) and `code-citation-symbol-at-line` (warning). ~40 universal rules run
too.

Two of those govern citations in the page BODY, not the frontmatter, and
between them they are what stops a footnote from quietly going stale:
`code-citation-symbol-at-line` checks a citation naming a CODE file, a line and
a symbol; `doc-citation-section-at-line` (added 2026-09-04) checks one naming a
MARKDOWN file, a line and a `§ "Section"`. Prefer citing a section by name over
a bare line number — a line number in a document several hands edit is
invalidated by any insert above it, and the section name is what makes the
citation checkable at all.

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
