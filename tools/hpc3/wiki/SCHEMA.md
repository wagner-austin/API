# Wiki Schema

This wiki is the persistent knowledge base for the `hpc3` package — the design record and incident narrative behind the rules the package enforces. AI maintains it. Humans consume it directly or through the README, which stays a command reference and points here for the reasoning.

The wiki is the **synthesis layer**. Raw sources live elsewhere (the package's own code and tests, the cluster's measured output, the git history, ledger and closure records). The wiki explains what those sources mean, links related topics, and tracks the incidents that forced each rule.

## Layout

```
wiki/
  SCHEMA.md           # this file (the contract)
  index.md            # ~25-line entry point listing all hubs (read first every session)
  log.md              # append-only operation log
  hubs/               # topic navigation pages
    <topic>.md        #   each hub lists its content pages with one-line descriptions
  pages/              # all content pages, flat, slug-named
    <slug>.md
```

## Three-tier knowledge graph

1. **`index.md`** (~25 lines) — lists all hubs with one-line descriptions and page counts. Read every session.
2. **Hubs** (`hubs/*.md`, ~30-50 lines each) — topic navigation. Each hub holds **inclusion links** to content pages: one link per line, format `[Title](../pages/<slug>.md) -- one-line description`. A content page can appear in multiple hubs (polyhierarchy).
3. **Content pages** (`pages/*.md`, 30-80 lines each) — atomic facts, one concept per page. Cite primary sources for every claim.

### Why this architecture
- **vs folders:** a page can have multiple parents without duplication
- **vs tags:** tags say "this is about X" but not HOW it relates; inclusion links from a hub show the specific relationship
- **vs flat index:** a flat index with 200+ entries becomes unnavigable; hubs partition navigation into manageable ~30-line chunks
- **vs a growing README:** this wiki exists because the README absorbed the incident narrative and reached 1,045 lines — a paragraph per lesson, forever. The README is the reference; this is the record.

## Atomicity rule

One concept per page, target 30-80 lines. When a page exceeds ~3,000 words or covers more than one distinct concept, split. "Atomic, not shallow" — a page carries full analytical depth on its one topic.

## Filename conventions

All lowercase, kebab-case, no special characters. Slugs name the concept, not the command (`image-ledger-lessons.md`, not `hpc3-image-build.md`) — commands are the README's vocabulary, concepts are this wiki's.

## Frontmatter

Every page opens with YAML frontmatter:

```yaml
---
title: Human-readable title
tags: [topic-tag, ...]              # cross-cutting, lowercase
related: ["[[other-slug]]", ...]    # explicit cross-refs — MUST be quoted
source_paths:                       # repo paths, relative to tools/hpc3/
  - "src/hpc3/contracts/job.py"
  - "README.md"
source_git_blobs:                   # one pin per source_paths entry, no exceptions
  "src/hpc3/contracts/job.py": "45f6be817460501c520ecca58b4f1dbc7341f4d0"
  "README.md": "c4cdcc31ae83beaede3c2635a943ddc0bcf0c083"
provenance:                         # evidence that is NOT a repo path
  - "sshare RawUsage measurement 2026-08-23 (cjmayer_lab)"
fact_checked: YYYY-MM-DD            # when claims were last confirmed current
confidence: high | medium | low     # how confident we are in this page's claims
---
```

`fact_checked` = when someone last confirmed the claims are still true. Git tracks edit history; no `updated:` field needed.

### The frontmatter is now machine-checked — three rules that were free before

This wiki was registered with the fleet on 2026-09-02 as the `hpc3` slug under
the **`code-paths`** source contract. Its pages are audited by
`wiki_audit_page` / `wiki_audit_run`; five `code-paths` rules are fatal.

1. **`related:` MUST be a quoted list.** The form this file documented until
   2026-09-02 — `related: [[a]], [[b]]` — is a YAML **syntax error** (a flow
   sequence followed by a comma). All 21 pages carried it and none of them
   parsed. Nothing caught it for the life of the wiki because an unregistered
   wiki is parsed by nothing. Write `related: ["[[a]]", "[[b]]"]`.
2. **`sources:` is replaced by `source_paths:` + `source_git_blobs:`.** Paths
   are relative to `tools/hpc3/` and must resolve at audit time
   (`source-path-exists`). The old shorthand — `contracts/budget.py`,
   `cli/triage`, `README.md@<commit>` — is not a resolvable path; write
   `src/hpc3/contracts/budget.py`, `src/hpc3/cli/triage.py`, `README.md`.
3. **Every `source_paths:` entry needs a `source_git_blobs:` pin.** Get it
   with `git hash-object <file>` from the repo root. `source-path-exists` only
   proves the path still resolves, which stays true across a total rewrite of
   the file; `git-blob-hash-pin` is the only rule that detects drift, and it
   fires only on pinned paths. An unpinned citation is a claim nothing will
   ever re-check.

Evidence that is not a repo path — job ids, `sshare`/`scontrol`/`sacct`
readings, `/pub` paths, and citations that genuinely point outside
`tools/hpc3/` (`platform_core.determinism_env`,
`model_trainer.cli.known_answer_registry`, RustedWarfareBot's
`member_command`) — goes in `provenance:`, never in `source_paths:`.

Also machine-checked: `index.md`'s page count (`enumeration-count`) and every
`[[slug]]` target (`wikilink-target-exists`). Bump the count when you add a
page.

## Cross-references

Internal links use `[[slug]]` syntax. When writing a page, **link rather than restate**. If a fact lives in another page, don't repeat it — link to it.

## Citations

Every non-common-knowledge factual claim carries its evidence. This wiki's claims are mostly **measurements and incidents**: a measured `scontrol`/`sshare`/`sacct` output, a dated job id, a commit hash, a guard rule that enforces the claim in code. The evidence therefore rides INLINE — dates, job ids, command output, rule names — exactly as the maintained README carried it, and the frontmatter says where the claim is checkable: `source_paths:` for anything in this repo (a module, `src/hpc3/clusters/hpc3.py`; a test, `tests/test_cluster.py`; the README), each with its `source_git_blobs:` pin, and `provenance:` for everything else (a ledger record, a dated job id, a cluster path). Footnote form (`[^1]: <target> <locator> — "<quote>"`) is used when a claim's evidence cannot ride inline.

The `README.md@<commit>` shorthand from the 2026-09-01 split is retired: it names no resolvable path, so `source-path-exists` cannot check it. Cite `README.md` as a path and let `git-blob-hash-pin` carry the version.

UNACCEPTABLE:
- Citing downstream artifacts of the wiki itself.
- Wiki-to-wiki citation chains beyond one hop.
- "Per a prior doc" without an inline source.
- A measured number with no date, machine, or command attached.

## Hub-link discipline (MANDATORY)

Every new page MUST be linked from at least one hub before the work ends. When adding a new page:
1. Pick the hub(s); polyhierarchy is fine.
2. Add the inclusion link in each hub: `[Title](../pages/<slug>.md) -- one-line description`.
3. Bump the hub's page count on its line in `index.md`.

## Index entry format

`index.md` lists every hub, one per line:

```markdown
[Hub Title](hubs/<slug>.md) -- one-line description of what's in this hub (N pages)
```

## Log entry format

`log.md` is append-only. Log every operation that changes the wiki's shape. Routine page edits don't need a log entry — git history covers those.

```markdown
## [YYYY-MM-DD] <operation> | <subject>
Pages written: <list>
Pages updated: <list>
Notes: <one-line summary>
```

## Common mistakes to avoid

- **Restating context that lives in a linked page.** Link, don't repeat.
- **Appending "Update:" sections to existing pages.** The page is the current truth; `log.md` is the journal.
- **A measured number without its measurement.** "free-gpu is preemptible" is a claim; "`PreemptMode=CANCEL`, read from `scontrol show partition`, 2026-08-28" is evidence.
- **Creating a page without a hub link.** Orphans accumulate fast and become invisible.
- **Growing the README instead of this wiki.** New incident narrative goes here; the README gets at most a one-line pointer.

## Schema version

v1.0 — initial scaffold, 2026-09-01, from the README split.
