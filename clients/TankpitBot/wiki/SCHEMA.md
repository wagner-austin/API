# TankpitBot Wiki — Schema

This wiki is the persistent knowledge base for TankpitBot. AI (Claude) maintains it. The human operator (Austin) contributes facts from live observation, wire captures, and gameplay experience.

The wiki is the **synthesis layer**. Raw sources live in capture files (`runs/`), wire dumps, client JS (`tpclient-*.js`), and the codebase. The wiki connects them — it explains what the data means, links related mechanics, and tracks verified-vs-unverified knowledge over time.

## Critical rules — fire automatically every time you touch the wiki

1. **Three-tier architecture only.** `index.md` (entry, ~25 lines) → `hubs/*.md` (topic navigation, ~30-50 lines) → `pages/*.md` (atomic content, 30-80 lines). No subdirectories under `pages/`. No nested hubs. No extra top-level dirs beyond `hubs/` and `pages/` — operational docs (handoffs, traces) live in the project's `docs/` tree, not in `wiki/`.
2. **Polyhierarchy via inclusion links.** A content page can be linked from multiple hubs. Hubs link to pages with `[Title](../pages/<slug>.md) -- one-line description`. Tags say "this is about X"; inclusion links from a hub show HOW it relates.
3. **Atomicity: one concept per page, 30-80 lines.** When a page exceeds ~3,000 words or covers more than one distinct concept, split. "Atomic, not shallow" — a page carries full analytical depth on its one topic. Reference catalogs (the V-table, the JS source map, the client-constants table) are one concept and stay whole even when long.
4. **Every new page MUST be hub-linked.** A page in `pages/` that no hub links to is an orphan: readers navigating from `index.md` never reach it. After writing a page, add an inclusion link from at least one hub and bump that hub's page count in `index.md`.
5. **Frontmatter required on every page.** Minimum: `title`, `tags`, `related`, `sources`, `fact_checked` (YYYY-MM-DD), `confidence` (high | medium | low).
6. **Citations travel with claims.** Every non-common-knowledge factual claim carries a footnote citation with a locator (file path + function/line, run ID + event, capture timestamp, URL fragment). No "per a prior source"; no "as documented earlier." Either inline the citation or weaken the claim.
7. **Cite primary sources, not downstream artifacts of the wiki itself.** The wiki is the synthesis layer. If a brief / report / summary is GENERATED from wiki content, that artifact is never cited back as a source. Cite whatever the artifact originally cited. (We do not currently render artifacts from wiki content; this rule is here to keep us honest if that ever changes.)
8. **Wiki-first retrieval.** Future AIs read `index.md` first, then follow the relevant hub link, then read the page. Do not re-derive facts the wiki already carries.

## Layout

```
wiki/
  SCHEMA.md           # this file (the contract)
  index.md            # ~25-line entry point listing all hubs (read first every session)
  log.md              # append-only operation log
  hubs/               # topic navigation pages
    game-mechanics.md
    protocol.md
    combat.md
    js-client.md
    architecture.md
    codebase.md
  pages/              # all content pages, flat, slug-named
    teleport-mechanics.md
    viewport-frame.md
    ...
```

## Three-tier knowledge graph

1. **`index.md`** (~25 lines) — lists all hubs with one-line descriptions and page counts. Read every session.
2. **Hubs** (`hubs/*.md`, ~30-50 lines each) — topic navigation. Each hub holds **inclusion links** to content pages, one per line: `[Title](../pages/<slug>.md) -- one-line description`. A content page can appear in multiple hubs (polyhierarchy).
3. **Content pages** (`pages/*.md`, 30-80 lines each) — atomic facts, one concept per page. Cite primary sources for every claim.

### Why this architecture (Karpathy LLM-wiki + IWE knowledge graph pattern)

- **vs memory files:** flat index, no cross-referencing, no citations, no staleness tracking. The wiki replaces them as the single source of truth for game mechanics, protocol, and architecture.
- **vs folders:** a page can have multiple parents without duplication.
- **vs flat index:** hubs partition navigation into manageable chunks.
- **vs code comments:** comments explain implementation; the wiki explains game mechanics and protocol behavior that the code implements but doesn't document.

## Page types

Every page is one of:

| Type | Purpose | Update cadence | Example |
|---|---|---|---|
| **reference** | Evergreen fact about a game mechanic, protocol format, or architecture decision | Updated in place when facts change | `teleport-mechanics.md`, `shoot-event-format.md` |
| **bug** | Diagnosed bug with root cause and fix status | Updated when fix lands or new data arrives | `combat-chase-bug.md` |

## Filename conventions

All lowercase, kebab-case, no special characters. No prefix needed (single game, unlike multi-jurisdiction wikis).

## Frontmatter

Every page opens with YAML frontmatter:

```yaml
---
title: Human-readable title
tags: [topic-tag, ...]
related: [[other-slug]], ...
sources: [run IDs, client JS refs, or "see footnotes"]
fact_checked: YYYY-MM-DD            # when claims were last confirmed against live data
confidence: high | medium | low
---
```

**Confidence levels:**
- **high** — wire-verified against multiple live captures, cross-confirmed
- **medium** — observed in gameplay or single-run data, not yet cross-verified
- **low** — inferred, user-reported without wire confirmation, or stale

`fact_checked` = when someone confirmed the claims are still true against current game behavior. Git tracks when the file was last edited.

## Cross-references

Internal links use `[[slug]]` syntax. **Link rather than restate.** If a fact about radar lives in `radar-mechanics.md`, don't repeat it in `fuel-system.md` — link to it.

## Citations

Every non-obvious factual claim must carry a footnote citation with a **locator**. Three formats:

**Data citation** (preferred for measured constants):
```
[^1]: run 20260611-004505, 255/255 hits at Manhattan distance 1
```

**Source citation** (for client JS or protocol analysis):
```
[^2]: tpclient-b45bd1ebc9c0c668.js, function Ig.h — skip-RLE decode loop
```

**User-reported** (for gameplay behavior not yet wire-verified):
```
[^3]: user (Austin), 2026-06-12 — "ferries can go anywhere on water"
```

Footnotes go at the bottom of the page, numbered sequentially.

If you cannot produce a citation for a claim, weaken it ("appears to..." / "user reports...") or drop it. **Never cite a downstream artifact of the wiki itself.**

## Hub-link discipline (MANDATORY)

Every new page MUST be linked from at least one hub before the work ends. When adding a page:

1. Pick the hub(s). Polyhierarchy is fine — link from multiple hubs when the page applies.
2. Add the inclusion link: `[Title](../pages/<slug>.md) -- one-line description`.
3. Bump the hub's page count on its line in `index.md`.

## Index entry format

```markdown
[Hub Title](hubs/<slug>.md) -- one-line description (N pages)
```

## Log entry format

`log.md` is append-only. Log every operation that changes the wiki's shape (new hubs, decomposition, audits, structural cleanups). Routine page edits don't need a log entry — git history covers those.

```markdown
## [YYYY-MM-DD] <operation> | <subject>
Pages written: <list>
Pages updated: <list>
Notes: <one-line summary>
```

## Common mistakes to avoid

- **Restating context that lives in a linked page.** If `radar-mechanics.md` exists, don't re-explain radar in `fuel-system.md` — link.
- **Appending "Update:" sections.** The page is the current truth; `log.md` is the journal.
- **Citing without a locator.** "Per a live run" is not a citation. "run 20260611-004505, 255/255 at distance 1" is.
- **Creating a page without a hub link.** Always link from at least one hub immediately — orphans accumulate fast and become invisible.
- **Putting operational scratch in `wiki/`.** Handoff briefs, raw analysis traces, and one-off planning docs go in the project's `docs/` tree, not under `wiki/`.

## Storage

Wiki pages are markdown files in `wiki/` within the TankpitBot repo, tracked by the repo's git. No separate git repo needed — the parent repo provides version history.

## Relationship to memory files

The wiki **replaces** the `.claude/projects/.../memory/` files for game mechanics, protocol, and architecture knowledge. Memory files may still hold user preferences and AI behavior feedback (things about how to work, not what the game does). Any game-knowledge memory file should be migrated to the wiki and then removed.

## Schema version

v1.0 — 2026-06-26. Aligned with the `/wiki-init` v1.0 spec: critical rules block, explicit no-extra-top-level-dirs, downstream-artifact citation ban, atomicity exception for reference catalogs. Prior v0.1 (2026-06-16) bootstrapped the three-tier graph; v1.0 codifies the rules and removes the `artifacts/` + `handoffs/` dirs that had accreted outside the spec layout.
