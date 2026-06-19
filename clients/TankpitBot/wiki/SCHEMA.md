# TankpitBot Wiki — Schema

This wiki is the persistent knowledge base for TankpitBot. AI (Claude) maintains it. The human operator (Austin) contributes facts from live observation, wire captures, and gameplay experience.

The wiki is the **synthesis layer**. Raw sources live in capture files (`runs/`), wire dumps, client JS (`tpclient-*.js`), and the codebase. The wiki connects them — it explains what the data means, links related mechanics, and tracks verified-vs-unverified knowledge over time.

## Layout

```
wiki/
  SCHEMA.md           # this file
  index.md            # ~30-line entry point listing all hubs (read first every session)
  log.md              # append-only operation log
  hubs/               # topic hub pages (navigation layer)
    game-mechanics.md
    protocol.md
    combat.md
    architecture.md
  pages/              # all content pages, flat, slug-named
    teleport-mechanics.md
    viewport-frame.md
    ...
```

**Three-tier knowledge graph (v0.1, 2026-06-16):**

1. **index.md** (~30 lines) -- lists all hubs. Read every session. An AI reads this to find which hub matches the current task.
2. **Hub pages** (20-40 lines each) -- topic navigation. Each hub contains **inclusion links** (a link on its own line) to content pages, with one-line descriptions. A content page can be linked from multiple hubs (polyhierarchy). Hub pages answer "what do we know about this topic area?"
3. **Content pages** (30-80 lines each) -- atomic facts, one concept per page. Content pages answer "what are the specific facts about this thing?"

**Why this architecture (Karpathy LLM-wiki + IWE knowledge graph pattern):**
- **vs memory files:** memory files are flat, limited index, no cross-referencing, no citations, no staleness tracking. The wiki replaces them as the single source of truth.
- **vs folders:** a page can have multiple parents without duplication
- **vs flat index:** hubs partition navigation into manageable chunks
- **vs code comments:** comments explain implementation; the wiki explains game mechanics and protocol behavior that the code implements but doesn't document

**Atomicity rule:** one concept per page, target 30-80 lines. When a page exceeds ~100 lines or covers more than one distinct concept, split.

## Page types

Every page is one of:

| Type | Purpose | Update cadence | Example |
|---|---|---|---|
| **reference** | Evergreen fact about a game mechanic, protocol format, or architecture decision | Updated in place when facts change | `teleport-mechanics.md`, `shoot-event-format.md` |
| **bug** | Diagnosed bug with root cause and fix status | Updated when fix lands or new data arrives | `combat-chase-bug.md` |

## Filename conventions

All lowercase, kebab-case, no special characters. No prefix needed (single game, unlike multi-jurisdiction CML wiki).

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

Internal links use `[[slug]]` syntax. Link rather than restate. If a fact about radar is in `radar-mechanics.md`, don't repeat it in `fuel-system.md` — link to it.

## Citations

Every non-obvious factual claim must carry a footnote citation. Two formats:

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

If you cannot produce a citation for a claim, weaken it ("appears to..." / "user reports...") or drop it.

## Common mistakes to avoid

- **Restating context that lives in a linked page.** If `radar-mechanics.md` exists, don't re-explain radar in `fuel-system.md` — link.
- **Appending "Update:" sections.** The page is the current truth; `log.md` is the journal.
- **Citing without a locator.** "Per a live run" is not a citation. "run 20260611-004505, 255/255 at distance 1" is.
- **Duplicating across wiki + memory files.** The wiki is the single source of truth. Memory files that duplicate wiki content should be removed.

## Storage

Wiki pages are markdown files in `wiki/` within the TankpitBot repo, tracked by the repo's git. No separate git repo needed — the parent repo provides version history.

## Relationship to memory files

The wiki **replaces** the `.claude/projects/.../memory/` files for game mechanics, protocol, and architecture knowledge. Memory files may still hold user preferences and AI behavior feedback (things about how to work, not what the game does). Any game-knowledge memory file should be migrated to the wiki and then removed.

## Schema version

v0.1 — initial schema, 2026-06-16. Modeled on the CML Office Wiki (v0.5) with adaptations for game-bot context (no jurisdiction prefix, run-ID citations instead of vault/drive, simplified page types).
