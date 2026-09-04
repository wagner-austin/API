# Wiki Schema

This wiki is the persistent knowledge base for **RustedWarfareBot** — a headless Rusted Warfare client that plays the game autonomously. AI maintains it. Humans consume it directly.

The wiki is the **synthesis layer**. Raw sources live elsewhere: the game's boot logs and `-printunits` dumps archived under `wiki/sources/`, the pinned game tree at `.game/` (mod `.ini` files, launcher scripts, `preferences.ini`), and this client's own source code. The wiki explains what those sources mean, links related topics, and tracks analytical conclusions over time.

**Where evidence lives.** Anything a page cites goes in `wiki/sources/<probe-name>/` and is versioned in git. `runs/` is for bulk, ephemeral run output and is not tracked. The distinction is load-bearing: an artifact that isn't in the repo makes its citation unverifiable on any machine but the one that produced it. If a run turns up something a page will cite, promote that artifact into `wiki/sources/` before writing the claim.

## Layout

```
wiki/
  SCHEMA.md           # this file (the contract)
  index.md            # ~25-line entry point listing all hubs (read first every session)
  log.md              # append-only operation log
  hubs/               # topic navigation pages
    <topic>.md        #   each hub lists its content pages with one-line descriptions
    ...               #   a content page can appear in multiple hubs (polyhierarchy)
  pages/              # all content pages, flat, slug-named
    <slug>.md
    ...
```

## Three-tier knowledge graph

1. **`index.md`** (~25 lines) — lists all hubs with one-line descriptions and page counts. Read every session.
2. **Hubs** (`hubs/*.md`, ~30-50 lines each) — topic navigation. Each hub holds **inclusion links** to content pages: one link per line, format `[Title](../pages/<slug>.md) -- one-line description`. A content page can appear in multiple hubs (polyhierarchy via inclusion links, Karpathy LLM-wiki + IWE pattern).
3. **Content pages** (`pages/*.md`, 30-80 lines each) — atomic facts, one concept per page. Cite primary sources for every claim.

### Why this architecture
- **vs folders:** a page can have multiple parents without duplication
- **vs tags:** tags say "this is about X" but not HOW it relates; inclusion links from a hub show the specific relationship
- **vs flat index:** a flat index with 200+ entries becomes unnavigable; hubs partition navigation into manageable ~30-line chunks
- **vs MOCs (manually maintained link lists):** hub inclusion links ARE the structure — adding a page means linking it from its hub(s), and navigation updates automatically

## Atomicity rule

One concept per page, target 30-80 lines. When a page exceeds ~3,000 words or covers more than one distinct concept, split. "Atomic, not shallow" — a page carries full analytical depth on its one topic. A 150-line page covering four loosely-related topics is four pages, not one.

## Filename conventions

All lowercase, kebab-case, no special characters. Prefix by sub-domain when it disambiguates (`engine-command-controller.md`, `harness-nodisplay.md`, `mechanics-unit-costs.md`). Avoid bare topic names that become ambiguous as the wiki grows.

## Frontmatter

Every page opens with YAML frontmatter:

```yaml
---
title: Human-readable title
tags: [topic-tag, ...]              # cross-cutting, lowercase
related: ["[[other-slug]]", ...]    # explicit cross-refs — MUST be quoted
source_paths:                       # see Citations — workspace-relative
  - "wiki/sources/m0-probe/x.log:42"#   <path>:<line> anchors are verified in-bounds
  - ".game/some-file.ini"           #   bare paths must resolve
source_git_blobs:                   # OPTIONAL: git blob pin per TRACKED path
  "src/thing.py": "<blob-sha>"      #   omit the field entirely when every source is untracked
game_version: "1.15 (code 176, ...)"# REQUIRED when claims depend on the game build
fact_checked: YYYY-MM-DD            # when claims were last confirmed current
confidence: high | medium | low     # how confident we are in this page's claims
hubs: [hub-slug, ...]               # every hub linking this page
---
```

Two frontmatter traps, both of which this file documented wrongly until
2026-09-03:

1. **`related:` MUST be a quoted list.** The form shown here previously —
   `related: [[a]], [[b]]` — is a YAML **syntax error**: a flow sequence
   followed by a comma. A page carrying it does not parse, and an unparseable
   page becomes a single `load-error` finding under which **no other rule
   runs** — pins, line anchors, and existence checks all go silent, so the
   page reads as clean. Two pages written on 2026-09-03 hit exactly this,
   copied from the example above. Write `related: ["[[a]]", "[[b]]"]`.
2. **`sources:` is not a field.** The contract reads `source_paths:` +
   `source_git_blobs:`. A freeform `sources:` list is ignored by every rule,
   so nothing it names is ever checked. Evidence that is not a repo path —
   a run label, a measurement, a log date range — goes in `provenance:`.

`source_paths` follows the workspace's **`code-paths`** source contract, the same one the sibling TankpitBot wiki uses. Entries resolve relative to this client directory. Three shapes are recognised: a bare path (must exist), a `<path>:<line>` anchor (line must be within the file), and an external URL (skipped by the existence rules). Gitignored artifacts — everything under `runs/` and `.game/` — are unpinnable by nature and exempt from `source_git_blobs`, but their existence is still enforced, so a typo'd path fails loudly.

A page whose sources are all untracked omits `source_git_blobs` entirely. A page citing tracked repo code must pin **every** tracked path it cites; partial pinning is a finding.

`fact_checked` = when someone last confirmed the claims are still true. Git tracks edit history; no `updated:` field needed. Pages without `fact_checked` have never been independently verified.

### Game-version pinning (domain-specific, MANDATORY)

Every page whose claims depend on the game build carries a `game_version` field:

```yaml
game_version: "1.15 (code 176, build #28)"
```

Rusted Warfare is obfuscated and Steam auto-updates. A class mapping, a CLI flag, or a unit stat that is true for 1.15 may be false for 1.16 — silently, with no error. A page asserting engine internals without `game_version` is unverifiable. When the pinned build changes, every page carrying the old version is stale until re-checked.

Pages about the bot's own code (architecture, coding standards) omit `game_version` and instead pin `source_paths` to the files they describe.

## Cross-references

Internal links use `[[slug]]` syntax (Obsidian/Roam convention). Example: `Orders are queued through [[engine-command-controller]].`

When writing a page, **link rather than restate**. If a fact lives in another page, don't repeat it — link to it.

## Citations

Every non-common-knowledge factual claim carries a footnote citation with a **locator**.

```
[^1]: <target> <locator> — "<verbatim quote>"
[^2]: <target> <locator> [synthesis] — <description of what supports the claim>
```

`<target>` is one of:
- A **captured game artifact** under `wiki/sources/` (boot log, `-printunits` dump, replay). Locator = filename + line number or timestamp.
- A **file in the game tree** (`.game/…`) — launcher scripts, `preferences.ini`, mod `.ini` files. Locator = path + line or key.
- A **jar-internal fact** — class/package path inside `game-lib.jar`. Locator = the fully-qualified name.
- A **file in this repo** — source path + symbol.
- An **external URL** (upstream projects, changelog). Test that it still resolves at fact-check time.

### Citation hierarchy

1. **Observed game behaviour** (a log line from a run we performed, a dump the game produced) — strongest. The game is the oracle.
2. **Static artifacts** (jar contents, `.ini` files, launcher scripts) — strong, but describes what *can* happen, not what *does*.
3. **External sources** (changelog, community repos) — weakest; useful for orientation, never sufficient for an engine-internals claim.

UNACCEPTABLE:
- **Inference presented as observation.** "`-sandbox` boots into sandbox mode" is a guess until a run proves it. Mark unverified claims `confidence: low` and say what test would settle them.
- **Citing a scratchpad, temp path, or untracked artifact.** Probe output that supports a claim must be archived under `wiki/sources/` and committed first — temp directories evaporate, and a gitignored artifact is invisible to everyone but the machine that made it. Either way the citation dies.
- Citing downstream artifacts generated *from* wiki content back as sources.
- Wiki-to-wiki citation chains beyond one hop.

## Hub-link discipline (MANDATORY)

Every new page MUST be linked from at least one hub before the work ends. A page in `pages/` that no hub links to is an orphan: readers navigating from `index.md` never reach it.

When adding a new page:
1. Pick the hub(s). Polyhierarchy is fine — link from multiple hubs if the page applies.
2. Add the inclusion link in each hub: `[Title](../pages/<slug>.md) -- one-line description`.
3. Bump the hub's page count on its line in `index.md`.
4. Bump the total page count in the `index.md` header.

## Index entry format

`index.md` lists every hub, one per line:

```markdown
[Hub Title](hubs/<slug>.md) -- one-line description of what's in this hub (N pages)
```

Read `index.md` before anything else — it's the navigation primitive.

## Log entry format

`log.md` is append-only. Log every operation that changes the wiki's shape (new hubs, decomposition, audits, structural cleanups) and every live run that produced findings. Routine page edits don't need a log entry — git history covers those.

```markdown
## [YYYY-MM-DD] <operation> | <subject>
Pages written: <list>
Pages updated: <list>
Notes: <one-line summary>
```

## Common mistakes to avoid

- **Restating context that lives in a linked reference page.** Link, don't repeat.
- **Appending "Update:" sections instead of editing in place.** The page is the current truth; `log.md` is the journal.
- **Citing without a locator.** "Per the boot log" is not a citation. "`wiki/sources/m0-probe/nodisplay-boot.log:47`" is.
- **Recording an obfuscated class name without the evidence that mapped it.** `game.i` means nothing in 1.16. Always cite the log line or method signature that identified it.
- **Creating a page without a hub link.** Orphans accumulate fast and become invisible.

## Schema version

v1.0 — initial scaffold, 2026-07-25. Derived from the `/wiki-init` v1.0 spec, plus two domain-specific additions: `game_version` pinning and the observed-behaviour-over-inference citation hierarchy.
