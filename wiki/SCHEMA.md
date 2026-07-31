# Wiki Schema

This wiki is the persistent codebase-engineering knowledge base for the api platform (`~/PROJECTS/api/`) — a typed Python monorepo for ML training, NLP, and media services (FastAPI + RQ + Redis, strict mypy, 100% test coverage). AI maintains it. Humans consume it directly or through derived artifacts.

The wiki is the **synthesis layer** for how the monorepo *works* — service architecture, shared platform libs, ML pipelines, client integrations, cross-cutting infrastructure. Raw sources live in the code itself, per-service READMEs, and inline comments; this wiki explains what those pieces mean together.

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
3. **Content pages** (`pages/*.md`, 30-80 lines each) — atomic facts, one concept per page. Cite primary sources (code paths, service READMEs, RFCs, papers) for every claim.

### Why this architecture
- **vs folders:** a page can have multiple parents without duplication
- **vs tags:** tags say "this is about X" but not HOW it relates; inclusion links from a hub show the specific relationship
- **vs flat index:** a flat index with 200+ entries becomes unnavigable; hubs partition navigation into manageable ~30-line chunks
- **vs MOCs (manually maintained link lists):** hub inclusion links ARE the structure — adding a page means linking it from its hub(s), and navigation updates automatically

## Atomicity rule

One concept per page, target 30-80 lines. When a page exceeds ~3,000 words or covers more than one distinct concept, split. "Atomic, not shallow" — a page carries full analytical depth on its one topic. A 150-line page covering four loosely-related topics is four pages, not one.

## Filename conventions

All lowercase, kebab-case, no special characters. Use a domain-meaningful prefix when the topic could be ambiguous (e.g. `covenant-ml-pipeline.md` vs `art-trainer-lora-config.md`). Avoid bare topic names that become ambiguous as the wiki grows.

## Frontmatter

Every page opens with YAML frontmatter:

```yaml
---
title: Human-readable title
tags: [topic-tag, ...]              # cross-cutting, lowercase
related:                            # explicit cross-refs, one "[[slug]]" per line
  - "[[other-slug]]"
source_paths:                       # see Citations
  - libs/covenant_ml/src/...        # workspace-relative repo path
source_git_blobs:                   # optional; see Blob pinning
  libs/covenant_ml/src/...: <blob-sha>
fact_checked: YYYY-MM-DD            # when claims were last confirmed current
confidence: high | medium | low     # how confident we are in this page's claims
hubs: [libs]                        # every hub whose file links this page
---
```

The field is `source_paths:`, **not** `sources:`. This wiki runs the `code-paths` source contract in `corvis-wiki-check`; the other contracts reserve `sources:` for CSL-JSON bibliographic objects, so a `sources:` list of plain path strings fails the audit with a `load-error` (`expected object, got string`) and the whole page drops out of the corpus — taking every `[[wikilink]]` that targets it down with it. Canonical key names: `packages/wiki-check/src/pure/canonical-contracts.ts::CODE_PATHS_FRONTMATTER_KEYS`.

`hubs:` must list exactly the hubs whose files link this page — `hubs-membership-consistent` compares the two directions and fails on either-way drift.

`fact_checked` = when someone last confirmed the claims are still true against the code. Git tracks edit history; no `updated:` field needed. Pages without `fact_checked` have never been independently verified.

### Blob pinning

`source_git_blobs:` maps a `source_paths:` entry to the `git ls-tree HEAD` blob-hash it was verified against. It is opt-in per page, but **all-or-nothing once opted in**: a page that declares the field must pin every git-tracked entry in its `source_paths:` (untracked artifacts are exempt — they are unpinnable by nature). When the file changes, `git-blob-hash-pin` fails; the fix is to re-verify the page's claims against the new content and *then* repin, never to repin alone.

### Verifying a page

Run the audit chain before finishing any page edit:

```
wiki_audit_page(wikiSlug="api-codebase", pageSlug="<slug>")
```

## Cross-references

Internal links use `[[slug]]` syntax (Obsidian/Roam convention). Example: `LoRA training runs through [[art-trainer-kohya-backend]].`

When writing a page, **link rather than restate**. If a fact lives in another page, don't repeat it — link to it.

## Citations

Every non-common-knowledge factual claim carries a footnote citation with a **locator**.

```
[^1]: <target> <locator> — "<verbatim quote>"
[^2]: <target> <locator> [synthesis] — <description of what supports the claim>
```

`<target>` for this wiki is one of:
- A code path (`services/Art-Trainer/src/...`, `libs/platform_core/src/...`)
- A service README (`services/covenant-radar-api/README.md`)
- A commit SHA
- An external URL (paper, RFC, vendor API doc)
- A `[[wiki-slug]]` page that itself cites the primary

`<locator>` is required: line range (`:120-135`), function name, section header (`§3.2`), or paper section.

### Citation hierarchy

1. **Primary sources** (code paths, service READMEs, inline docstrings) — preferred.
2. **External URLs** — papers, RFCs, vendor API docs. Test that the URL still resolves at fact-check time.
3. **Other wiki pages** — acceptable when they themselves cite primaries.

UNACCEPTABLE:
- Citing **downstream artifacts of the wiki itself** — generated reports, rendered summaries. Cite whatever they originally cited.
- Wiki-to-wiki citation chains beyond one hop.
- "Per an earlier decision" or "as discussed" without an inline source.

## Hub-link discipline (MANDATORY)

Every new page MUST be linked from at least one hub before the work ends. A page in `pages/` that no hub links to is an orphan.

When adding a new page:
1. Pick the hub(s). Polyhierarchy is fine — link from multiple hubs if the page applies.
2. Add the inclusion link in each hub: `[Title](../pages/<slug>.md) -- one-line description`.
3. Bump the hub's page count on its line in `index.md`.
4. Bump the total page count in the `index.md` header if you track one there.

## Index entry format

`index.md` lists every hub, one per line:

```markdown
[Hub Title](hubs/<slug>.md) -- one-line description of what's in this hub (N pages)
```

Read `index.md` before anything else — it's the navigation primitive.

## Log entry format

`log.md` is append-only. Log every operation that changes the wiki's shape (new hubs, decomposition, audits, structural cleanups). Routine page edits don't need a log entry — git history covers those.

```markdown
## [YYYY-MM-DD] <operation> | <subject>
Pages written: <list>
Pages updated: <list>
Notes: <one-line summary>
```

## Common mistakes to avoid

- **Restating context that lives in a linked reference page.** Link, don't repeat.
- **Appending "Update:" sections instead of editing in place.** The page is the current truth; `log.md` is the journal.
- **Citing without a locator.** "Per `platform_core`" is not a citation. "Per `libs/platform_core/src/http.py/AsyncHttpClient`" is.
- **Skipping the backlink audit on new pages.** When adding `[[new-slug]]`, scan existing pages mentioning the same topic and add the link.
- **Creating a page without a hub link.** Always link from at least one hub immediately.
- **Duplicating what per-service READMEs already say.** READMEs are the front door for each service; the wiki extends them with cross-service context and subsystem-depth pages.

## Schema version

v1.0 — initial scaffold, 2026-07-06.
