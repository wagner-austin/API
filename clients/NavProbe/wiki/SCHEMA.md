# Wiki Schema

This wiki is the persistent knowledge base for **NavProbe**, a reproducibility instrument for simulated navigation. AI maintains it. Humans consume it directly.

The wiki is the **synthesis layer**. Raw sources live elsewhere: this package's own source code, the installed vendor packages, and published papers. The wiki explains what those sources mean, links related topics, and tracks analytical conclusions over time.

It records two kinds of thing, and the distinction matters: **what the instrument was built to be** (design decisions and the reasoning behind them) and **what the instrument has measured** (results, with the conditions they were taken under). A results page whose conditions are not stated is worthless, because a determinism verdict is meaningless without the seed, the step count, the batch width, and the backend.

## Layout

```
wiki/
  SCHEMA.md           # this file (the contract)
  index.md            # entry point listing all hubs (read first every session)
  log.md              # append-only operation log
  hubs/               # topic navigation pages
    <topic>.md        #   each hub lists its content pages with one-line descriptions
  pages/              # all content pages, flat, slug-named
    <slug>.md
```

## Three-tier knowledge graph

1. **`index.md`** (~25 lines) — lists all hubs with one-line descriptions and page counts. Read every session.
2. **Hubs** (`hubs/*.md`, ~30-50 lines each) — topic navigation. Each hub holds **inclusion links** to content pages: one per line, format `[Title](../pages/<slug>.md) -- one-line description`. A content page can appear in multiple hubs (polyhierarchy).
3. **Content pages** (`pages/*.md`, 30-80 lines each) — atomic findings, one concept per page. Cite primary sources for every claim.

### Why this architecture

- **vs folders:** a page can have multiple parents without duplication
- **vs tags:** tags say "this is about X" but not HOW it relates; inclusion links from a hub show the specific relationship
- **vs flat index:** a flat index with 200+ entries becomes unnavigable; hubs partition navigation into ~30-line chunks
- **vs manually maintained link lists:** hub inclusion links ARE the structure — adding a page means linking it from its hub(s)

## Atomicity rule

One concept per page, target 30-80 lines. When a page exceeds ~3,000 words or covers more than one distinct concept, split. "Atomic, not shallow" — a page carries full analytical depth on its one topic.

## Filename conventions

All lowercase, kebab-case, no special characters. Name the page after **the finding, not the subject**: `digest-fold-requires-length-prefix.md`, not `digests.md`. A page whose title is a noun accumulates unrelated facts until it is no longer atomic; a page whose title is a claim can only ever be about that claim.

## Frontmatter

Every page opens with YAML frontmatter:

```yaml
---
title: Human-readable title, stated as the finding
tags: [topic-tag, ...]              # cross-cutting, lowercase
related: ["[[other-slug]]", ...]    # explicit cross-refs — MUST be quoted
source_paths:                       # repo paths, relative to clients/NavProbe/
  - "src/navprobe/determinism.py"
source_git_blobs:                   # one pin per tracked source_paths entry
  "src/navprobe/determinism.py": "<40-hex blob sha>"
provenance:                         # evidence that is NOT a repo path
  - "mujoco-warp 3.11.0"
fact_checked: YYYY-MM-DD            # when claims were last confirmed current
confidence: high | medium | low
---
```

Two frontmatter traps, both of which this file documented wrongly until
2026-09-03:

1. **`related:` MUST be a quoted list.** The form shown here previously —
   `related: [[a]], [[b]]` — is a YAML **syntax error**: a flow sequence
   followed by a comma. A page carrying it does not parse, and an unparseable
   page becomes a single `load-error` finding under which **no other rule
   runs** — the pin checks, the line anchors, and the existence checks all go
   silent. It reads as a clean page. Write `related: ["[[a]]", "[[b]]"]`.
2. **`sources:` is not a field.** This wiki audits under the `code-paths`
   source contract, which reads `source_paths:` + `source_git_blobs:`. A
   freeform `sources:` list is ignored by every rule, so its contents are
   never checked against anything. Evidence that is not a repo path goes in
   `provenance:`.

`fact_checked` = when someone last confirmed the claims are still true. Git tracks edit history; no `updated:` field. Pages without `fact_checked` have never been independently verified.

### Additional frontmatter for measurement pages

A page reporting a measurement MUST additionally carry the conditions, because the number means nothing without them:

```yaml
measured_with:
  package: mujoco-mjx 3.11.0        # exact version, not "latest"
  backend: cpu | cuda
  seed: 7
  step_count: 200
  repetitions: 5
  world_counts: [1, 2, 4, 8, 16, 32, 64]
```

A measurement page missing `measured_with` is a claim without a locator and is treated as unverified.

## Cross-references

Internal links use `[[slug]]` syntax. Example: `The fold is length-prefixed for the reason in [[digest-fold-requires-length-prefix]].`

When writing a page, **link rather than restate**. If a fact lives in another page, don't repeat it — link to it.

## Citations

Every non-common-knowledge factual claim carries a footnote citation with a **locator**.

```
[^1]: <target> <locator> — "<verbatim quote>"
[^2]: <target> <locator> [synthesis] — <description of what supports the claim>
```

`<target>` is one of:

- **A source file in this repo**, cited by path and line range: `src/navprobe/digest.py L56-80`
- **A test in this repo**, cited by path and test name: `tests/test_digest.py::TestDigestRun::test_equal_length_runs_with_the_same_concatenation_do_not_collide`
- **An installed vendor package**, cited by distribution and version: `mujoco-mjx 3.11.0`
- **An observed command result**, cited as `[observed]` with the exact command that produced it
- **An external URL or paper**, cited with a section or line locator

### Executable claims are cited by their test

This wiki documents a package with 100% branch coverage. Where a claim is enforced by a test, **cite the test** — a citation that can fail on the next `make check` is stronger than a quotation, because it decays loudly. A claim about this package's behaviour with no test to cite is a claim the package does not actually guarantee, and should be weakened or dropped.

### Citation hierarchy

1. **This repo's source and tests** — preferred for claims about the instrument.
2. **Installed vendor packages at a pinned version** — preferred for claims about MJX, JAX, or MuJoCo. Record the version; a vendor claim without one has no locator.
3. **Observed command output** — acceptable for measurements, with the command recorded so it can be re-run.
4. **External URLs and papers** — for published results this wiki did not produce.

UNACCEPTABLE:

- Citing the package README or this wiki's own pages as the source for a factual claim. Both are downstream syntheses. Cite the code, the test, or the run.
- "As measured earlier" or "per the sweep" without the conditions inline.
- A vendor claim with no version.

## Hub-link discipline (MANDATORY)

Every new page MUST be linked from at least one hub before the work ends. A page in `pages/` that no hub links to is an orphan: readers navigating from `index.md` never reach it.

When adding a new page:

1. Pick the hub(s). Polyhierarchy is fine — link from multiple hubs if the page applies.
2. Add the inclusion link in each hub: `[Title](../pages/<slug>.md) -- one-line description`.
3. Bump the hub's page count on its line in `index.md`.
4. Bump the total page count in the `index.md` header.

## Index entry format

```markdown
[Hub Title](hubs/<slug>.md) -- one-line description of what's in this hub (N pages)
```

## Log entry format

`log.md` is append-only. Log every operation that changes the wiki's shape (new hubs, decomposition, audits, structural cleanups) and every measurement run. Routine page edits don't need a log entry — git history covers those.

```markdown
## [YYYY-MM-DD] <operation> | <subject>
Pages written: <list>
Pages updated: <list>
Notes: <one-line summary>
```

## Common mistakes to avoid

- **Restating context that lives in a linked page.** Link, don't repeat.
- **Appending "Update:" sections instead of editing in place.** The page is the current truth; `log.md` is the journal.
- **Citing without a locator.** "Per the MJX docs" is not a citation. "`mujoco-mjx 3.11.0`, `mjx.step` signature" is.
- **Recording a measurement without its conditions.** See `measured_with` above.
- **Creating a page without a hub link.** Orphans accumulate fast and become invisible.
- **Writing a page titled after a noun.** Title it after the finding.

## Schema version

v1.0 — initial scaffold, 2026-08-13. Extends the base v1.0 spec with the
`measured_with` frontmatter block and the cite-the-test citation rule, both of
which exist because this wiki documents an instrument rather than a subject.
