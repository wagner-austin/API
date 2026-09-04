# hpc3 Wiki

**Read this first.** 4 topic hubs, 24 content pages. This is the design record
and incident narrative behind the rules the `hpc3` package enforces; the
README next door is the command reference. Follow the hub link for your
topic; each hub lists its pages with one-line descriptions.

## Hubs

[Submission](hubs/submission.md) -- run/sweep/chain documents, job arrays, campaign convergence, the rules that refuse, unsupported shapes, invariant placement (8 pages)
[Images and staging](hubs/images-and-staging.md) -- identity over transport: images, pins, staged bytes, determinism, known answers, capture-source and spec-symbol drift (8 pages)
[Cluster facts](hubs/cluster-facts.md) -- partitions, billing, facts-are-code, node-local scratch, what a job looks like from the cluster, which interpreters it actually has (6 pages)
[Operations](hubs/operations.md) -- triage, closures, budgets (4 pages)

## How this works

**Three tiers:** this index (read every session) → hub pages (read when topic
matches) → content pages (read when you need the facts). A content page can
be linked from multiple hubs.

**Adding new content:** create the content page in `pages/`, add an inclusion
link from the relevant hub(s), bump page counts here. New incident narrative
goes HERE, not into the README — the README grew to 1,045 lines that way.

**The rules:** see `SCHEMA.md` — atomicity, frontmatter, citations, hub-link
discipline.
