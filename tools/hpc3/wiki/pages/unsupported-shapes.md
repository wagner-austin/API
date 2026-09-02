---
title: What this package cannot submit, as decisions rather than discoveries
tags: [submission, scope]
related: [[chains]], [[submission-rules]], [[facts-are-code]]
sources: [contracts/job.py, README.md@4dc63f17]
fact_checked: 2026-09-01
confidence: high
---

# What this package cannot submit, as decisions rather than discoveries

`JobSpec` describes **one single-node job**, with GPUs or without. The
current table of inexpressible shapes — multi-node/MPI, job arrays, explicit
`--qos`, `--constraint`/`--exclusive` — lives in the README's
"What this cannot submit" section, where a test (`test_examples.py`) holds it
present and holds lifted limits OUT of it; this page carries the reasoning
and the history the table cannot.

None of the absent shapes are hard to add, and the cluster-facts layer
already carries what the checks would need. They are absent because they were
never built, not because they were judged wrong — recorded so the gap is a
decision rather than a discovery.

## Three things left this list

**Job dependencies** were on it and are not any more — see [[chains]].

**Job arrays** were on it and are not any more — inverted, in fact: a sweep
is now submitted AS one array call, and the member-by-member loop the row
described no longer exists. The measured identity rules and the
script-is-the-member-table design are in [[job-arrays]].

**CPU-only** was on it and is not any more. `"gpu": null` on a CPU partition
submits, so `cleargbm_rs`, SIRIUS and ZODIAC are reachable. One caveat worth
stating: `pinned_packages` verification runs the environment's own
`bin/python`, and an empty pin map makes **no round trip at all**. A JVM
project is therefore submittable while getting only `test -d` on its
environment — the weakest guarantee here, and exactly the "both paths exist,
both pass, the results aren't comparable" failure the pin check was built for
([[environment-pins]]).
