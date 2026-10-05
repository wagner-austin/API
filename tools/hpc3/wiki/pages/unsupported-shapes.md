---
title: What this package cannot submit, as decisions rather than discoveries
tags: [submission, scope]
hubs: [submission]
related: ["[[chains]]", "[[submission-rules]]", "[[facts-are-code]]"]
source_paths:
  - "src/hpc3/contracts/job.py"
  - "src/hpc3/core/env_probe.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/contracts/job.py": "1bc6b5aaafdbf0f5dc0e8b3c8b61ddca89db936f"
  "src/hpc3/core/env_probe.py": "19d41901aac9ea756003760b58338d1b163a9c47"
  "README.md": "7ac1acb5c894dafe7537926323ae96d7d5b57dbe"
fact_checked: 2026-09-14
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
stating: a JVM project pins no Python packages, so the pin check holds
nothing about its payload to a version. Since 2026-10-05 its environment is
still probed — `verify_environment` runs the environment's own `bin/python`
whatever the pins say, and refuses one whose interpreter belongs to another
installation — but what the JVM itself is goes unchecked, which is the "both
paths exist, both pass, the results aren't comparable" failure the pin check
was built for ([[environment-pins]]).
