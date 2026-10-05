---
title: The environment is the pinned one, proven by its own interpreter
tags: [identity, environments]
hubs: [images-and-staging]
related: ["[[image-build-flow]]", "[[submission-rules]]", "[[unsupported-shapes]]"]
source_paths:
  - "src/hpc3/contracts/pins.py"
  - "src/hpc3/core/env_probe.py"
source_git_blobs:
  "src/hpc3/contracts/pins.py": "258a892f9f7b32394bfd72940cebb516ea25fd4a"
  "src/hpc3/core/env_probe.py": "19d41901aac9ea756003760b58338d1b163a9c47"
fact_checked: 2026-09-01
confidence: high
---

# The environment is the pinned one, proven by its own interpreter

`env_path` proves a directory exists. `pinned_packages` proves what is in it:
preflight runs that environment's own interpreter and holds it to the
declared versions.

This is not hypothetical. `/pub/wagnera3/envs/abl` and
`/pub/wagnera3/envs/abl-pinned` both exist, both pass an existence check, and
they differ by transformers 4.46.3 vs 5.15.1 and torch 2.6.0+cu124 vs
2.11.0+cu128. Seven characters in a path, a major version underneath, and a
McNemar comparison against published arms that silently means nothing.

## "No pins" is an answer, not an omission

Declaring `{}` is allowed and deliberate — a project whose payload is a
compiled binary has no Python packages to pin — but the field is required, so
the empty map is a statement someone made rather than a default nobody
noticed. Until 2026-10-05 the statement had a cost: an empty pin map made no
interpreter round trip at all, which left a JVM project with only `test -d`
on its environment ([[unsupported-shapes]]). It no longer does. The probe now
opens by asking the interpreter for its version and `sys.base_prefix`, and
`verify_environment` makes the round trip whatever the pins say, because the
interpreter check is not about packages: it refuses an environment running
another installation's Python ([[interpreter-availability]]). An empty map
now means "no package is held to a version", and nothing more.

## The refusal, verbatim

```
$ hpc3-preflight --config hpc3.json --run runs/arm-b.json
ENV_PACKAGE_MISMATCH: /pub/wagnera3/envs/abl has torch==2.11.0+cu128, but this
project pins torch==2.6.0+cu124. A version difference under a published
comparison is a confound, not a detail.
$ echo $?
2
```
