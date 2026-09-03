---
title: Where an invariant belongs, and what happens when it sits too deep
tags: [submission, contracts, design, refusals]
hubs: [submission]
related: ["[[submission-rules]]", "[[image-build-flow]]", "[[environment-pins]]", "[[unsupported-shapes]]"]
source_paths:
  - "src/hpc3/core/preflight.py"
  - "src/hpc3/contracts/image_spec.py"
source_git_blobs:
  "src/hpc3/core/preflight.py": "c642109e539eb3a0a7cfb97c6e44ff161fe7cb0a"
  "src/hpc3/contracts/image_spec.py": "ef0e7c294cd0c8cca9e515605e0be486a6d77f78"
provenance:
  - "PROJECT_UNIMAGED observed raised from contracts/workspace.py at decode, 2026-09-02, while that file was uncommitted work in progress"
  - "decode_workspace observed reached from cli/_config.py, the shared loader, 2026-09-02"
  - "hpc3-image-capture --config runs/hpc3-tankpit.json --project tankpit refused with PROJECT_UNIMAGED, 2026-09-02"
  - "hpc3-preflight on the same project after its image was built: OK tankpit.tankpit-sim-smoke, 2026-09-02"
fact_checked: 2026-09-02
confidence: high
---

# Where an invariant belongs, and what happens when it sits too deep

An invariant about what a project may **do** belongs where doing happens.
Only an invariant about what a document **is** belongs in decode.

That one line is the whole page. The rest is the incident that produced it,
kept because the failure it caused was invisible until someone tried to
onboard a project and could not.

## The two layers this package already has

| question | layer |
|---|---|
| is this document well-formed — shapes, types, no contradictions? | `decode_workspace` |
| may this project *run work* — environment present, packages pinned, image resolvable? | `core/preflight.py` |

`ENV_PATH_MISSING` is a "not ready to run" check and it lives in preflight.[^1]
It has never deadlocked anything, and the reason is worth stating plainly:
**capture does not preflight.** The commands that build an image are free to
read a project that is not yet runnable, because the gate that would stop
them is one layer above.

## The incident

On 2026-09-02 a rule was added asserting that every registered project must
declare an `image`. The rule is correct and the reasoning behind it is sound
— a directory environment carries no digest and can be edited in place while
`pinned_packages` is edited to match. It was placed in `decode_workspace`.

`decode_workspace` is reached from the shared CLI config loader, so it runs
for *every* command. Including the three whose entire purpose is to produce
the image: capture, render, build. The result:

```
$ hpc3-image-capture --config runs/hpc3-tankpit.json --project tankpit ...
PROJECT_UNIMAGED: Every registered project must declare an 'image'. ...
    Build one with hpc3-image-build, or adopt an existing one with
    hpc3-image-capture then hpc3-image
```

The error instructs the reader to run the command that just refused them.
**To obtain an image you must already have one.** Every existing project was
unaffected, because every existing project already had its image; the rule
was invisible to everything except the one case it made impossible.

It also refused a committed workspace. `runs/hpc3.json` declares `cleargbm`,
which has no image, so that document stopped decoding — and because
`hpc3-research-index` reads every workspace, the generated project table
could no longer be regenerated at all. One unimaged project took a tool down
for all six.

## Why the obvious repairs are worse

- **A `--allow-unimaged` flag** is the exemption the rule was written to
  refuse. An exemption that exists is an exemption that gets used.
- **A `status: provisioning` field** is the same defect wearing a lifecycle
  costume: it is *declarable*, so a project asserts its own compliance.
- **A carve-out naming capture inside decode** leaves decode knowing which
  command called it, which is the coupling that makes layers stop meaning
  anything.

The layer correction is the only repair that is not something a project can
say about itself.

## The correction

Move the check to the submit path, beside `ENV_PATH_MISSING`. Every property
the rule was written for survives: no project runs unimaged, no exemption
field exists, `cleargbm` is still refused — at submission, which is where the
refusal was always meant to bite. What changes is that the commands which
produce an image may read a project that does not yet have one, which is the
only way a first image can ever exist.

**This is a move, not an addition.** The decode-level check is deleted
outright rather than deprecated in place; nothing forwards from the old site
to the new one, and no flag preserves the old behaviour. A shim here would
reintroduce exactly the exemption the rule exists to deny.

## The general test

Before adding a refusal, ask which question it answers.

- *Is this document readable?* → decode. Shape, types, required fields,
  internal contradictions.
- *May this project act?* → the submit path. Anything about the world outside
  the document: does the path exist, does the digest resolve, is the
  environment what it claims.

A rule of the second kind placed in the first location will not fail
loudly. It will pass for every project that is already finished and refuse
only the ones still being built — so it looks correct precisely until
somebody tries to start something.

[^1]: `src/hpc3/core/preflight.py` — `ENV_PATH_MISSING` is raised there, and in `core/array_submit.py` and `core/image_exec.py`, all on the submit path.
