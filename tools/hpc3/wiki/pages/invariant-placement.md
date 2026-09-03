---
title: Where an invariant belongs, and what happens when it sits too deep
tags: [submission, contracts, design, refusals]
hubs: [submission]
related: ["[[submission-rules]]", "[[image-build-flow]]", "[[environment-pins]]", "[[unsupported-shapes]]"]
source_paths:
  - "src/hpc3/core/preflight.py"
  - "src/hpc3/contracts/project.py"
  - "src/hpc3/cli/_config.py"
source_git_blobs:
  "src/hpc3/core/preflight.py": "c642109e539eb3a0a7cfb97c6e44ff161fe7cb0a"
  "src/hpc3/contracts/project.py": "c2ff0d9ae27e41570308a457bd875b6c3acb0251"
  "src/hpc3/cli/_config.py": "81b8e8ee8fc1a7f8a747da7fa481fee84b114a71"
provenance:
  - "PROJECT_UNIMAGED observed raised from contracts/workspace.py at decode, 2026-09-02, while that file was uncommitted work in progress"
  - "decode_workspace observed reached from cli/_config.py, the shared loader, 2026-09-02"
  - "811c64cb landed the fix on 2026-09-03 by splitting the reader, not by moving the rule; this page's original prescription was superseded and is corrected in place"
  - "hpc3-image-capture --config runs/hpc3-tankpit.json --project tankpit refused with PROJECT_UNIMAGED, 2026-09-02"
  - "hpc3-preflight on the same project after its image was built: OK tankpit.tankpit-sim-smoke, 2026-09-02"
fact_checked: 2026-09-03
confidence: high
---

# Where an invariant belongs, and what happens when it sits too deep

An invariant about what a project may **do** belongs where doing happens.
Only an invariant about what a document **is** belongs in decode.

That one line is the whole page. The rest is the incident that produced it,
kept because the failure it caused was invisible until someone tried to
onboard a project and could not.

## The two layers this package already has

The package already separates the two questions, and the split is visible in
where each refusal is raised:[^2]

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
`pinned_packages` is edited to match. It was placed in `decode_workspace`.[^3]

`decode_workspace` is reached from the shared CLI config loader, so it runs
for *every* command.[^4] Including the three whose entire purpose is to
produce the image: capture, render, build. The result:

```
$ hpc3-image-capture --config runs/hpc3-tankpit.json --project tankpit ...
PROJECT_UNIMAGED: Every registered project must declare an 'image'. ...
    Build one with hpc3-image-build, or adopt an existing one with
    hpc3-image-capture then hpc3-image
```

The error instructs the reader to run the command that just refused them.
**To obtain an image you must already have one.** Every existing project was
unaffected, because every existing project already had its image; the rule
was invisible to everything except the one case it made impossible.[^5]

It also refused a committed workspace. `runs/hpc3.json` declares `cleargbm`,
which has no image, so that document stopped decoding — and because
`hpc3-research-index` reads every workspace, the ordinary invocation of the
generator fails outright. One unimaged project takes a tool down for all
six.[^6]

The table is still *recoverable*, which is worth stating so nobody reads the
above as data loss: running the generator against a contract without the rule
regenerates it correctly. That is a workaround available to someone who
knows the rule is new and uncommitted, not a property anyone should rely
on.[^7]

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

**What this page originally prescribed was to move the check to the submit
path, beside `ENV_PATH_MISSING`. That is not what shipped, and what shipped is
better.** Commit `811c64cb` fixed it by splitting the READER instead of
relocating the rule.[^8]

The invariant stays in decode. What changed is that the onboarding path no
longer decodes the project registry at all:[^9]

| loader | reads | used by |
|---|---|---|
| `load_workspace_connection` → `WorkspaceConnection` | cluster, host, root, ledger, quiet_seconds | the onboarding/capture path |
| `load_workspace` → `Workspace` | all of that **plus the project registry** | everything else |

`PROJECT_UNIMAGED` now lives in `contracts/project.py`, raised from
`require_image`.[^9] Capture never reaches it, because capture asks for a
connection and a connection has no projects in it.

**Why that beats the move.** Relocating the rule to the submit path would have
left every command still reading the registry, with the refusal merely firing
later — the onboarding command would still decode a project it cannot satisfy,
and would be spared only by the accident that nothing checked it yet. Under
the split, the onboarding path cannot trip the rule because it never loads the
thing the rule is about. And the distinction is carried by two **types**
rather than by a convention about which layer calls what, so a new command
picks its guarantee by choosing a return type and cannot silently pick wrong.

**It is still a move, not an addition.** The decode-level check on the shared
loader is gone rather than deprecated in place; nothing forwards from the old
site, and no flag preserves the old behaviour. A shim here would reintroduce
exactly the exemption the rule exists to deny.

## The general test

Before adding a refusal, ask which question it answers.

- *Is this document readable?* → decode. Shape, types, required fields,
  internal contradictions.
- *May this project act?* → the submit path. Anything about the world outside
  the document: does the path exist, does the digest resolve, is the
  environment what it claims.

And a third question this incident added, which is the one the adopted fix
actually answers: *does this caller even need that part of the document?* A
rule that only bites callers who had no business reading the field is not
mislocated — the READING is.

A rule of the second kind placed in the first location will not fail
loudly. It will pass for every project that is already finished and refuse
only the ones still being built — so it looks correct precisely until
somebody tries to start something.

[^1]: `src/hpc3/core/preflight.py` — `ENV_PATH_MISSING` is raised there, and in `core/array_submit.py` and `core/image_exec.py`, all on the submit path.
[^2]: `grep -rn ENV_PATH_MISSING src/hpc3/` → `core/preflight.py`, `core/array_submit.py`, `core/image_exec.py`; `grep -rn PROJECT_UNIMAGED src/hpc3/` → `contracts/workspace.py` only. Measured 2026-09-02.
[^3]: Observed in `src/hpc3/contracts/workspace.py` on 2026-09-02 while that file was uncommitted work in progress; recorded under `provenance:` rather than pinned, since a blob pin on a live edit would be stale within the hour.
[^4]: `grep -rln decode_workspace src/hpc3/cli/` → `_config.py` (the shared loader) and `research_index.py`. Measured 2026-09-02.
[^5]: `hpc3-image-capture --config runs/hpc3-tankpit.json --project tankpit --commit bfdce7a5 --base-image python:3.11.16-slim-bookworm --env-prefix /opt/env --first-party platform_core,monorepo_guards,tankpit_bot --out specs/tankpit-image.json`, run 2026-09-02, refused with `PROJECT_UNIMAGED`.
[^6]: Decoding each committed workspace individually on 2026-09-02: `hpc3-floor.json`, `hpc3-mi.json`, `hpc3-rusted.json`, `hpc3-tankpit.json` and `hpc3-turkic-lstm.json` all OK; `hpc3.json` alone refused. `hpc3-research-index --write` failed with the same error.
[^7]: `tools/hpc3/src/hpc3/core/research_index.py`, regenerated 2026-09-02 by importing the generator from a `git archive` extract of HEAD, whose contract predates the rule, with `index_path` and `runs_directory` rebound to the real paths. All six rows restored and `tankpit`'s image digest picked up; one line changed.
[^8]: Commit `811c64cb`, "Every registered project ships an image, and onboarding can still make one" — 15 files, including a new 333-line `src/hpc3/contracts/project.py` and 63 added lines in `src/hpc3/cli/_config.py`. Read 2026-09-03.
[^9]: `src/hpc3/contracts/project.py:233` raises `Hpc3ErrorCode.PROJECT_UNIMAGED` from `require_image`; `src/hpc3/cli/_config.py:49` is `load_workspace_connection`, whose docstring states it leaves "the project registry unread" for the onboarding path. `grep -rn PROJECT_UNIMAGED src/hpc3/` now returns `contracts/project.py` and a docstring reference in `contracts/workspace.py`, and no longer the shared loader's decode path. Measured 2026-09-03.
