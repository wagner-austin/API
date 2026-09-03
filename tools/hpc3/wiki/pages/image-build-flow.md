---
title: The image build flow, and why EVERY registered project declares an image
tags: [images, identity, provenance, onboarding]
hubs: [images-and-staging]
related: ["[[image-ledger-lessons]]", "[[environment-pins]]", "[[known-answers]]", "[[capture-source-drift]]"]
source_paths:
  - "src/hpc3/contracts/project.py"
  - "src/hpc3/contracts/workspace.py"
  - "src/hpc3/contracts/image_spec.py"
  - "src/hpc3/contracts/image_spec_fields.py"
  - "src/hpc3/cli/image.py"
  - "src/hpc3/cli/image_build.py"
  - "src/hpc3/cli/image_capture.py"
  - "src/hpc3/core/env_probe.py"
  - "src/hpc3/core/image_capture.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/contracts/project.py": "c2ff0d9ae27e41570308a457bd875b6c3acb0251"
  "src/hpc3/contracts/workspace.py": "843e854bdd634c97029050366e3381ea5a48aac6"
  "src/hpc3/contracts/image_spec.py": "26354bdfe7ae4f19d4bac5fa6936f851010f93e2"
  "src/hpc3/contracts/image_spec_fields.py": "88d23d5fc4d2646f89f75b1ad5c85d7df9c4c4b2"
  "src/hpc3/cli/image.py": "fa9f24c8972291940866c7fd0a790e5ec45f3215"
  "src/hpc3/cli/image_build.py": "690afc1d5b6cc9732ead9a3d619346b50f563643"
  "src/hpc3/cli/image_capture.py": "af0766bc1d4da403e2e566d6f0540bc3e1766f40"
  "src/hpc3/core/env_probe.py": "e83c330acd07bdb53dfdcc8fe1ee8a64de3af529"
  "src/hpc3/core/image_capture.py": "dc635e24b966209acc57e44e0f130c55015a9417"
  "README.md": "f52e3c3fc49ebeadf34f228748f207253ad726c0"
provenance:
  - "runs/hpc3*.json -- the six committed workspace documents; every project declares an image as of 2026-09-03"
  - "runs/ledger.jsonl -- image-build rows; earliest recorded build artifact is images/v23, while floor declares images/v4"
  - "registry-1.docker.io Docker-Content-Digest for python:3.11-slim-bookworm, read 2026-09-03: sha256:528257d4..., against the sha256:0bee7276... rusted pinned"
  - "/pub/wagnera3/api/libs/cleargbm_rs/target/wheels/cleargbm_rs-0.1.0-cp311-cp311-linux_x86_64.whl -- the built wheel on the cluster"
fact_checked: 2026-09-03
confidence: high
---

# The image build flow, and why EVERY registered project declares an image

**Every registered project must declare an `image`, and that image's base must
be pinned by digest.** Both are enforced when the workspace is decoded, so
neither is something a project can opt out of, and a run may override the
image but may not set it to `null`[^1][^2].

`ProjectConfig.image` is not an optional field. A project without one cannot
be decoded at all, so no reader downstream carries a branch for its absence —
the illegal state is unrepresentable rather than caught[^3].

## The CPU exemption was wrong, and it was also a template

This rule used to exempt CPU-only projects, and this page used to say so:
*"CPU-only projects may omit it. `cleargbm` has no card and no driver stack."*
That was wrong for the reason the GPU case was already right. **What an image
pins is not the card.** It is the compiler, the libc and the BLAS as much as
the CUDA runtime, and none of those appear in `env_path` or in a Python
package list[^1].

`cleargbm` is the demonstration rather than the exception. Its headline is a
*timing* claim against LightGBM, and the arm being timed is compiled Rust
(`cleargbm_rs`) — so neither a directory path nor a package list describes
what produced a benchmark number[^4].

The exemption also worked as a template. `turkic-lstm` was onboarded unimaged
in August by copying the shape from the project that was *allowed* to be
unimaged. An exemption that exists is an exemption that gets copied to where
it does not apply, which is why the fix was to delete it rather than to
document it more loudly[^1].

## A tag is not a pin

`base_image` must carry an `@sha256:` digest. A tag is a mutable pointer the
publisher can move, so two builds of one spec a week apart can start from
different bytes and neither says so — the same argument `system_packages`
already makes about an unpinned `apt-get install`[^5].

This is not hypothetical. `rusted` pinned
`python:3.11-slim-bookworm@sha256:0bee7276…`; that tag now resolves to
`sha256:528257d4…`. **The tag moved under four specs that named it bare and
nothing in the workspace noticed**, because nothing was looking[^6].

The digest is *required*, not resolved. Resolving would make reading a
document depend on a registry being reachable, and would silently re-pin a
spec every time the tag moved — which is the failure being prevented. Get it
with `docker buildx imagetools inspect <ref>`, or from the registry's
`Docker-Content-Digest` header[^5][^6].

## The wheel tag is read, never assumed

Capture used to synthesise every first-party wheel filename as
`py3-none-any`. That is right for a pure-Python distribution and wrong for a
compiled one, and the constant's own docstring named the bound it would
eventually cross[^7].

`cleargbm_rs` crossed it. Its real wheel is
`cleargbm_rs-0.1.0-cp311-cp311-linux_x86_64.whl`, so the captured spec named a
file that has never existed and the build would have failed on a missing path.
The probe now reports each distribution's tag out of its own `WHEEL`
metadata[^7][^8].

A **first-party** distribution reporting no tag is refused with
`WHEEL_TAG_UNKNOWN`: it was not installed from a wheel — conda, or an
editable checkout — so there is no filename to name. A **third-party** one
with no tag is fine, because it becomes a requirement line the build resolves
from an index, and a conda-installed package legitimately has no `WHEEL` file[^8].

## Four commands, in order — and a step zero

**`hpc3-bootstrap`** comes before all four, and only for a project that does
not have an environment yet. Capture PROBES a live environment; a newcomer has
nothing to probe, so this list used to begin one step after the beginning and
that step was improvised every time. It creates the environment through
`miniconda3` — the cluster's `python` modules are 2.7, 3.8, 3.10 and 3.14, none
of which is the 3.11 everything here needs — and then refuses to return one
whose interpreter belongs to another installation
([[interpreter-availability]]).

An established project skips it. The four below are the recurring flow.

1. **`hpc3-image-capture`** reads a live environment and writes the spec. Do
   NOT hand-write one: pip-freeze-and-paste is unrepeatable and silently
   incomplete, which is how the first spec got made. It records the project
   as a **field**, not a label.
2. **`hpc3-image`** renders the spec into definition, requirements,
   self-check and build script. Pure; builds nothing. It takes **no
   `--project` and no `--config`** — the project half of the job name comes
   from the spec, so the renderer cannot be handed a different one
   ([[image-ledger-lessons]]).
3. **`hpc3-stage`**, or `scp`, puts the rendered files AND the first-party
   wheels into the IMAGE directory — `build.sbatch` does `cd <image-dir>` and
   runs `build.sh` there. Anywhere else fails with
   `build.sh: No such file or directory`.
4. **`hpc3-image-build`** preflights, submits, and writes the ledger row.
   ~25 minutes on `free`, which is `PreemptMode=CANCEL`, so the rendered
   `--requeue` is inert there and a preempted build is simply gone; re-run
   the command, it is idempotent apart from the job id.

Then put the built image's path and `sha256` in the project's `image`, set
`env_path` to the in-image prefix (`/opt/env`), and preflight: the pinned
packages are checked **inside the image** rather than against a directory
someone can edit[^9].

## Onboarding, and the deadlock it used to be

Registration needs a digest. The digest comes from a build. The build is
driven by a spec that capture writes by probing a live environment. Every
command in that chain used to decode the **whole** workspace to reach one or
two strings — so a project mid-onboarding, which by definition has no image
yet, was refused by all four, including the one whose output would have fixed
it[^10].

The fix is a split by role rather than an exemption anyone can declare[^10][^11]:

| reads | commands |
|---|---|
| the connection only (host, ledger, cluster) | `hpc3-stage` |
| the connection, and nothing from the registry | `hpc3-image`, `hpc3-image-build` |
| the full registry | everything that submits real work |

`hpc3-image-capture` takes `--env-path` for onboarding: it reads only the
connection and probes the named host directory. Without that flag the project
must be registered, and the probe runs inside the image registration
guarantees — the version-bump case, which was always the recurring job[^12].

`hpc3-image-build`'s old registry lookup turned out to be **redundant as well
as blocking**. `submit_build` already calls `check_name_agrees` against the
rendered script's own job name, before preflight, so a mistyped project is
refused there — and that check is strictly stronger, because it also catches
a renderer and a submitter that have drifted apart. Agreement across
artifacts replaces membership in a table, and unlike the table it still
answers while a project is being onboarded[^13].

## CPU-only is a single spelling

`"gpu": null` is how a CPU-only job is stated, and it is the only way — there
is no zero-count request, because two spellings of one state is how they
drift apart. The partition must agree in both directions
([[submission-rules]]). What has changed is only that CPU-only no longer
implies *unimaged*.

## What is still open

Nothing checks that a project's declared image was actually **built by a job
this ledger recorded**. Such a gate would work for `mi`, `rusted`, `tankpit`
and `turkic-lstm` — and `floor` could not satisfy it. The ledger's earliest
recorded build is `v23`; `floor` declares `v4`, from the era when builds went
through raw `sbatch` and left no row at all. The evidence was never written
and cannot be created retroactively, so enforcing the gate means `floor`
rebuilds first[^14].

Onboarding also cannot check a project NAME against the registry, because the
project is not in the registry yet — that is inherent, not an oversight. The
name is validated for well-formedness, so it can be wrong but not malformed[^12].

[^1]: `src/hpc3/contracts/project.py` section `_require_project_image` -- the refusal, and the recorded reasons the CPU exemption was removed.
[^2]: `src/hpc3/contracts/workspace.py` and `src/hpc3/contracts/run.py` -- a run may pin a newer image, never null it.
[^3]: `src/hpc3/contracts/project.py` section `ProjectConfig.image`, typed `ImageReference` rather than `ImageReference | None`.
[^4]: `runs/hpc3.json` -- cleargbm's entry, and the compiled `cleargbm_rs` arm its timing claim measures.
[^5]: `src/hpc3/contracts/image_spec_fields.py` sections `DIGEST_SEPARATOR` and `require_digest_pinned_image`.
[^6]: `specs/rusted-image.json` `base_image` against the registry's Docker-Content-Digest for the same tag, read 2026-09-03 (see `provenance`).
[^7]: `src/hpc3/core/image_capture.py` section `_first_party_wheel`; the removed `WHEEL_TAG` constant is described there in past tense.
[^8]: `src/hpc3/core/env_probe.py` sections `_PROBE_SOURCE` and `InstalledDistribution`.
[^9]: `README.md` -- the command table and the preflight step.
[^10]: `src/hpc3/contracts/workspace.py` sections `WorkspaceConnection` and `decode_workspace_connection`.
[^11]: `src/hpc3/cli/image.py` and `src/hpc3/cli/image_build.py` -- neither loads the workspace registry.
[^12]: `src/hpc3/cli/image_capture.py` section `_environment_to_probe`, and its module docstring "TWO ROUTES".
[^13]: `src/hpc3/cli/image_build.py` section `main`, where the registry lookup was removed because `check_name_agrees` runs first.
[^14]: `runs/ledger.jsonl` image-build rows against the `image` each project declares in `runs/hpc3*.json` (see `provenance`).
