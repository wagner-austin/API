---
title: Art-Trainer publishes LoRAs that a different repository consumes
tags: [services, art-trainer, lora, comfyui, image-generation, cross-repo]
related:
  - "[[monorepo-discipline]]"
  - "[[service-port-map]]"
source_paths:
  - services/Art-Trainer/README.md
  - services/Art-Trainer/src/art_trainer/core/services/deployment/lora_deployer.py
  - services/Art-Trainer/pyproject.toml
source_git_blobs:
  "services/Art-Trainer/README.md": 13d03b8f6eb7a340b4aceca1c85126582e14fd20
  "services/Art-Trainer/src/art_trainer/core/services/deployment/lora_deployer.py": cd8e99c89800fc18240d8582378a0a5cad67c63d
  "services/Art-Trainer/pyproject.toml": 40124a973a5eb4359b5c040aca2130e49c2b7aa8
provenance:
  - "THE CONSUMER IS IN ANOTHER REPOSITORY and therefore cannot be a source_path here: ~/PROJECTS/chat, whose workspaceRoot this wiki does not cover. Cited by path and commit instead. src/chat/domain/lora_provenance.py declares the LoraProvenance TypedDict (filename, base_model, training_name) and the sd15/sdxl/flux1/unknown family literal; src/chat/infra/lora_metadata.py reads it off disk."
  - "chat commit 3702b89 (2026-09-11) added the cross-repo sentence to that repo's README. Before it, NEITHER repository's README named the other."
  - "chat commit c3734f2 (2026-09-11) corrected docs/image-generation-roadmap.md, which had listed 'LoRA training pipeline' under 'What's missing' while this service existed and shipped."
  - "NOT A RESEARCH SURFACE, measured 2026-09-11 rather than assumed: services/Art-Trainer/runs/ is EMPTY (0 entries) and no file under its src/ references RunRecord, run_record or RunFingerprint. That is why it is absent from docs/RESEARCH.md and why it belongs here instead -- it produces ARTIFACTS, not numbers anyone subtracts."
fact_checked: "2026-09-11"
confidence: high
hubs: [services]
---

# Art-Trainer publishes LoRAs that a different repository consumes

`services/Art-Trainer` trains LoRA adapters and writes them into ComfyUI's
models directory. It is not the thing that uses them: the consumer is the
`chat` repository, which applies them at inference and never trains anything.
**The two halves live in different git repositories and, until 2026-09-11,
neither one's README mentioned the other.**

That is the fact this page exists for. Everything below is downstream of it.

## What it does

Per the service's own README[^1]: LoRA training for SD 1.5, SDXL and FLUX;
style, character and concept presets; dataset management with multi-backend
auto-captioning (BLIP locally, Google Gemini, OpenAI GPT-4o); Kohya-ss /
sd-scripts as the training engine; durable jobs on Redis + RQ with progress,
cancellation and retry; and Data-Bank integration for datasets and artifacts.

Its service layer is organised by that list — `captioning/`, `dataset/`,
`queue/`, `training/`, `deployment/` — under `core/services/`[^2].

## The seam, and why it is invisible from either side

`core/services/deployment/lora_deployer.py` is the whole cross-repo contract,
and it is a filesystem copy rather than an API call[^2]:

```python
class DeploymentResult(TypedDict, total=True):
    success: bool
    source_path: str
    deployed_path: str | None
    error_message: str | None
```

It copies a trained `.safetensors` into ComfyUI's models directory. ComfyUI
then serves it to anything that asks, and `chat` asks.

**So there is no import, no HTTP call, and no shared package between producer
and consumer.** The coupling is a directory on disk. Nothing in either
repository's dependency graph records it, which is exactly why a reader of
`chat` could not discover this service, and a reader of this service could not
tell who consumed its output.

The measured cost: `chat`'s own `docs/image-generation-roadmap.md` listed
"LoRA training pipeline" under **What's missing** while this service was
built and shipping, and a session reading that document proposed building one
from scratch[^3].

## What the consumer does that this service does not

`chat/domain/lora_provenance.py` models what a LoRA declares about itself —
`filename`, `base_model` (`sd15` / `sdxl` / `flux1` / `unknown`) and
`training_name`[^4]. Two details are load-bearing:

- **`training_name` is this service's published name**, kept because the file
  may be renamed on disk afterwards. It is the only field that survives the
  hand-off as evidence of where the adapter came from.
- **`unknown` is a first-class value, not a failure.** Adapters published
  without provenance metadata are common and their family genuinely cannot be
  determined.

The check exists because **ComfyUI validates that a `lora_name` exists on disk
but not that its family matches the checkpoint**, so a mismatched adapter
loads without error and then discards every weight[^4]. A silent no-op is the
worst available failure, and the consumer catches it because the producer
cannot.

## Why this is not in `docs/RESEARCH.md`

Measured rather than reasoned: `runs/` is empty, and nothing under `src/`
references `RunRecord`, `run_record` or `RunFingerprint`[^5]. That index names
bodies of work producing numbers someone compares; this service produces
adapters. It has the shape of a research surface — a `runs/` directory, a GPU,
training jobs — and it is not one.

Stated because the resemblance is misleading in the direction that costs
something: a reader checking RESEARCH.md for "do we train image models" finds
nothing and may conclude we do not.

[^1]: `services/Art-Trainer/README.md` — the feature list, the supported
      architectures and the Kohya-ss integration, verbatim from the service's
      own documentation.
[^2]: `services/Art-Trainer/src/art_trainer/core/services/deployment/lora_deployer.py`
      — `DeploymentResult` and the copy into ComfyUI's models directory. The
      sibling service directories (`captioning/`, `dataset/`, `queue/`,
      `training/`) are the rest of `core/services/`.
[^3]: `chat` commits `c3734f2` and `3702b89`, 2026-09-11. The first rewrote
      that roadmap's Current State against an audit of the code; the second
      added the cross-repo link to `chat`'s README. Both are in the
      `provenance` block above, being outside this wiki's workspaceRoot.
[^4]: `chat/src/chat/domain/lora_provenance.py` — the `LoraProvenance`
      TypedDict, the `LoraBaseModel` literal, and the module docstring
      recording the ComfyUI behaviour this guards against. Outside this
      wiki's workspaceRoot; see `provenance`.
[^5]: Measured 2026-09-11 in `services/Art-Trainer`: `ls runs/` returns
      nothing, and `grep -rl "RunRecord\|run_record\|RunFingerprint" src/`
      matches no file. Both are negative results, so both are recorded with
      the command that produced them rather than asserted.
