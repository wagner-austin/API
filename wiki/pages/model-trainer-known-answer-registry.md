---
title: The known-answer registry — three outcomes, and why an entry must prove it can fail
tags: [services, model-trainer, provenance, known-answers, reproducibility, platform-core]
related:
  - "[[model-trainer-run-record-provenance]]"
  - "[[model-trainer-cloze-scoring-path]]"
  - "[[model-trainer-service-architecture]]"
source_paths:
  - libs/platform_core/src/platform_core/known_answer.py
  - libs/platform_core/src/platform_core/known_answer_registry.py
  - services/Model-Trainer/src/model_trainer/cli/known_answer_registry.py
  - services/Model-Trainer/src/model_trainer/cli/known_answer_probe.py
  - services/Model-Trainer/src/model_trainer/core/services/model/known_answer_probe.py
source_git_blobs:
  "libs/platform_core/src/platform_core/known_answer.py": 676b6e318dc37593a24f385051a0ea64864ea6ea
  "libs/platform_core/src/platform_core/known_answer_registry.py": 881872750bd6916876d69e98b5f7303750ebae71
  "services/Model-Trainer/src/model_trainer/cli/known_answer_registry.py": 5f72b834c73127ac3fc2cb202e9a3217e69fbf6f
  "services/Model-Trainer/src/model_trainer/cli/known_answer_probe.py": b829b087c7c5e87513ae4037ac43f6527d1a0279
  "services/Model-Trainer/src/model_trainer/core/services/model/known_answer_probe.py": 4ed920572b58faa229cbcb15960546dc7e57f409
provenance:
  - "the registry file itself is ~/PROJECTS/wiki/tools/extraction-eval/runs/known-answers.json, in the personal wiki repository, outside this wiki's workspaceRoot. Read 2026-10-05 (file mtime 2026-08-29): 13 entries -- 11 labelled gpt2-tiny-L2-d128-h2-v512-len64-seed42, all expected 6.250983715057373 at tolerance 0.0, across V100, A30, A100 and L40S and three image digests (2b89283f..., ebb61ed0..., 55651342...); and 2 floors on the A100 under image 2b89283f..., gpt2-floor-2627items 0.523030072325847 and gpt2-medium-floor-2627items 0.5572896840502475. Every entry's determinism stack is torch."
  - "the hpc3 wiki's page known-answers (tools/hpc3/wiki/pages/known-answers.md) covers the same registry from the image-selfcheck side and records why it lives in the wiki repo"
fact_checked: "2026-10-05"
confidence: high
hubs: [services, libs]
---

# The known-answer registry

An image that still builds is not an image that still computes what it used
to. On this project a rebuilt image silently changed its torch major version
and it was found only after a training run nobody could interpret; a known
answer catches that in seconds, before anything is staged.[^ka]

## An answer is a value *under a configuration*

`KnownAnswer` is a `label`, the `fingerprint` it was established under, the
`expected` value and an absolute `tolerance`.[^ka] Tolerance zero means
bit-exact, which is the right default **within** one configuration once
determinism is pinned — and is exactly why moving configuration is a separate
outcome rather than a wider band.[^ka]

## Three outcomes, not two

`check_known_answer` compares the fingerprint **first**:[^check]

- `configuration_differs` — some axis differs from the answer's; no number is
  compared and none is reported, and the differing axes are named;
- `matches` — same configuration, deviation within tolerance;
- `deviates` — same configuration, deviation outside tolerance.

Collapsing the first into the third "would report a working image as broken
every time it moved to a different card, which trains everyone to ignore the
check".[^ka] The module decides; it does not enforce — refusing a submission
is the submitter's job.[^ka] The six axes are those of
[[model-trainer-run-record-provenance]].

## Registration must discriminate

`model_trainer.cli.known_answer_registry` has two named modes. `--mode gate`
checks a record against the registry and changes nothing. `--mode register`
adds an entry **only if it can fail**: `discrimination_failures` runs three
controls — the entry must match its own measurement, must *deviate* on a drift
of `1e-9`, and must report `configuration_differs` when only the card is
swapped for `SYNTHETIC CONTROL CARD`.[^cli] Matching its own measurement
alone is "very nearly circular"; an entry that cannot fail passes everything
forever, silently.[^cli]

Two more refusals live in the collection layer, each from an incident:[^reg]

- **no empty fingerprint axis** — unknown differs from every real value, so
  such an entry could never match again; the first registered probe nearly
  carried an empty `driver_version` because its batch prologue, not the
  process, had recorded the card;
- **the file is written indented**, so a new entry is a readable diff rather
  than a rewrite of one long line.

## The probe that feeds it

`known_answer_probe` runs a tiny randomly-initialised GPT-2 forward pass and
reports its loss, in seconds and with nothing staged. It takes its fingerprint
from `capture_run_fingerprint` — the same function the scorer uses — and does
**not** pin determinism itself, because a pin inside the probe would come
after the cuBLAS handle already exists.[^probe]

What the registry holds today is in provenance: eleven probe entries that
agree **bit-for-bit** at 6.250983715057373 across four GPU models and three
images, and the two cloze floors that [[model-trainer-cloze-scoring-path]]
produces. Because each entry is pinned to its card and image, a new image
starts as `configuration_differs` everywhere until a run on it is registered.[^check][^ka]

[^ka]: `libs/platform_core/src/platform_core/known_answer.py` — module docstring ("SO THERE ARE THREE OUTCOMES, NOT TWO"), `KnownAnswer`.
[^check]: `libs/platform_core/src/platform_core/known_answer.py` / `check_known_answer`, `AnswerMatches`, `AnswerDeviates`, `AnswerNotApplicable`.
[^cli]: `services/Model-Trainer/src/model_trainer/cli/known_answer_registry.py` — module docstring ("TWO MODES, NAMED RATHER THAN INFERRED"), `discrimination_failures`, `_CONTROL_DRIFT`, `_CONTROL_CARD`.
[^reg]: `libs/platform_core/src/platform_core/known_answer_registry.py` — module docstring ("WHY IT IS NOT JUST json.load").
[^probe]: `services/Model-Trainer/src/model_trainer/cli/known_answer_probe.py` (module docstring); `services/Model-Trainer/src/model_trainer/core/services/model/known_answer_probe.py` / `probe_forward_loss`.
