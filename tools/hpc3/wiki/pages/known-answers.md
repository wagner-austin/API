---
title: An image that still builds is not an image that still computes
tags: [identity, images, known-answers]
related: ["[[image-build-flow]]", "[[determinism-posture]]", "[[staging-identity]]"]
source_paths:
  - "src/hpc3/core/image_selfcheck.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/core/image_selfcheck.py": "14a72028e8f5fc4e80d5ce7742ba018f46f5b40c"
  "README.md": "c4cdcc31ae83beaede3c2635a943ddc0bcf0c083"
provenance:
  - "wiki/tools/extraction-eval/runs/known-answers.json (the wiki repo, not this one)"
  - "model_trainer.cli.known_answer_registry (services/Model-Trainer, outside this workspaceRoot)"
fact_checked: 2026-09-01
confidence: high
---

# An image that still builds is not an image that still computes

Nothing in the reproducibility standards checks the second: CWL validates the
description, MLflow describes entry points, and neither runs the tool and
compares the answer. On this project a rebuilt image silently changed its
torch major version, and it was found only after a training run whose result
could not be interpreted.

The registry lives at **`wiki/tools/extraction-eval/runs/known-answers.json`,
in git — in the wiki repo, not this one** — beside the ablation audit chain
whose floors it also carries. Version control is the point: it is the record
of what this environment computed and when, and a copy on cluster scratch is
one cleanup away from losing every baseline that makes a future run
interpretable. `write_registry` writes indented JSON for the sole reason that
a new entry should be a readable one-line diff, which only pays off under
version control.

**It briefly lived in the hpc3 package too, and that was a mistake worth
recording.** On 2026-08-29 a session searched this repo, `/pub/wagnera3`, the
cluster home directory and `artifacts/`, found no registry, and built a
second one — having written the first into the wiki repo hours earlier in the
same session. The duplicate reconstructed the original's six entries
byte-for-byte from the same artifacts through the same command, which is
exactly the signal that should have prompted a wider search. Four exhaustive
searches of the wrong namespace are not an absence proof; this workspace
spans three repositories plus a cluster filesystem.

## Three outcomes, not two

A value can match, deviate, or *not apply*. An expected loss is not a
property of an experiment; it is a property of an experiment on a particular
image, on a particular card, under particular determinism settings. Without
the third outcome, moving to a new GPU reports a working image as broken —
and everyone learns to ignore the check.

## Two invariants refuse an entry, and both have fired for real

- *No empty fingerprint axis.* An unknown axis differs from every real value,
  so such an entry could never match anything again. This is why the two
  RTX 3090 Ti cloze floors are **not** registered: measured on the
  workstation outside any image, their `image_digest` is `""`.
- *The entry must discriminate.* Registration also checks that a drifted
  value deviates and that the same value on another card does not apply.
  Verifying an answer against the measurement it was built from proves only
  that the checker can subtract. An entry that cannot fail is not a gate, and
  it fails silently — everything passes forever.

Tolerance is `0.0`: bit-exact is the right band *within* one configuration
once determinism is pinned ([[determinism-posture]]), which is exactly why
moving configuration is a separate outcome rather than a wider band.
