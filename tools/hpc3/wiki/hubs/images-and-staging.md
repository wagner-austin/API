# Images and staging

Identity, not transport: proving that what runs is the pinned thing —
container images and their build discipline, environment pins, staged-corpus
identity, the declared determinism posture, and the known-answer gate.

[The image build flow](../pages/image-build-flow.md) -- why EVERY registered project declares a digest-pinned image, the four commands in order, and how onboarding produces the first one
[Twenty-one unledgered image builds](../pages/image-ledger-lessons.md) -- the raw-sbatch era, v22's malformed name, why --job-name died
[The environment is the pinned one](../pages/environment-pins.md) -- abl vs abl-pinned, and why {} is an answer
[Staged bytes are held to a record written by a different act](../pages/staging-identity.md) -- --expect-from, and the provenance block
[Determinism is declared, split, and recorded](../pages/determinism-posture.md) -- the launcher/payload split and TRAIN_DETERMINISTIC
[An image that still builds is not an image that still computes](../pages/known-answers.md) -- the known-answer registry, its three outcomes and two invariants
[A capture source drifts from the image it produced](../pages/capture-source-drift.md) -- abl-pinned lost cupy while v31 kept it, the probe follows the image once one is declared, and three spec fields capture cannot carry forward
