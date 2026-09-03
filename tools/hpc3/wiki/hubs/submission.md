# Submission

How work reaches the cluster and what is refused before it can: run and sweep
documents, campaign convergence under preemption, chained stages, the
submission rules, and the shapes the contract deliberately cannot express.

[Run documents say what is specific to this run](../pages/run-documents.md) -- the experiment block, overrides through the real decoder, refused unknown fields
[Sweeps, and why every member declares its artifact](../pages/sweeps-and-artifacts.md) -- artifacts as identity, sweep-vs-campaign, ceiling checks
[Preemption cancels, checkpoints protect, campaigns converge](../pages/preemption-and-campaigns.md) -- PreemptMode=CANCEL measured, the resume-document race, artifact-keyed convergence
[The submission rules, each with the failure it refuses](../pages/submission-rules.md) -- the eight resolve-time rules and the non-skippable preflight
[Chains stop when a stage fails](../pages/chains.md) -- kill-on-invalid-dep, whole-pipeline budgeting, chain-vs-sweep
[A sweep is one sbatch call, and the script is the member table](../pages/job-arrays.md) -- arrays as the sweep transport, sparse campaign resubmission, the measured pending-aggregate identity rules
[Where an invariant belongs, and what happens when it sits too deep](../pages/invariant-placement.md) -- decode answers "is this document readable", the submit path answers "may this project act"; the image rule placed at decode made the first image unobtainable and took a committed workspace down with it
[What this package cannot submit](../pages/unsupported-shapes.md) -- multi-node, arrays, qos, constraints; the two shapes that left the list
