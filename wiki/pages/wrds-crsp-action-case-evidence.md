---
title: Native CRSP action-case evidence
tags: [wrds, crsp, diagnostics, provenance]
hubs: [libs]
related: ["[[wrds-crsp-action-case-inspection]]", "[[crsp-outcome-qualification]]"]
source_paths:
  - tools/hpc3/artifacts/wrds-action-cases-20261103/source.json
  - tools/hpc3/artifacts/wrds-action-cases-20261103/requests.json
  - tools/hpc3/artifacts/wrds-action-cases-20261103/native-reports.json
  - tools/hpc3/artifacts/wrds-action-cases-20261103/licensed-cases.json
  - tools/hpc3/artifacts/wrds-action-cases-20261103/licensed-case-coverage.json
  - tools/hpc3/artifacts/wrds-action-cases-20261103/licensed-metadata-review.json
  - tools/hpc3/artifacts/wrds-action-cases-20261103/synthetic-cases.json
  - tools/hpc3/artifacts/wrds-action-cases-20261103/synthetic-replay.json
source_git_blobs:
  tools/hpc3/artifacts/wrds-action-cases-20261103/source.json: 55b22a6734fe3647135002daf73681071b053c76
  tools/hpc3/artifacts/wrds-action-cases-20261103/requests.json: db447eb5f8d17646f9148970d4a9956b33e5ba8f
  tools/hpc3/artifacts/wrds-action-cases-20261103/native-reports.json: 034855cb48a0597da32a608c141a178bc41c0f15
  tools/hpc3/artifacts/wrds-action-cases-20261103/licensed-cases.json: a73933ae08cbd801e528eda5ac36d4b9133b2c90
  tools/hpc3/artifacts/wrds-action-cases-20261103/licensed-case-coverage.json: ebb276cf55c23b7b583ae17ec215c0196ea5a385
  tools/hpc3/artifacts/wrds-action-cases-20261103/licensed-metadata-review.json: ec155603cd0f1959c7072c4288314e6739c927e1
  tools/hpc3/artifacts/wrds-action-cases-20261103/synthetic-cases.json: e1e62f6f21eb71d5086cb533773c558bd8b09312
  tools/hpc3/artifacts/wrds-action-cases-20261103/synthetic-replay.json: 6f45e56f6b14777522a2cb3b49d93ec2d84456a6
fact_checked: 2026-11-03
confidence: medium
---

# Native CRSP action-case evidence

Two native WRDS jobs completed, were collected and received controller-owned
reports on November 3. This verifies diagnostic execution; independent review
of board task `4fca73b7` remains pending. Economic qualification remains open
under [[crsp-outcome-qualification]].[^native][^licensed]

## Frozen execution and synthetic replay

Both requests used source `89f23c71290e`, the existing WRDS Python 3.11.16
runtime and the existing native controller. Each requested two slots, 12 GB
per slot and 900 seconds; these are resource requests, not measured usage.
The frozen diagnostic source overlay is not a replacement study image.[^source]

Synthetic job `40569430` replayed the encrypted inputs of the successful fleet
child process. Its two-case output is byte-for-byte identical to that oracle:
SHA-256 `09a5dff90990e6be5c1c475f6071205ab435fc8f4298e5299deb3482e6968421`.
No final key was transferred. Entry/exit, event-only and late-detail assertions
ran separately in the passed fleet suite; this native replay made no database
query and did not use the staged setup SQL.[^synthetic]

## Licensed coverage and its limits

Licensed job `40569412` matched all 27 original refused cases: 25 training and
two validation. The authenticated capture has 19 delisting events and 328
distribution events, each with a detail row. Output SHA-256 is
`6ac30bf99f64c8bcff00f6d6d583b35b87bc4acf3403e42a6ddfcf2c3d053736`.[^licensed]

Eighteen cases have a captured action within the holding window or on exit;
sixteen have a delisting there. Four delistings are on entry and three on exit.
Eleven cases have a relevant detail with some dated field after exit. That
last count includes storage and next dates; it is not a count of late payments
or valuations. The other nine cases lack an event within that window or on
exit, which does not establish absence outside those boundaries.[^coverage]

The report contains hashes, counts, ordinals, relative offsets, booleans and
nulls, with no licensed identifiers, absolute dates or financial values.
Both outputs report normal plaintext cleanup and `outcomes_qualified: false`.
Neither is a performance score, settlement proof, forced-kill cleanup test or
permission to admit a new study. The report's meaning and authentication rules
are defined in [[wrds-crsp-action-case-inspection]].[^privacy][^licensed][^synthetic]

[^native]: `tools/hpc3/artifacts/wrds-action-cases-20261103/native-reports.json`, controller-owned board posts `a87ce79e-2836-4250-908e-f51467a2a6b5` and `715e543b-ef96-4051-aed4-77d7123f5417`, scheduler completed, accounting failed=0/exit_status=0, output and RunRecord digests. The excerpt omits individual code/input lines retained by the immutable posts.
[^source]: `tools/hpc3/artifacts/wrds-action-cases-20261103/source.json`, source commit, worker digest and frozen inventory; sibling `requests.json`, both requests' resources, runtime, arguments and inputs; sibling `native-reports.json`, measured runtime.
[^synthetic]: `tools/hpc3/artifacts/wrds-action-cases-20261103/synthetic-replay.json`, fixture provenance, expected/actual digest, exact byte comparison and boundary-test scope; sibling `synthetic-cases.json`, cases and cleanup/outcome fields.
[^licensed]: `tools/hpc3/artifacts/wrds-action-cases-20261103/licensed-cases.json`, original case anchors, periods, offsets, action counts and outcome field; sibling `native-reports.json`, licensed output digest.
[^coverage]: `tools/hpc3/artifacts/wrds-action-cases-20261103/licensed-case-coverage.json`, case counts, ordinal sets and definition, bound to the native output digest.
[^privacy]: `tools/hpc3/artifacts/wrds-action-cases-20261103/licensed-metadata-review.json`, exhaustive leaf-value inventory; sibling `licensed-cases.json` and `synthetic-cases.json`, plaintext_removed and outcomes_qualified.
