# Encrypted CRSP protocol revision evidence

`evidence.json` records the real metadata revision and separate readback on
Ziggurat at API `b69daf97a0747732787d2036694b2c8d84e48c06`. It includes the
service-written report IDs, original extraction identity, revised bundle and
manifest identities, unchanged extraction vintage and test verdict.

No licensed observations or keys are included here. All sixteen relation
CSV/SQL ciphertext files were retained unchanged. The new protocol and manifest
were encrypted to the original public recipients, with the original extraction
and failed-study journal preserved.

The existing `scripts.crsp_extraction` commands `revise-protocol` and
`verify-revision` take the same explicit arguments:

- `--outputs`: original collected extraction output directory.
- `--receipt-sha256`: the original receipt digest in `measurement.revision`.
- `--previous-protocol`: `tools/hpc3/artifacts/wrds-baseline-freeze-20261031/protocol.json`.
- `--protocol`: `libs/covenant_ml/docs/crsp-monthly.json` at the recorded commit.
- `--access`: the evaluator's existing public recipient configuration.
- `--destination`: an exclusive new root for creation, that same root for readback.
- `--report-task`: the full board task UUID.

Both commands validate the originals. Readback verifies all original and new
ciphertext, reconstructs the manifests and checks the recorded lineage. Creation
refuses an occupied destination. Partial failed output is not a valid revision;
only a verified complete receipt is usable.

The execution used the existing controller's Python with checked source imports,
as recorded in `execution`; it did not install a new controller. HPC3 validation
staging, the version-2 runtime, a new preregistration and the complete queue
rehearsal remain separate work. Final observations remain unreleased.
