# CRSP protocol version-5 runtime evidence

`evidence.json` records, for board task `f963ee0b`, the licensed version-5
metadata revision, the separate version-5 controller, the native image and the
private train-validation staging, all from API
`cfd829d03590ea57c247057e27a48510ab2b2078` after the version-5 peer review.

No licensed observation or key is in this directory. The revision changed only
protocol and manifest metadata and retained all sixteen relation ciphertexts;
the HPC3 key directory holds the train-validation key alone.

Every file here is a measured input or a program-written report:

- `revision-create.json`, `revision-verify.json`: the two identical revision
  reports, also posted by the service as load reports.
- `controller-provenance.json`, `controller-report.json`: the installed
  controller's frozen wheel digests and its measured integrity.
- `image-spec.json`: the version-3 spec with the new commit and the required
  symbols version 5 changed; `capture-config.json`: the cluster workspace.
- `build-stage.json`, `wheels-stage.json`, `frozen-inputs.sha256`,
  `image-expected.sha256`, `image-report.json`: build inputs, their independent
  digest record and the image inspection.
- `validation-stage.json`, `validation-stage-without-returns.json`,
  `validation-expected.sha256`: the encrypted partition staging. The expected
  record comes from `bundle.json`'s declared ciphertext digests and the
  verified revision report, not from the staged copies.

`evidence.json` `staging.deviation` states how `returns.csv.age` reached HPC3:
the per-command 600-second transfer bound stopped it, so the byte-identical
version-3 staged copy was copied on the cluster and checked there.

This is a runtime record, not an audit of it and not a qualification result.
