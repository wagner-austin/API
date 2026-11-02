# Version-2 encrypted validation staging

This record concerns transport and complete remote readback of the revised
validation bundle. The original extraction and revision lineage are recorded in
`../wrds-protocol-revision-20261102/evidence.json`. The final bundle and private
keys are excluded from this transfer.

`evidence.json` retains three program-written readings: installed controller
verification, successful staging, and the separate complete tree inspection.
The latter measured eleven bundle files plus certificate, no extra entries,
and mode-0700 root and partition directories. The staged bundle totals
1,789,968,348 bytes.

The controller is the separately installed `wrds-summary-b3843a3d3` runtime.
Its staging, expected-digest and digest modules have no source differences from
API `715e159ac69deff5565a7a7763afd1bf58f5a2e1`; this says nothing about the
currency of its evaluator. Its frozen wheel, interpreter, selfcheck and
dependency inspection must pass before transport.

The shared `encrypted_stage_manifest` owns the eleven-file manifest. The
expected-digest record combines the frozen revision service report with the
bundle metadata pinned by that report. The stage manifest also names the
original extraction receipt and revision receipt in its provenance.

The existing `hpc3.cli.stage` is used twice:

1. With `--source-dir`, `--manifest`, `--expect-from`, `--config` and
   `--report-task` to transfer and certify the bytes.
2. With the same manifest and expected record, `--inspect-root` replacing
   `--source-dir`, to inspect every remote entry and its digest without writing.

The fresh destination root and its validation subdirectory are created as
mode 0700 before transport. Readback requires exactly the manifest files plus
the certificate, verifies each file and private directory, and refuses an
unexpected entry or a traversal that changes during inspection.

Staging is not an evaluator installation, a licensed validation result or a
final-scoring release. The old study journal and original staged directory
remain historical. Native study preparation and queue rehearsal follow
separately after the current runtime is installed.
