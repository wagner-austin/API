# CRSP version-5 native full-panel qualification

`evidence.json` records the native qualification of the licensed
train-validation partition under protocol version 5, for board task
`f963ee0b`, run twice with one sealed request (`request.json`) on the version-5
image recorded in `../wrds-protocol-v5-runtime-20261103/`.

Measured: 262,000 paths in 262 decisions, 14 refused. Every refused path is a
version-3 diagnosed path with the same entry and exit record digests; the 13
others version 3 refused are valued by version 5's explicit rules. The
full-universe target is therefore not qualified under version 5.

Files:

- `request.json`, `run-1.json`, `run-2.json`, `checksums-1.txt`,
  `checksums-2.txt`: the sealed request, the two native run documents and the
  in-job checksum gates. Only the second gate includes the image.
- `qualification-1-public.json`, `qualification-2-public.json`: the published
  results with the licensed extract's plaintext file identities removed; each
  names the full file's SHA-256 under `projection_of`.
- `qualification-1.runrecord.json`, `qualification-2.runrecord.json`: the
  shared run records, unchanged.
- `qualification-2-evidence-public.json`: the hpc3 execution inspector's report
  for the second attempt, projected the same way.

No licensed observation or key is in this directory. A refused path is
identified only by its walk ordinal, decision index, period, reason and the
unsalted digests of its entry and exit records, as in the version-3 diagnosis.
