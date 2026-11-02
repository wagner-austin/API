"""Measure declared inputs and private scratch; never read licensed datasets."""

import sys
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from platform_core.comparability import encode_run_fingerprint
from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import pin_single_thread
from platform_core.json_utils import JSONObject, dump_json_str, require_dict, require_str

from covenant_ml.benchmarking.harness import write_observed_manifest
from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.validation import read_document


request = read_document(Path(sys.argv[1]))
output = Path(sys.argv[2])
mode = require_str(request, "mode")
assert mode in {"home", "shared"}
digests: JSONObject = {}
for name, value in require_dict(request, "inputs").items():
    assert isinstance(value, dict)
    path = Path(require_str(value, "path"))
    digest = sha256(path.read_bytes()).hexdigest()
    assert digest == require_str(value, "sha256"), name
    digests[name] = digest
mounts = []
for line in Path("/proc/self/mountinfo").read_text().splitlines():
    fields = line.split()
    filesystem = fields[fields.index("-") + 1]
    if "beegfs" in filesystem.lower():
        mounts.append(fields[4])
if mode == "home":
    assert not mounts, mounts
scratch = config_test_hooks.get_env("TMPDIR")
assert scratch is not None and Path(scratch).is_dir()
with TemporaryDirectory(prefix="mount-probe-", dir=scratch) as directory:
    private = Path(directory)
    assert private.stat().st_mode & 0o777 == 0o700
    payload = b"nonlicensed-node-scratch-probe\n"
    target = private / "payload"
    target.write_bytes(payload)
    assert target.read_bytes() == payload
    scratch_digest = sha256(target.read_bytes()).hexdigest()
assert not private.exists()
determinism = pin_single_thread()
fingerprint = market_fingerprint(determinism, config_test_hooks.get_env)
measurement: JSONObject = {
    "kind": "declared-mount-probe",
    "mode": mode,
    "native_job_id": config_test_hooks.get_env("SLURM_JOB_ID"),
    "controller_source": require_str(request, "controller_source"),
    "program_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_sha256": digests,
    "beegfs_mounts": list(mounts),
    "scratch_root": scratch,
    "scratch_mode": "0700",
    "scratch_sha256": scratch_digest,
    "scratch_removed": True,
    "scope": "Nonlicensed fixture reads and normal-exit scratch cleanup only.",
}
document: JSONObject = {
    "dataset": {"kind": "nonlicensed-mount-fixtures", "input_sha256": digests},
    "results": [measurement],
    "fingerprint": encode_run_fingerprint(fingerprint),
}
write_observed_manifest(
    document, fingerprint, output,
    sha256(dump_json_str(request).encode()).hexdigest(), "declared-mount-probe", (),
)
print("Declared inputs verified; private scratch removed.", flush=True)
