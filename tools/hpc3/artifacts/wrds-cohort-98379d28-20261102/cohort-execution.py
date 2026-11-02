"""Run the published cohort owner and record native metadata without market observations."""

import sys
from hashlib import sha256
from pathlib import Path

from platform_core.determinism_cpu import pin_single_thread

determinism = pin_single_thread()

from platform_core.comparability import encode_run_fingerprint
from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, dump_json_str, require_dict, require_str
from platform_core.run_record import run_record_sidecar

from covenant_ml.benchmarking.harness import write_observed_manifest
from covenant_ml.markets import crsp_cohort
from covenant_ml.markets.crsp_cohort_records import decode_crsp_cohort_plan
from covenant_ml.markets.encrypted_storage import verify_encrypted_bundle
from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.research_keys import EvaluatorKeys, decode_research_access
from covenant_ml.markets.validation import read_document

request = read_document(Path(sys.argv[1]))
output = Path(sys.argv[2])
assert not output.exists() and not run_record_sidecar(output).exists()
scratch = config_test_hooks.get_env("TMPDIR")
assert scratch is not None
plan = decode_crsp_cohort_plan(request["plan"])
owner_digest = sha256(Path(crsp_cohort.__file__).read_bytes()).hexdigest()
assert owner_digest == require_str(request, "cohort_owner_sha256")
keys = EvaluatorKeys(decode_research_access(read_document(Path(require_str(request, "access")))))
destination = Path(require_str(request, "destination"))
measurement = crsp_cohort.derive_crsp_cohort(
    Path(require_str(request, "bundle")),
    Path(require_str(request, "audit")),
    plan,
    keys,
    scratch=Path(scratch),
    destination=destination,
)
bundle = verify_encrypted_bundle(
    destination / "bundle.json",
    require_str(require_dict(measurement, "bundle"), "sha256"),
)
assert bundle["partition"] == "train-validation"
assert bundle["manifest"]["name"] == "cohort.json"
assert measurement["plaintext_removed"] is True
measurement["ciphertext_verified"] = True
measurement["native_job_id"] = config_test_hooks.get_env("SLURM_JOB_ID")
measurement["source_commit"] = require_str(request, "source_commit")
measurement["source_archive_sha256"] = require_str(request, "source_archive_sha256")
measurement["program_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
measurement["loaded_owner_sha256"] = owner_digest
fingerprint = market_fingerprint(determinism, config_test_hooks.get_env)
document: JSONObject = {
    "results": [measurement],
    "fingerprint": encode_run_fingerprint(fingerprint),
    "dataset": {"sha256": plan["bundle_sha256"]},
}
write_observed_manifest(
    document,
    fingerprint,
    output,
    sha256(dump_json_str(request).encode()).hexdigest(),
    "crsp-diagnostic-cohort",
    (),
)
print("Encrypted diagnostic cohort verified; private plaintext removed.", flush=True)
