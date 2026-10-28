"""Execute one admitted validation diagnostic using the existing paired evaluator."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint, encode_run_fingerprint
from platform_core.json_utils import JSONObject

from ..benchmarking import make_baseline_trainers
from ..benchmarking.declarations import PAIRED_EVALUATION
from ..benchmarking.harness import declared_observations, results_of, write_observed_manifest
from ..benchmarking.types_codec import encode_benchmark_config, encode_dataset_info
from ..evaluation.codec_records import encode_arm_result, encode_seed_comparisons
from ..evaluation.provenance import policy_observations
from ..evaluation.runner import evaluate_seed
from ..evaluation.wrds import load_wrds_evaluation
from .research_journal import research_policy_digest
from .research_limits import policy_refusal
from .research_records import candidate_digest
from .research_trial_records import ResearchTrial, encode_research_evaluation, evaluation_digest


def execute_trial(
    trial: ResearchTrial, fingerprint: RunFingerprint, code_digest: str, output: Path
) -> Path:
    """Fit existing trainers on the permitted partition and emit shared evidence.

    Args:
        trial: Frozen controller request.
        fingerprint: Runtime measured by the existing benchmark fingerprint builder.
        code_digest: Digest of the actual entrypoint source, measured by that entrypoint.
        output: Result path in a new, per-attempt directory.

    Returns:
        Shared numerical record sidecar.

    Raises:
        ValueError: RESEARCH_TRIAL_LINEAGE or RESEARCH_TRIAL_REFUSED before fitting.
        FileExistsError: Attempt output directory already exists; never overwrite a trial.
        OSError: Frozen data or result storage is unavailable.
    """
    policy, proposal, evaluation = trial["policy"], trial["proposal"], trial["evaluation"]
    reason = policy_refusal(policy, proposal)
    if reason:
        raise ValueError(f"RESEARCH_TRIAL_REFUSED: {reason}")
    manifest = Path(trial["manifest"])
    if (
        evaluation_digest(evaluation) != policy["protocol_digest"]
        or code_digest != policy["code_digest"]
        or fingerprint["image_digest"] != policy["image_digest"]
        or sha256(manifest.read_bytes()).hexdigest() != policy["dataset_digest"]
    ):
        raise ValueError("RESEARCH_TRIAL_LINEAGE: frozen protocol, code, image or manifest changed")
    if proposal["features"] != ("return_1",):
        raise ValueError("RESEARCH_TRIAL_REFUSED: diagnostic supports only return_1")
    data = load_wrds_evaluation(manifest)
    grouped = data.splits.validation_split(evaluation["inner_train_fraction"])
    output.parent.mkdir(parents=True, exist_ok=False)
    results, comparisons = evaluate_seed(
        make_baseline_trainers(proposal["config"]),
        grouped,
        proposal["seed"],
        evaluation["reference"],
        evaluation["resampling"],
        evaluation["costs"],
    )
    encoded: JSONObject = {
        "kind": "research-validation-diagnostic-v1",
        "attempt": trial["number"],
        "policy_digest": research_policy_digest(policy),
        "candidate_digest": candidate_digest(proposal),
        "partition": "validation",
        "dataset": encode_dataset_info(data.info),
        "config": encode_benchmark_config(proposal["config"]),
        "evaluation": encode_research_evaluation(evaluation),
        "results": [encode_arm_result(result) for result in results],
        "seed_comparisons": [encode_seed_comparisons(comparisons)],
        "fingerprint": encode_run_fingerprint(fingerprint),
    }
    observations = declared_observations(PAIRED_EVALUATION, results_of(encoded))
    return write_observed_manifest(
        encoded,
        fingerprint,
        output,
        policy["protocol_digest"],
        str(trial["number"]),
        observations + policy_observations([comparisons]),
    )


__all__ = ["execute_trial"]
