"""Monthly study requests using the real CRSP reader and benchmark configuration."""

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject

from covenant_ml.benchmarking.factory import make_benchmark_config
from covenant_ml.benchmarking.types_codec import encode_benchmark_config
from covenant_ml.markets.research_records import encode_research_policy
from covenant_ml.markets.research_trial_records import decode_research_evaluation, evaluation_digest

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import study_protocol, study_rows
from ._research_fixtures import policy, proposal


def frozen_request(evaluation: JSONObject, manifest: Path, features: list[str]) -> JSONObject:
    """Bind an evaluation and staged manifest into an admitted request.

    Args:
        evaluation: Monthly validation protocol document.
        manifest: Staged CRSP extract manifest.
        features: Features both the policy and the proposal name.

    Returns:
        Complete request document whose digests match its parts.
    """
    fixed = encode_research_policy(policy())
    fixed["protocol_digest"] = evaluation_digest(decode_research_evaluation(evaluation))
    fixed["dataset_digest"] = sha256(manifest.read_bytes()).hexdigest()
    fixed["features"] = list(features)
    request = proposal()
    request["protocol_digest"] = fixed["protocol_digest"]
    request["features"] = list(features)
    request["config"] = encode_benchmark_config(make_benchmark_config(3, 2, 16, 4, 1, 0))
    return {
        "policy": fixed,
        "proposal": request,
        "evaluation": evaluation,
        "number": 1,
        "manifest": str(manifest),
    }


def trial_document(root: Path) -> JSONObject:
    """Create a frozen request over the seeded four-year extract with cheap real fits.

    Args:
        root: Synthetic input directory.

    Returns:
        Complete request document.
    """
    path, _ = write_sources(root, study_rows(), study_protocol())
    evaluation: JSONObject = {
        "design": study_protocol(),
        "stopping_months": 6,
        "fold_months": 12,
        "confidence": 0.9,
    }
    return frozen_request(evaluation, path, ["momentum", "liquidity"])


__all__ = ["frozen_request", "trial_document"]
