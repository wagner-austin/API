"""Small synthetic requests using real CRSP adapter and benchmark configuration."""

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject

from covenant_ml.benchmarking.factory import make_benchmark_config
from covenant_ml.benchmarking.types_codec import encode_benchmark_config
from covenant_ml.markets.research_records import encode_research_policy
from covenant_ml.markets.research_trial_records import decode_research_evaluation, evaluation_digest
from tests.evaluation.test_wrds import write_extract

from ._research_fixtures import policy, proposal


def trial_document(root: Path) -> JSONObject:
    """Create a frozen request with cheap, real model fits.

    Args:
        root: Synthetic input directory.

    Returns:
        Complete request document.
    """
    path, _ = write_extract(root)
    evaluation: JSONObject = {
        "reference": "lightgbm",
        "inner_train_fraction": 0.8,
        "resampling": {
            "unit": "block",
            "block_length": 3,
            "n_resamples": 20,
            "resample_seed": 7,
            "confidence": 0.95,
        },
        "costs": {"true_alert_value": 1.0, "false_alert_cost": 1.0, "missed_cost": 0.0},
    }
    fixed = encode_research_policy(policy())
    fixed["protocol_digest"] = evaluation_digest(decode_research_evaluation(evaluation))
    fixed["dataset_digest"] = sha256(path.read_bytes()).hexdigest()
    request = proposal()
    request["protocol_digest"] = fixed["protocol_digest"]
    request["config"] = encode_benchmark_config(make_benchmark_config(3, 2, 16, 4, 1, 0))
    return {
        "policy": fixed,
        "proposal": request,
        "evaluation": evaluation,
        "number": 1,
        "manifest": str(path),
    }


__all__ = ["trial_document"]
