"""Policy limits shared by admission and the actual compute entrypoint."""

from __future__ import annotations

from ..benchmarking.factory import make_benchmark_config
from .research_records import ResearchPolicy, ResearchProposal


def policy_refusal(policy: ResearchPolicy, proposal: ResearchProposal) -> str:
    """Check immutable protocol, feature, seed and resource restrictions.

    Args:
        policy: Evaluator-owned frozen policy.
        proposal: Decoded proposed execution.

    Returns:
        Named refusal or an empty string when policy permits this execution.
    """
    if proposal["protocol_digest"] != policy["protocol_digest"]:
        return "protocol_changed"
    if proposal["partition"] != "validation":
        return "holdout_forbidden"
    if not set(proposal["features"]).issubset(policy["features"]):
        return "features_forbidden"
    if proposal["seed"] not in policy["seeds"]:
        return "seed_forbidden"
    return _configuration_refusal(policy, proposal)


def _configuration_refusal(policy: ResearchPolicy, proposal: ResearchProposal) -> str:
    """Check resource ceilings and the existing benchmark's fixed settings.

    Args:
        policy: Frozen ceilings.
        proposal: Decoded candidate.

    Returns:
        Named refusal or empty string.
    """
    if proposal["seconds"] > policy["max_seconds"]:
        return "time_budget"
    config = proposal["config"]
    if not (
        1 <= config["n_estimators"] <= policy["max_trees"]
        and 2 <= config["num_leaves"] <= policy["max_leaves"]
        and 1 <= config["max_depth"] <= 16
        and 2 <= config["max_bins"] <= 255
    ):
        return "model_budget"
    expected = make_benchmark_config(
        config["n_estimators"],
        config["max_depth"],
        config["max_bins"],
        config["num_leaves"],
        repeats=1,
        warmups=0,
    )
    if config != expected:
        return "fixed_configuration_changed"
    return ""


__all__ = ["policy_refusal"]
