"""Execute one admitted monthly validation trial and publish shared evidence."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint, encode_run_fingerprint
from platform_core.json_utils import JSONObject, JSONValue
from platform_core.run_record import Observation

from ..benchmarking import make_baseline_trainers
from ..benchmarking.harness import write_observed_manifest
from ..benchmarking.types_codec import encode_benchmark_config
from ..evaluation.codec_records import encode_comparison
from ..evaluation.types import PairedComparison
from .crsp_extract import encode_crsp_extract, load_crsp_sources
from .crsp_protocol import encode_crsp_protocol
from .crsp_study import CrspFold, CrspSeries, require_crsp_features, run_crsp_validation
from .research_journal import research_policy_digest
from .research_limits import policy_refusal
from .research_records import candidate_digest
from .research_trial_records import ResearchTrial, encode_research_evaluation, evaluation_digest


def _encode_fold(fold: CrspFold) -> JSONObject:
    """Encode one fold's boundaries and purged counts."""
    return {
        "first_stopping": fold.first_stopping,
        "first_scored": fold.first_scored,
        "n_train": fold.n_train,
        "n_stopping": fold.n_stopping,
        "n_purged": fold.n_purged,
        "n_scored": fold.n_scored,
    }


def _encode_series(series: CrspSeries, seed: int) -> JSONObject:
    """Encode one arm's monthly accounting, holdings and receivables included."""
    months: list[JSONValue] = []
    for period in series.periods:
        held: list[JSONValue] = []
        for permno, weight in period.held:
            pair: list[JSONValue] = [permno, weight]
            held.append(pair)
        owed: list[JSONValue] = []
        for item in period.receivables:
            receivable: JSONObject = {
                "permno": item.permno,
                "valued": item.valued,
                "amount": item.amount,
            }
            owed.append(receivable)
        selected: list[JSONValue] = list(period.selected)
        month: JSONObject = {
            "decision": period.window.decision,
            "entry": period.window.entry,
            "exit": period.window.exit,
            "selected": selected,
            "turnover": period.turnover,
            "cost": period.cost,
            "net_return": period.net_return,
            "held": held,
            "receivables": owed,
        }
        months.append(month)
    returns = [period.net_return for period in series.periods]
    turnover = [period.turnover for period in series.periods]
    return {
        "arm": series.arm,
        "seed": seed,
        "cost_bps": series.cost_bps,
        "mean_net_return": sum(returns) / len(returns),
        "mean_turnover": sum(turnover) / len(turnover),
        "months": months,
    }


def _observations(
    series: tuple[CrspSeries, ...], comparisons: tuple[PairedComparison, ...]
) -> tuple[Observation, ...]:
    """Name each arm's mean net return and turnover and each paired interval."""
    observations: list[Observation] = []
    for item in series:
        name = f"{item.arm}.{item.cost_bps}bps"
        count = len(item.periods)
        observations.append(
            Observation(
                name=f"mean_net_return.{name}",
                value=sum(period.net_return for period in item.periods) / count,
            )
        )
        observations.append(
            Observation(
                name=f"mean_turnover.{name}",
                value=sum(period.turnover for period in item.periods) / count,
            )
        )
    for comparison in comparisons:
        base = f"{comparison['arm']}_vs_{comparison['reference']}.net_return"
        interval = comparison["difference"]
        observations.extend(
            [
                Observation(name=f"{base}.difference", value=interval["estimate"]),
                Observation(name=f"{base}.ci_low", value=interval["ci_low"]),
                Observation(name=f"{base}.ci_high", value=interval["ci_high"]),
                Observation(name=f"{base}.p_value", value=interval["p_value"]),
            ]
        )
    return tuple(observations)


def execute_trial(
    trial: ResearchTrial, fingerprint: RunFingerprint, code_digest: str, output: Path
) -> Path:
    """Run the frozen monthly study on the validation period and emit shared evidence.

    Args:
        trial: Frozen controller request.
        fingerprint: Runtime measured by the existing benchmark fingerprint builder.
        code_digest: Digest of the actual entrypoint source, measured by that entrypoint.
        output: Result path in a new, per-attempt directory.

    Returns:
        Shared numerical record sidecar.

    Raises:
        ValueError: RESEARCH_TRIAL_LINEAGE or RESEARCH_TRIAL_REFUSED before fitting;
            CRSP and evaluation refusals for incomplete evidence propagate.
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
    require_crsp_features(proposal["features"])
    sources = load_crsp_sources(manifest)
    if encode_crsp_protocol(sources.protocol) != encode_crsp_protocol(evaluation["design"]):
        raise ValueError("RESEARCH_TRIAL_LINEAGE: staged CRSP protocol differs from the design")
    output.parent.mkdir(parents=True, exist_ok=False)
    validation = run_crsp_validation(
        sources,
        make_baseline_trainers(proposal["config"]),
        proposal["seed"],
        features=proposal["features"],
        stopping_months=evaluation["stopping_months"],
        fold_months=evaluation["fold_months"],
        confidence=evaluation["confidence"],
    )
    exclusions: list[JSONValue] = []
    for decision in validation.decisions:
        counts: JSONObject = dict(decision.exclusions)
        entry: JSONObject = {"decision": decision.window.decision, "counts": counts}
        exclusions.append(entry)
    encoded: JSONObject = {
        "kind": "research-monthly-validation-v1",
        "attempt": trial["number"],
        "policy_digest": research_policy_digest(policy),
        "candidate_digest": candidate_digest(proposal),
        "partition": "validation",
        "source": encode_crsp_extract(sources.manifest),
        "config": encode_benchmark_config(proposal["config"]),
        "evaluation": encode_research_evaluation(evaluation),
        "features": list(proposal["features"]),
        "folds": [_encode_fold(fold) for fold in validation.folds],
        "exclusions": exclusions,
        "results": [_encode_series(item, proposal["seed"]) for item in validation.series],
        "comparisons": [encode_comparison(item) for item in validation.comparisons],
        "fingerprint": encode_run_fingerprint(fingerprint),
    }
    return write_observed_manifest(
        encoded,
        fingerprint,
        output,
        policy["protocol_digest"],
        str(trial["number"]),
        _observations(validation.series, validation.comparisons),
    )


__all__ = ["execute_trial"]
