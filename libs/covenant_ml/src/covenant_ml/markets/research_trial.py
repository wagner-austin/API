"""Execute one admitted monthly validation trial and publish shared evidence.

The lineage checks and record publication here are shared with the single
final-period run in ``research_final``, which releases the selected candidate first.
"""

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
from .crsp_extract import CrspSources, encode_crsp_extract, load_crsp_sources
from .crsp_protocol import encode_crsp_protocol
from .crsp_study import (
    CrspFold,
    CrspScored,
    CrspSeries,
    require_crsp_features,
    run_crsp_validation,
)
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


def require_trial_lineage(
    trial: ResearchTrial,
    fingerprint: RunFingerprint,
    code_digest: str,
    manifest: Path,
    manifest_digest: str,
) -> None:
    """Recheck the admitted request and every frozen identity before reading study data.

    Only the manifest's own bytes are hashed here; no source relation is read.

    Args:
        trial: Frozen controller request.
        fingerprint: Runtime measured by the existing benchmark fingerprint builder.
        code_digest: Digest of the actual entrypoint source, measured by that entrypoint.
        manifest: Staged CRSP extract manifest this run will read.
        manifest_digest: The evaluator-bound SHA-256 of that manifest's bytes.

    Raises:
        ValueError: RESEARCH_TRIAL_REFUSED for a proposal the policy refuses;
            RESEARCH_TRIAL_LINEAGE for a changed protocol, code, image or manifest;
            CRSP-009 for a feature the monthly panel cannot construct.
        OSError: The manifest cannot be read.
    """
    policy, proposal, evaluation = trial["policy"], trial["proposal"], trial["evaluation"]
    reason = policy_refusal(policy, proposal)
    if reason:
        raise ValueError(f"RESEARCH_TRIAL_REFUSED: {reason}")
    if (
        evaluation_digest(evaluation) != policy["protocol_digest"]
        or code_digest != policy["code_digest"]
        or fingerprint["image_digest"] != policy["image_digest"]
        or sha256(manifest.read_bytes()).hexdigest() != manifest_digest
    ):
        raise ValueError("RESEARCH_TRIAL_LINEAGE: frozen protocol, code, image or manifest changed")
    require_crsp_features(proposal["features"])


def load_trial_sources(trial: ResearchTrial, manifest: Path) -> CrspSources:
    """Load verified CRSP evidence and require its protocol to be the admitted design.

    Args:
        trial: Frozen controller request whose lineage was already checked.
        manifest: Staged CRSP extract manifest.

    Returns:
        Verified sources carrying exactly the frozen design.

    Raises:
        ValueError: RESEARCH_TRIAL_LINEAGE for a different staged protocol;
            extract verification refusals propagate.
        OSError: Staged evidence cannot be read.
    """
    sources = load_crsp_sources(manifest)
    if encode_crsp_protocol(sources.protocol) != encode_crsp_protocol(
        trial["evaluation"]["design"]
    ):
        raise ValueError("RESEARCH_TRIAL_LINEAGE: staged CRSP protocol differs from the design")
    return sources


def publish_monthly_record(
    trial: ResearchTrial,
    fingerprint: RunFingerprint,
    sources: CrspSources,
    output: Path,
    *,
    partition: str,
    label: str,
    study: CrspScored,
) -> Path:
    """Write one monthly study's complete accounting and its shared numerical record.

    Args:
        trial: Frozen controller request.
        fingerprint: Measured runtime identity.
        sources: The verified evidence the study read.
        output: Result path in a directory created for this run.
        partition: ``validation`` or ``final``, naming the evaluated period.
        label: RunRecord label identifying the run within the admitted protocol.
        study: Scored decisions, fold counts, series and paired comparisons.

    Returns:
        Shared numerical record sidecar.

    Raises:
        ValueError: Invalid results or shared numerical record.
        OSError: Either artifact cannot be written.
    """
    policy, proposal = trial["policy"], trial["proposal"]
    exclusions: list[JSONValue] = []
    for decision in study.decisions:
        counts: JSONObject = dict(decision.exclusions)
        entry: JSONObject = {"decision": decision.window.decision, "counts": counts}
        exclusions.append(entry)
    encoded: JSONObject = {
        "kind": f"research-monthly-{partition}-v1",
        "attempt": trial["number"],
        "policy_digest": research_policy_digest(policy),
        "candidate_digest": candidate_digest(proposal),
        "partition": partition,
        "source": encode_crsp_extract(sources.manifest),
        "config": encode_benchmark_config(proposal["config"]),
        "evaluation": encode_research_evaluation(trial["evaluation"]),
        "features": list(proposal["features"]),
        "folds": [_encode_fold(fold) for fold in study.folds],
        "exclusions": exclusions,
        "results": [_encode_series(item, proposal["seed"]) for item in study.series],
        "comparisons": [encode_comparison(item) for item in study.comparisons],
        "fingerprint": encode_run_fingerprint(fingerprint),
    }
    return write_observed_manifest(
        encoded,
        fingerprint,
        output,
        policy["protocol_digest"],
        label,
        _observations(study.series, study.comparisons),
    )


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
    proposal, evaluation = trial["proposal"], trial["evaluation"]
    manifest = Path(trial["manifest"])
    require_trial_lineage(
        trial, fingerprint, code_digest, manifest, trial["policy"]["dataset_digest"]
    )
    sources = load_trial_sources(trial, manifest)
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
    return publish_monthly_record(
        trial,
        fingerprint,
        sources,
        output,
        partition="validation",
        label=str(trial["number"]),
        study=validation,
    )


__all__ = [
    "execute_trial",
    "load_trial_sources",
    "publish_monthly_record",
    "require_trial_lineage",
]
