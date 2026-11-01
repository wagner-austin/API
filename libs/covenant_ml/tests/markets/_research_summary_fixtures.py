"""Exact dyadic monthly returns for five seed records on a 36-month calendar."""

from calendar import monthrange
from datetime import date

from platform_core.json_utils import require_dict
from platform_core.run_record import Observation, run_record
from platform_core.testing import sample_run_fingerprint

from covenant_ml.markets.research_ranking import SeedResult
from covenant_ml.markets.research_records import (
    ResearchPolicy,
    candidate_digest,
    decode_research_policy,
    decode_research_proposal,
    encode_research_policy,
)
from covenant_ml.markets.research_trial_records import (
    ResearchEvaluation,
    decode_research_evaluation,
    evaluation_digest,
)

from ._crsp_study_fixtures import study_protocol
from ._research_fixtures import policy, proposal


def summary_evidence() -> tuple[ResearchPolicy, ResearchEvaluation, tuple[SeedResult, ...]]:
    """Return a frozen five-seed study with exactly representable paired differences."""
    design = study_protocol()
    design.update(
        validation_start="2019-01-01",
        holdout_start="2022-03-01",
        holdout_end="2022-06-30",
        coverage_end="2022-08-31",
        bootstrap_block_months=12,
        bootstrap_draws=10000,
    )
    evaluation = decode_research_evaluation(
        {"design": design, "stopping_months": 6, "fold_months": 12, "confidence": 0.95}
    )
    digest = evaluation_digest(evaluation)
    request = proposal()
    request["protocol_digest"] = digest
    fixed = encode_research_policy(policy())
    fixed["protocol_digest"] = digest
    fixed["seeds"] = [11, 23, 37, 53, 71]
    selection = require_dict(fixed, "selection")
    selection["last_month"] = 24263
    selection["required_candidates"] = [candidate_digest(decode_research_proposal(request))]
    frozen = decode_research_policy(fixed)
    rows: list[SeedResult] = []
    for number, seed in enumerate(frozen["seeds"], start=1):
        request["seed"] = seed
        observations: list[Observation] = []
        for month in range(36):
            year, ordinal = divmod(24228 + month, 12)
            decision = date(year, ordinal + 1, monthrange(year, ordinal + 1)[1])
            entry_year, entry_month = divmod(24229 + month, 12)
            exit_year, exit_month = divmod(24230 + month, 12)
            entry, end = date(entry_year, entry_month + 1, 1), date(exit_year, exit_month + 1, 1)
            calendar = f"{decision.isoformat()}.{entry.isoformat()}.{end.isoformat()}"
            observations.append(
                Observation(name=f"monthly_net_return.momentum.10bps.{calendar}", value=0.125)
            )
            for arm_index, arm in enumerate(("cleargbm", "lightgbm", "xgboost", "equal_weight")):
                difference = ((month // 6 - 2) * 4 + arm_index + number - 3) / 1024
                observations.append(
                    Observation(
                        name=f"monthly_net_return.{arm}.10bps.{calendar}",
                        value=0.125 + difference,
                    )
                )
        rows.append(
            SeedResult(
                number,
                decode_research_proposal(request),
                run_record(
                    digest,
                    str(number),
                    sample_run_fingerprint(image_digest="d" * 64),
                    tuple(observations),
                    str(number) * 64,
                ),
            )
        )
    return frozen, evaluation, tuple(rows)
