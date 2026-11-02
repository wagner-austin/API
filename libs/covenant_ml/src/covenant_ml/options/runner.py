"""Run the realized-path Asian-put study once from a frozen request.

The request pins every input by digest: the encrypted train-validation CRSP
bundle, the carry-rate download, the contract, the simulated training model
and the interval settings. The run fits the policy on simulated paths first,
so nothing it learns comes from the realized data. It then decrypts the
bundle only inside the scheduler's private scratch
(:func:`~covenant_ml.markets.encrypted_storage.decrypted_bundle`), builds the
contracts and leaves the plaintext before scoring. The manifest and its shared
numerical record are written together
(:func:`~covenant_ml.benchmarking.harness.write_observed_manifest`). The
evaluator's final partition is never requested.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint, encode_run_fingerprint
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_int,
    require_str,
)
from platform_core.run_record import Observation
from typing_extensions import TypedDict

from ..benchmarking.harness import write_observed_manifest
from ..markets.campaign_records import require_digest
from ..markets.crsp_extract import encode_crsp_extract, load_crsp_sources
from ..markets.encrypted_records import encode_encrypted_bundle
from ..markets.encrypted_storage import decrypted_bundle, read_encrypted_bundle
from ..markets.research_keys import EvaluatorKeys
from .crsp_paths import encode_path_counts, realized_contracts
from .lsm import fit_policy, simulated_value
from .rates import parse_rate_series
from .study import StudySettings, run_study
from .types import (
    AsianPutContract,
    StudyResult,
    decode_contract,
    encode_contract,
    encode_policy,
    encode_study_result,
)

ERR_INPUT_DIGEST = "CVML-OPT-007"

#: The experiment every run's shared numerical record names.
EXPERIMENT = "asian-option-realized-paths"


class AsianStudyRequest(TypedDict, total=True):
    """Every input of one run, pinned.

    Args:
        bundle: Staged encrypted train-validation bundle metadata.
        bundle_sha256: Its admitted digest.
        rates: Carry-rate download (FRED CSV).
        rates_sha256: Its digest.
        rates_series: The series column it carries, e.g. ``DGS1``.
        first_decision: Inclusive first decision bound.
        last_decision: Inclusive last decision bound.
        contract: The contract written on every realized path.
        training_rate: Continuously compounded rate of the simulated model.
        training_volatility: Annual volatility of the simulated model.
        training_paths: Simulated training paths.
        training_seed: Their seed.
        block_length: Consecutive months per bootstrap block.
        n_resamples: Bootstrap draws.
        seed: Bootstrap seed.
        confidence: Interval coverage.
    """

    bundle: str
    bundle_sha256: str
    rates: str
    rates_sha256: str
    rates_series: str
    first_decision: str
    last_decision: str
    contract: AsianPutContract
    training_rate: float
    training_volatility: float
    training_paths: int
    training_seed: int
    block_length: int
    n_resamples: int
    seed: int
    confidence: float


def encode_request(request: AsianStudyRequest) -> JSONObject:
    """Lower a request to JSON.

    Args:
        request: The request.

    Returns:
        A JSON object mirroring :class:`AsianStudyRequest`.
    """
    return {
        "bundle": request["bundle"],
        "bundle_sha256": request["bundle_sha256"],
        "rates": request["rates"],
        "rates_sha256": request["rates_sha256"],
        "rates_series": request["rates_series"],
        "first_decision": request["first_decision"],
        "last_decision": request["last_decision"],
        "contract": encode_contract(request["contract"]),
        "training_rate": request["training_rate"],
        "training_volatility": request["training_volatility"],
        "training_paths": request["training_paths"],
        "training_seed": request["training_seed"],
        "block_length": request["block_length"],
        "n_resamples": request["n_resamples"],
        "seed": request["seed"],
        "confidence": request["confidence"],
    }


def decode_request(value: JSONValue) -> AsianStudyRequest:
    """Raise a request out of untrusted JSON.

    Args:
        value: The parsed document.

    Returns:
        The validated request.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
        ValueError: CAMPAIGN-001 for a malformed digest; CVML-OPT-001 for the
            contract.
    """
    obj = narrow_json_to_dict(value)
    return AsianStudyRequest(
        bundle=require_str(obj, "bundle"),
        bundle_sha256=require_digest(obj, "bundle_sha256"),
        rates=require_str(obj, "rates"),
        rates_sha256=require_digest(obj, "rates_sha256"),
        rates_series=require_str(obj, "rates_series"),
        first_decision=require_str(obj, "first_decision"),
        last_decision=require_str(obj, "last_decision"),
        contract=decode_contract(require_dict(obj, "contract")),
        training_rate=require_float(obj, "training_rate"),
        training_volatility=require_float(obj, "training_volatility"),
        training_paths=require_int(obj, "training_paths"),
        training_seed=require_int(obj, "training_seed"),
        block_length=require_int(obj, "block_length"),
        n_resamples=require_int(obj, "n_resamples"),
        seed=require_int(obj, "seed"),
        confidence=require_float(obj, "confidence"),
    )


def request_digest(request: AsianStudyRequest) -> str:
    """Identify a request by the digest of its canonical JSON.

    Args:
        request: The request.

    Returns:
        Lowercase SHA-256 of the encoded request.
    """
    return sha256(dump_json_str(encode_request(request)).encode("utf-8")).hexdigest()


def study_observations(result: StudyResult, simulated: float) -> tuple[Observation, ...]:
    """Name the numbers a run's shared record carries.

    Args:
        result: The realized-path comparison.
        simulated: The policy's value on fresh simulated paths of its own model.

    Returns:
        The policy's simulated value, both realized monthly means, and the
        paired difference with its interval.
    """
    comparison = result["comparison"]
    interval = comparison["difference"]
    return (
        Observation(name="simulated_value", value=simulated),
        Observation(name="realized_policy_mean", value=comparison["value_arm"]),
        Observation(name="realized_hold_mean", value=comparison["value_reference"]),
        Observation(name="difference", value=interval["estimate"]),
        Observation(name="difference.ci_low", value=interval["ci_low"]),
        Observation(name="difference.ci_high", value=interval["ci_high"]),
    )


def execute_asian_study(
    request: AsianStudyRequest,
    fingerprint: RunFingerprint,
    code_digest: str,
    output: Path,
    *,
    keys: EvaluatorKeys,
    scratch: Path,
) -> Path:
    """Run the study once and publish its manifest and shared record.

    Args:
        request: The frozen request.
        fingerprint: Measured runtime identity.
        code_digest: Digest of the entrypoint source that ran.
        output: Manifest path inside a directory this run creates.
        keys: Evaluator key custody; only the train-validation identity is read.
        scratch: Private job-local scratch supplied by the scheduler.

    Returns:
        The shared numerical record sidecar.

    Raises:
        ValueError: CVML-OPT-007 when the rate download is not the pinned
            bytes; bundle, CRSP, path and interval refusals propagate.
        FileExistsError: The run's directory exists; a run is never overwritten.
        OSError: Inputs or result storage are unavailable.
    """
    payload = Path(request["rates"]).read_bytes()
    if sha256(payload).hexdigest() != request["rates_sha256"]:
        raise ValueError(f"[{ERR_INPUT_DIGEST}] the rate download is not the pinned bytes")
    series = parse_rate_series(payload.decode("utf-8"), request["rates_series"])
    contract = request["contract"]
    policy = fit_policy(
        contract,
        request["training_rate"],
        request["training_volatility"],
        request["training_paths"],
        request["training_seed"],
    )
    simulated = simulated_value(policy, request["training_paths"], request["training_seed"] + 1)
    encrypted = Path(request["bundle"])
    bundle = read_encrypted_bundle(encrypted, request["bundle_sha256"])
    with decrypted_bundle(
        encrypted,
        request["bundle_sha256"],
        partition="train-validation",
        identity=keys.acquire("train-validation"),
        scratch=scratch,
        max_plaintext_bytes=keys.access["max_plaintext_bytes"],
    ) as manifest:
        sources = load_crsp_sources(manifest)
        contracts, counts = realized_contracts(
            sources, request["first_decision"], request["last_decision"], contract, series
        )
    settings = StudySettings(
        request["block_length"], request["n_resamples"], request["seed"], request["confidence"]
    )
    result = run_study(policy, "fixed-gbm-transfer", contracts, settings)
    output.parent.mkdir(parents=True, exist_ok=False)
    encoded: JSONObject = {
        "kind": "asian-option-realized-paths-v1",
        "request": encode_request(request),
        "request_sha256": request_digest(request),
        "code_sha256": code_digest,
        "dataset": {
            "bundle_sha256": request["bundle_sha256"],
            "bundle": encode_encrypted_bundle(bundle),
        },
        "source": encode_crsp_extract(sources.manifest),
        "policy": encode_policy(policy),
        "simulated_value": simulated,
        "paths": encode_path_counts(counts),
        "results": [encode_study_result(result)],
        "fingerprint": encode_run_fingerprint(fingerprint),
    }
    return write_observed_manifest(
        encoded,
        fingerprint,
        output,
        EXPERIMENT,
        request_digest(request)[:12],
        study_observations(result, simulated),
    )


__all__ = [
    "ERR_INPUT_DIGEST",
    "EXPERIMENT",
    "AsianStudyRequest",
    "decode_request",
    "encode_request",
    "execute_asian_study",
    "request_digest",
    "study_observations",
]
