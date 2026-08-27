"""Measure which attention backend each card's dispatcher selects.

The follow-up the forward trace asked for. That measurement localised the
`tiny` rung's whole cross-card divergence to one operation --
`scaled_dot_product_attention` -- with the QKV matmul feeding it bit-identical
on a V100, an A30 and an A100. It could not say WHY, because
`F.scaled_dot_product_attention` is a dispatcher and nothing recorded which
kernel it dispatched to.

This runs each of the ladder's attention calls five times: once unforced, and
once with each backend forced. The backend whose forced output is
bit-identical to the unforced one is the one the dispatcher chose, which makes
selection a measurement rather than an inference from
``can_use_flash_attention``. Torch's eligibility opinion is recorded too, so
the two can be checked against each other.

WHY IT SHARES THE PROBE'S DETERMINISM PIN. Same reason the ladder, the gemm
probe and the forward trace do: a measurement taken under a different posture
than the ones it explains is not describing the same configuration.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
)

from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.cli.probe_trace import workspace_observation
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.sdpa_probe import SdpaMeasurement, probe_sdpa
from model_trainer.core.services.model.sdpa_shapes import (
    AVAILABLE_SUFFIX,
    BACKEND_KEYS,
    DEFAULT_KEY,
    DIGEST_SUFFIX,
    ELIGIBLE_SUFFIX,
    FALSE_VALUE,
    SDPA_EXPERIMENT,
    TRUE_VALUE,
    SdpaShape,
    sdpa_label,
    sdpa_shapes,
)

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (DEVICE_FLAG, OUT_FLAG)

#: Label for the record. Fixed rather than digest-derived, because the table
#: is read as a whole: a record missing a shape shows up as missing
#: observations, which `agree_across_runs` reports as unmatched.
SDPA_LABEL = "sdpa-backend-selection-v1"


def measurement_observations(
    shape: SdpaShape, measured: SdpaMeasurement
) -> tuple[Observation, ...]:
    """Name everything one attention call produced.

    A backend that could not run contributes its ``available`` row and NO
    digest row. That asymmetry is deliberate: an absent digest is reported by
    :func:`~platform_core.run_record.agree_across_runs` as an unmatched
    observation, so a backend available on one card and not another is
    visible as a structural difference rather than hidden behind a sentinel
    number that would compare equal to nothing in particular.

    Args:
        shape: The call.
        measured: What it produced.

    Returns:
        The observations, unordered.
    """
    observations = [
        Observation(
            name=sdpa_label(shape, DEFAULT_KEY, DIGEST_SUFFIX),
            value=measured["default_digest"],
        )
    ]
    for name in BACKEND_KEYS:
        observations.append(
            Observation(
                name=sdpa_label(shape, name, AVAILABLE_SUFFIX),
                value=TRUE_VALUE if measured["available"][name] else FALSE_VALUE,
            )
        )
        digest = measured["digests"].get(name)
        if digest is not None:
            observations.append(
                Observation(name=sdpa_label(shape, name, DIGEST_SUFFIX), value=digest)
            )
    for name, verdict in measured["eligible"].items():
        observations.append(
            Observation(
                name=sdpa_label(shape, name, ELIGIBLE_SUFFIX),
                value=TRUE_VALUE if verdict else FALSE_VALUE,
            )
        )
    return tuple(observations)


def selected_backend(measured: SdpaMeasurement) -> tuple[str, ...]:
    """Name the backend or backends whose output matches the unforced call.

    Args:
        measured: One call's measurement.

    Returns:
        Every backend key whose digest equals the default's, in
        :data:`~sdpa_shapes.BACKEND_KEYS` order. Empty when none matches and
        several when several do -- both are real outcomes of this method and
        neither is resolved here, because resolving them would mean guessing.
    """
    return tuple(
        name for name in BACKEND_KEYS if measured["digests"].get(name) == measured["default_digest"]
    )


def sdpa_run_record(device: str) -> RunRecord:
    """Pin determinism, probe every attention call, and record the results.

    Determinism is pinned FIRST, before any operand reaches the device,
    because ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is
    created and the first transfer is enough to create it.

    Args:
        device: Device to probe on.

    Returns:
        The record, carrying the split-K condition alongside, since forcing
        the math backend routes through cuBLAS and the two questions meet
        there.

    Raises:
        RuntimeError: Propagated when a call did not reproduce itself.
        ValueError: Propagated from :func:`workspace_observation` when the
            split-K condition cannot be read as an integer.
    """
    workspace = workspace_observation()
    fingerprint: RunFingerprint = capture_run_fingerprint(device, probe_determinism(device))

    observations: list[Observation] = [workspace]
    for shape in sdpa_shapes():
        measured = probe_sdpa(shape, device)
        chosen = selected_backend(measured)
        _log.info(
            "sdpa %s h%d s%d: selected=%s available=%s eligible=%s",
            shape["rung"],
            shape["heads"],
            shape["sequence_len"],
            ",".join(chosen) if chosen else "NONE-MATCHED",
            ",".join(n for n in BACKEND_KEYS if measured["available"][n]),
            ",".join(n for n, ok in measured["eligible"].items() if ok),
        )
        observations.extend(measurement_observations(shape, measured))

    return run_record(
        experiment=SDPA_EXPERIMENT,
        label=SDPA_LABEL,
        fingerprint=fingerprint,
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Probe every attention call once and write the record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required flag is absent.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    record = sdpa_run_record(cli_args.require_flag(parsed, DEVICE_FLAG))

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "sdpa probe over %d calls %s -> %s",
        len(sdpa_shapes()),
        describe_run_fingerprint(record["fingerprint"]),
        out,
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="sdpa-probe",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "SDPA_LABEL",
    "entrypoint",
    "main",
    "measurement_observations",
    "sdpa_run_record",
    "selected_backend",
]


# Without this, `python -m model_trainer.cli.sdpa_probe` imports the module,
# runs nothing and exits 0 -- measured on the gate probe, where two Slurm jobs
# "succeeded" in six seconds having written no record and no stderr.
if __name__ == "__main__":
    entrypoint()
