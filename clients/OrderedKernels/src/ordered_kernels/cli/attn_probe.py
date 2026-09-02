"""Stage-digest attention at chosen sequence lengths, vendor and ordered.

THE QUESTION THIS ANSWERS. The sm_75 card broke the fixed-order scoring
identity at exactly 15- and 16-token sequences -- below anything the SDPA
probe tables sample -- and the leading explanation blamed attention, the one
operation the arms leave to the vendor. This probe takes attention apart at
EXACT lengths of the caller's choosing and digests every stage separately:
the two matmuls, the softmax, and the dispatcher's own composite. Comparing
one card's record against another's names the diverging stage instead of
suspecting it; the digests are one-OS quantities (seeded ``randn``), so a
comparison is Windows-to-Windows or Linux-to-Linux, never across.

Both operand layouts are probed because the real path hands the math kernel
NON-CONTIGUOUS views -- ``_split_heads`` is a view and a permute -- and
kernel selection is allowed to depend on strides.

The ordered stages ride in the same record: their cross-card agreement is
the closure evidence, measured by the same instrument that measured the
break.
"""

from __future__ import annotations

import math
import pathlib
import sys
from collections.abc import Sequence

import torch
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.tensor_digest import (
    describe_tensor,
    require_reproduced,
)
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

from ordered_kernels.attention import causal_bias, ordered_causal_attention, ordered_softmax
from ordered_kernels.kernels import gemm_batched
from ordered_kernels.torch_surface import split_three

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
LENGTHS_FLAG = "--lengths"
OUT_FLAG = "--out"

_FLAGS = (DEVICE_FLAG, LENGTHS_FLAG, OUT_FLAG)

ATTN_EXPERIMENT = "attn-attribution-v1"

#: GPT-2 small's attention geometry -- the one the scorer runs.
HEADS = 12
HEAD_DIM = 64

#: Operand seed base; each length draws from its own stream so adding a
#: length never shifts another's operands.
SEED_BASE = 4200

#: The two operand layouts: densely packed, and the view-and-permute strides
#: the real ``_split_heads`` produces.
LAYOUTS = ("contig", "strided")


def require_lengths(raw: str) -> tuple[int, ...]:
    """Parse the probed lengths, refusing junk before anything computes.

    Args:
        raw: Comma-separated positive integers, e.g. ``15,16,17,64``.

    Returns:
        The lengths, in the caller's order.

    Raises:
        ValueError: For an empty list, a non-integer, a length below 1, or a
            duplicate -- a record with two identically named observations
            would silently keep one.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{LENGTHS_FLAG} needs at least one length")
    lengths: list[int] = []
    for part in parts:
        if not part.isdigit() or int(part) < 1:
            raise ValueError(f"{LENGTHS_FLAG} takes positive integers, got {part!r}")
        lengths.append(int(part))
    if len(set(lengths)) != len(lengths):
        raise ValueError(f"{LENGTHS_FLAG} has duplicates: {raw!r}")
    return tuple(lengths)


def attn_operands(
    length: int, layout: str, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seeded q, k, v at one length in one layout.

    Generated on CPU under a per-length seed exactly as the GEMM probe
    generates its operands, then moved -- so every card of one OS computes
    from identical bytes.

    Args:
        length: Sequence length.
        layout: ``contig`` for densely packed operands, ``strided`` for the
            view-and-permute layout ``_split_heads`` hands the dispatcher.
        device: Where the probe runs.

    Returns:
        ``(q, k, v)``, each ``[1, HEADS, length, HEAD_DIM]`` float32.

    Raises:
        ValueError: For an unknown layout name.
    """
    torch.manual_seed(SEED_BASE + length)
    if layout == "contig":
        q = torch.randn(1, HEADS, length, HEAD_DIM).to(device)
        k = torch.randn(1, HEADS, length, HEAD_DIM).to(device)
        v = torch.randn(1, HEADS, length, HEAD_DIM).to(device)
        return q, k, v
    if layout == "strided":
        packed = torch.randn(1, length, 3 * HEADS * HEAD_DIM).to(device)
        query, key, value = split_three(packed, HEADS * HEAD_DIM, 2)
        return (
            query.view(1, length, HEADS, HEAD_DIM).permute(0, 2, 1, 3),
            key.view(1, length, HEADS, HEAD_DIM).permute(0, 2, 1, 3),
            value.view(1, length, HEADS, HEAD_DIM).permute(0, 2, 1, 3),
        )
    raise ValueError(f"unknown layout {layout!r}")


def stage_tensors(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[tuple[str, torch.Tensor], ...]:
    """Every stage of both arms over one operand set.

    The vendor stages take attention apart with the ops the math kernel's
    shape dispatch governs -- two ``torch.matmul`` calls and a
    ``torch.softmax`` -- plus the dispatcher's own composite, which under
    this probe's posture is the math kernel itself. The ordered stages are
    the same decomposition on the owned kernels. Later stages consume
    EARLIER STAGES of their own arm, so a divergence names the first stage
    that moved rather than echoing through all of them.

    Args:
        q: ``[1, HEADS, L, HEAD_DIM]``.
        k: Same shape.
        v: Same shape.

    Returns:
        ``(stage name, tensor)`` pairs.
    """
    length = int(q.shape[2])
    scale = 1.0 / math.sqrt(float(HEAD_DIM))
    bias = causal_bias(length, q.device)

    qk = torch.matmul(q, k.transpose(-1, -2))
    probs = torch.softmax(qk * scale + bias, dim=-1)
    av = torch.matmul(probs, v)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    folded = HEADS
    oqk = gemm_batched(
        q.reshape(folded, length, HEAD_DIM), k.transpose(-1, -2).reshape(folded, HEAD_DIM, length)
    )
    oprobs = ordered_softmax((oqk * scale + bias).reshape(folded * length, length))
    oattn = ordered_causal_attention(q, k, v)

    return (
        ("vendor-qk", qk),
        ("vendor-softmax", probs),
        ("vendor-av", av),
        ("vendor-sdpa", sdpa),
        ("ordered-qk", oqk),
        ("ordered-softmax", oprobs),
        ("ordered-attn", oattn),
    )


def stage_observations(device: str, lengths: tuple[int, ...]) -> tuple[Observation, ...]:
    """Digest every stage at every length in both layouts, twice-run.

    Args:
        device: Where the stages run.
        lengths: The sequence lengths to probe.

    Returns:
        Two observations per stage -- the folded digest and the float64 sum.

    Raises:
        RuntimeError: Propagated from ``require_reproduced`` when a stage
            cannot repeat itself on this card.
    """
    observations: list[Observation] = []
    for length in lengths:
        for layout in LAYOUTS:
            q, k, v = attn_operands(length, layout, device)
            first = stage_tensors(q, k, v)
            second = stage_tensors(q, k, v)
            for (name, tensor_a), (_, tensor_b) in zip(first, second, strict=True):
                what = f"attention stage {name} at L{length} ({layout})"
                digest, total = describe_tensor(
                    require_reproduced(tensor_a.cpu(), tensor_b.cpu(), what, device)
                )
                base = f"attn-L{length}-{layout}-{name}"
                observations.append(Observation(name=f"{base}|digest48", value=digest))
                observations.append(Observation(name=f"{base}|sum", value=total))
                _log.info("%s digest=%.0f sum=%.17g", base, digest, total)
    return tuple(observations)


def attn_run_record(device: str, lengths: tuple[int, ...]) -> RunRecord:
    """Pin the posture and take the whole record.

    Args:
        device: Where the probe runs.
        lengths: The sequence lengths to probe.

    Returns:
        The record, labelled ``attn-attribution-v1-both-stages``.

    Raises:
        RuntimeError: Propagated from :func:`stage_observations`.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device,
        probe_determinism(device, remove_split_k=True, math_attention=True),
    )
    return run_record(
        experiment=ATTN_EXPERIMENT,
        label=f"{ATTN_EXPERIMENT}-both-stages",
        fingerprint=fingerprint,
        observations=stage_observations(device, lengths),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Probe every requested length and write the record.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            absent, or the lengths do not parse -- resolved before anything
            computes.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    device = cli_args.require_flag(parsed, DEVICE_FLAG)
    lengths = require_lengths(cli_args.require_flag(parsed, LENGTHS_FLAG))
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = attn_run_record(device, lengths)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")
    _log.info(
        "%d lengths staged %s %s -> %s",
        len(lengths),
        record["label"],
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
        service_name="ordered-attn-probe",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "ATTN_EXPERIMENT",
    "HEADS",
    "HEAD_DIM",
    "LAYOUTS",
    "SEED_BASE",
    "attn_operands",
    "attn_run_record",
    "entrypoint",
    "main",
    "require_lengths",
    "stage_observations",
    "stage_tensors",
]


if __name__ == "__main__":
    entrypoint()
