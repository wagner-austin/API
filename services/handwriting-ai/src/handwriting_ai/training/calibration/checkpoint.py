from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

from handwriting_ai.training.calibration.cache import _decode_float, _decode_int
from handwriting_ai.training.calibration.measure import CalibrationResult


class CalibrationStage(str, Enum):
    A = "A"
    B = "B"


class CalibrationCheckpoint(TypedDict):
    stage: CalibrationStage
    index: int
    results: list[CalibrationResult]
    shortlist: list[CalibrationResult] | None
    seed: int | None


def __encode_checkpoint(
    ck: CalibrationCheckpoint,
) -> dict[str, str | int | float | bool | list[dict[str, str | int | float | bool | None]] | None]:
    return {
        "stage": ck["stage"].value,
        "index": int(ck["index"]),
        "results": [
            {
                "intra_threads": int(r["intra_threads"]),
                "interop_threads": int(r["interop_threads"])
                if r["interop_threads"] is not None
                else None,
                "num_workers": int(r["num_workers"]),
                "batch_size": int(r["batch_size"]),
                "samples_per_sec": float(r["samples_per_sec"]),
                "p95_ms": float(r["p95_ms"]),
            }
            for r in ck["results"]
        ],
        "shortlist": (
            [
                {
                    "intra_threads": int(r["intra_threads"]),
                    "interop_threads": int(r["interop_threads"])
                    if r["interop_threads"] is not None
                    else None,
                    "num_workers": int(r["num_workers"]),
                    "batch_size": int(r["batch_size"]),
                    "samples_per_sec": float(r["samples_per_sec"]),
                    "p95_ms": float(r["p95_ms"]),
                }
                for r in ck["shortlist"]
            ]
            if ck["shortlist"] is not None
            else None
        ),
        "seed": (int(ck["seed"]) if ck["seed"] is not None else None),
    }


def _decode_checkpoint(d: dict[str, JSONValue]) -> CalibrationCheckpoint:
    stage_raw = d.get("stage")
    idx_raw = d.get("index")
    res_raw = d.get("results")
    shortlist_raw = d.get("shortlist")
    seed_raw = d.get("seed")

    if not isinstance(stage_raw, str) or not isinstance(idx_raw, int):
        raise ValueError("invalid checkpoint header")
    stage = CalibrationStage(stage_raw)

    def _decode_result(x: JSONValue) -> CalibrationResult:
        if not isinstance(x, dict):
            raise ValueError("invalid result entry")
        # Use direct keyed lookups to avoid iterating arbitrary keys,
        # preserving semantics and strict typing.
        it = _decode_int({"intra_threads": x.get("intra_threads")}, "intra_threads", 0)
        inter_raw = x.get("interop_threads")
        inter_i = (
            _decode_int({"interop_threads": inter_raw}, "interop_threads", 0)
            if inter_raw is not None
            else None
        )
        nw = _decode_int({"num_workers": x.get("num_workers")}, "num_workers", 0)
        bs = _decode_int({"batch_size": x.get("batch_size")}, "batch_size", 0)
        sps = _decode_float({"samples_per_sec": x.get("samples_per_sec")}, "samples_per_sec", 0.0)
        p95 = _decode_float({"p95_ms": x.get("p95_ms")}, "p95_ms", 0.0)
        return {
            "intra_threads": it,
            "interop_threads": inter_i,
            "num_workers": nw,
            "batch_size": bs,
            "samples_per_sec": sps,
            "p95_ms": p95,
        }

    results: list[CalibrationResult] = []
    if isinstance(res_raw, list):
        results = [_decode_result(x) for x in res_raw]

    shortlist: list[CalibrationResult] | None = None
    if isinstance(shortlist_raw, list):
        shortlist = [_decode_result(x) for x in shortlist_raw]

    seed: int | None = int(seed_raw) if isinstance(seed_raw, int) else None
    return {
        "stage": stage,
        "index": int(idx_raw),
        "results": results,
        "shortlist": shortlist,
        "seed": seed,
    }


def read_checkpoint(path: Path) -> CalibrationCheckpoint | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    data: JSONValue = load_json_str(raw)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be an object")
    return _decode_checkpoint(data)


def write_checkpoint(path: Path, ckpt: CalibrationCheckpoint) -> None:
    payload = dump_json_str(__encode_checkpoint(ckpt), compact=True)
    # Ensure parent directory exists for CI and runtime environments
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


__all__ = [
    "CalibrationCheckpoint",
    "CalibrationStage",
    "read_checkpoint",
    "write_checkpoint",
]
