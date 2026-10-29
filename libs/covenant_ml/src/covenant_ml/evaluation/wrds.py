"""Bind a single-security CRSP extract to the existing paired evaluation.

This is a data adapter, not a new evaluator. It uses today's total return as
one feature and the next observed return's sign as the label. Chronological
partitions omit at least one observation at each boundary, so a training
label cannot reach the next fold. This diagnostic is not a trading strategy.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from ..benchmarking.types import DatasetInfo
from .evidence import read_evidence, require_evidence_digest, require_evidence_filename
from .splits import ChronologicalGroupedSplits


class WrdsExtract(TypedDict):
    """A bounded, independently materialized extract and its evaluation design."""

    source_table: ReadOnly[str]
    security_id: ReadOnly[int]
    extracted_at: ReadOnly[str]
    csv_file: ReadOnly[str]
    csv_sha256: ReadOnly[str]
    query_file: ReadOnly[str]
    query_sha256: ReadOnly[str]
    train_fraction: ReadOnly[float]
    validation_fraction: ReadOnly[float]
    embargo: ReadOnly[int]


class WrdsEvaluationData(NamedTuple):
    """Existing split factory plus an identity covering data and extraction."""

    splits: ChronologicalGroupedSplits
    info: DatasetInfo


def decode_wrds_extract(value: JSONValue) -> WrdsExtract:
    """Validate provenance and the chronological evaluation design.

    Args:
        value: Untrusted extract manifest.

    Returns:
        An immutable extraction contract.

    Raises:
        ValueError: WRDS_DATA_MANIFEST for unknown fields or invalid design.
        TypeError: A required field has the wrong type.
    """
    obj = narrow_json_to_dict(value)
    if set(obj) != set(WrdsExtract.__annotations__):
        raise ValueError("WRDS_DATA_MANIFEST: exact extract fields are required")
    table = require_str(obj, "source_table")
    security = require_int(obj, "security_id")
    extracted = require_str(obj, "extracted_at")
    instant = datetime.fromisoformat(extracted)
    train = require_float(obj, "train_fraction")
    validation = require_float(obj, "validation_fraction")
    embargo = require_int(obj, "embargo")
    if (
        table not in {"crsp.dsf", "crsp_a_stock.dsf"}
        or security <= 0
        or instant.tzinfo != UTC
        or not 0 < train < 1
        or not 0 < validation < 1
        or train + validation >= 1
        or embargo < 1
    ):
        raise ValueError("WRDS_DATA_MANIFEST: require CRSP daily returns, UTC and purged folds")
    return WrdsExtract(
        source_table=table,
        security_id=security,
        extracted_at=extracted,
        csv_file=require_evidence_filename(require_str(obj, "csv_file")),
        csv_sha256=require_evidence_digest(require_str(obj, "csv_sha256")),
        query_file=require_evidence_filename(require_str(obj, "query_file")),
        query_sha256=require_evidence_digest(require_str(obj, "query_sha256")),
        train_fraction=train,
        validation_fraction=validation,
        embargo=embargo,
    )


def encode_wrds_extract(value: WrdsExtract) -> JSONObject:
    """Encode all extraction and evaluation inputs.

    Args:
        value: Validated manifest.

    Returns:
        JSON document with no machine-local paths.
    """
    return {
        "source_table": value["source_table"],
        "security_id": value["security_id"],
        "extracted_at": value["extracted_at"],
        "csv_file": value["csv_file"],
        "csv_sha256": value["csv_sha256"],
        "query_file": value["query_file"],
        "query_sha256": value["query_sha256"],
        "train_fraction": value["train_fraction"],
        "validation_fraction": value["validation_fraction"],
        "embargo": value["embargo"],
    }


def load_wrds_evaluation(path: Path) -> WrdsEvaluationData:
    """Load explicit CRSP rows and reuse chronological grouped partitions.

    Missing returns, duplicate dates, mixed securities and unsorted rows are
    refused rather than filled or silently dropped. Labels refer to the next
    observed date, not necessarily the next exchange session. Query provenance
    records how the extract was made; it does not prove vendor vintage timing.

    Args:
        path: Manifest beside its CSV and exact extraction SQL.

    Returns:
        One-return features, next-observation sign labels and shared dataset identity.

    Raises:
        ValueError: WRDS_DATA_ROWS for an invalid CSV or insufficient observations.
        OSError: Missing source evidence.
    """
    manifest = decode_wrds_extract(load_json_str(path.read_text(encoding="utf-8")))
    payload = read_evidence(path.parent, manifest["csv_file"], manifest["csv_sha256"])
    query = read_evidence(path.parent, manifest["query_file"], manifest["query_sha256"])
    if not query.strip():
        raise ValueError("WRDS_DATA_ROWS: extraction SQL cannot be empty")
    reader = csv.reader(io.StringIO(payload.decode("utf-8")))
    if next(reader, None) != ["permno", "date", "ret"]:
        raise ValueError("WRDS_DATA_ROWS: expected exactly permno,date,ret")
    days: list[int] = []
    returns: list[float] = []
    extracted_day = datetime.fromisoformat(manifest["extracted_at"]).date().toordinal()
    for row in reader:
        if len(row) != 3:
            raise ValueError("WRDS_DATA_ROWS: expected three values per row")
        security, day_text, return_text = row
        day = date.fromisoformat(day_text).toordinal()
        value = float(return_text)
        if (
            int(security) != manifest["security_id"]
            or not math.isfinite(value)
            or value < -1
            or day >= extracted_day
            or (days and day <= days[-1])
        ):
            raise ValueError("WRDS_DATA_ROWS: invalid security, chronology or return")
        days.append(day)
        returns.append(value)
    if len(days) < 3:
        raise ValueError("WRDS_DATA_ROWS: at least three dated returns are required")
    features = np.asarray(returns[:-1], dtype=np.float64).reshape(-1, 1)
    label_values: list[bool] = [value > 0 for value in returns[1:]]
    labels = np.asarray(label_values, dtype=np.int64)
    splits = ChronologicalGroupedSplits(
        features,
        labels,
        np.asarray(days[:-1], dtype=np.int64),
        manifest["train_fraction"],
        manifest["validation_fraction"],
        manifest["embargo"],
        label_available_at=np.asarray(days[1:], dtype=np.int64),
    )
    splits(0)  # Validate fold sizes before any learner is constructed.
    return WrdsEvaluationData(
        splits=splits,
        info={
            "sha256": hashlib.sha256(
                dump_json_str(encode_wrds_extract(manifest)).encode()
            ).hexdigest(),
            "n_rows": len(labels),
            "n_features": 1,
        },
    )


__all__ = [
    "WrdsEvaluationData",
    "WrdsExtract",
    "decode_wrds_extract",
    "encode_wrds_extract",
    "load_wrds_evaluation",
]
