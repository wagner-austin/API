"""Validated records of the WRDS PostgreSQL metadata catalog.

One :class:`SqlRelation` is one row the catalog query writes with ``COPY ...
FORMAT CSV``: a relation's name and kind, the account's privileges on it, the
privilege guard a WRDS alias view names and whether the relations the view
reads are accessible. No row carries dataset contents.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

#: Every access status :func:`hpc3.core.wrds_sql.classify` assigns.
ACCESS_STATUSES: tuple[str, ...] = (
    "direct_table_granted",
    "alias_and_underlying_granted",
    "alias_guard_target_unresolved",
    "alias_underlying_access_denied",
    "direct_access_denied",
    "retired_error_view",
    "system_metadata",
    "unguarded_view",
)

#: The statuses whose relations a zero-row query must open.
ACCESSIBLE_STATUSES: tuple[str, ...] = ("direct_table_granted", "alias_and_underlying_granted")

_FIELDS = 13


def psql_argv(script: str, output: str) -> list[str]:
    """Build the noninteractive, fail-fast database command used by WRDS jobs.

    Args:
        script: Saved SQL file, passed literally without shell interpretation.
        output: File receiving query output.

    Returns:
        Arguments disabling user startup scripts and password prompting.
    """
    return ["psql", "-X", "-w", "-q", "-v", "ON_ERROR_STOP=1", "-f", script, "-o", output]


class SqlRelation(TypedDict):
    """One visible relation and the privileges the account holds on it."""

    library: str
    relation: str
    kind: str
    schema_usage: bool
    table_select: bool
    extension_member: bool
    error_view: bool
    guard_target: str
    guard_found: bool
    guard_schema_usage: bool
    guard_table_select: bool
    dependencies: int
    accessible_dependencies: int


class SqlReceipt(TypedDict):
    """Where and how one SQL catalog was produced and validated.

    Both job ids are WRDS Grid Engine jobs. ``accessible_relations`` counts
    the relations the validation job opened with ``SELECT * LIMIT 0``.
    """

    host: str
    scheduler: str
    metadata_job_id: str
    validation_job_id: str
    query_sha256: str
    validation_sha256: str
    catalog_sha256: str
    remote_catalog: str
    visible_relations: int
    accessible_relations: int
    status_counts: dict[str, int]
    scope: str


class SqlRun(TypedDict):
    """A validated SQL catalog's classified relations and its receipt."""

    relations: list[SqlRelation]
    statuses: list[str]
    receipt: SqlReceipt


def _flag(value: str) -> bool:
    """Read one PostgreSQL CSV boolean.

    Args:
        value: ``t`` or ``f``.

    Returns:
        The boolean.

    Raises:
        ValueError: ``WRDS_SQL_VALUE`` for any other text.
    """
    if value not in {"t", "f"}:
        raise ValueError(f"WRDS_SQL_VALUE: expected t or f, got {value!r}")
    return value == "t"


def _count(value: str) -> int:
    """Read one non-negative CSV count.

    Args:
        value: Decimal digits.

    Returns:
        The count.

    Raises:
        ValueError: ``WRDS_SQL_VALUE`` for anything but decimal digits.
    """
    if not value.isdecimal():
        raise ValueError(f"WRDS_SQL_VALUE: expected a count, got {value!r}")
    return int(value)


def decode_sql_relation(fields: Sequence[str]) -> SqlRelation:
    """Validate one catalog row.

    Args:
        fields: The row's CSV fields, in the query's column order.

    Returns:
        The typed relation.

    Raises:
        ValueError: ``WRDS_SQL_RECORD`` for a row of the wrong width, an empty
            name or more accessible dependencies than dependencies, and
            ``WRDS_SQL_VALUE`` for a malformed boolean or count.
    """
    if len(fields) != _FIELDS:
        raise ValueError(f"WRDS_SQL_RECORD: expected {_FIELDS} fields, got {len(fields)}")
    relation = SqlRelation(
        library=fields[0],
        relation=fields[1],
        kind=fields[2],
        schema_usage=_flag(fields[3]),
        table_select=_flag(fields[4]),
        extension_member=_flag(fields[5]),
        error_view=_flag(fields[6]),
        guard_target=fields[7],
        guard_found=_flag(fields[8]),
        guard_schema_usage=_flag(fields[9]),
        guard_table_select=_flag(fields[10]),
        dependencies=_count(fields[11]),
        accessible_dependencies=_count(fields[12]),
    )
    if relation["library"] == "" or relation["relation"] == "":
        raise ValueError("WRDS_SQL_RECORD: empty library or relation name")
    if relation["accessible_dependencies"] > relation["dependencies"]:
        raise ValueError("WRDS_SQL_RECORD: more accessible dependencies than dependencies")
    return relation


def encode_sql_relation(relation: SqlRelation) -> list[str]:
    """Encode a relation back to its CSV fields.

    Args:
        relation: Validated relation.

    Returns:
        The thirteen fields :func:`decode_sql_relation` reads.
    """
    flags = (
        relation["schema_usage"],
        relation["table_select"],
        relation["extension_member"],
        relation["error_view"],
    )
    guard = (
        relation["guard_found"],
        relation["guard_schema_usage"],
        relation["guard_table_select"],
    )
    return [
        relation["library"],
        relation["relation"],
        relation["kind"],
        *("t" if flag else "f" for flag in flags),
        relation["guard_target"],
        *("t" if flag else "f" for flag in guard),
        str(relation["dependencies"]),
        str(relation["accessible_dependencies"]),
    ]


def encode_sql_receipt(receipt: SqlReceipt) -> JSONObject:
    """Encode a receipt for the JSON file saved beside its catalog.

    Args:
        receipt: Validated receipt.

    Returns:
        The JSON object.
    """
    counts: JSONObject = dict(receipt["status_counts"])
    return {
        "host": receipt["host"],
        "scheduler": receipt["scheduler"],
        "metadata_job_id": receipt["metadata_job_id"],
        "validation_job_id": receipt["validation_job_id"],
        "query_sha256": receipt["query_sha256"],
        "validation_sha256": receipt["validation_sha256"],
        "catalog_sha256": receipt["catalog_sha256"],
        "remote_catalog": receipt["remote_catalog"],
        "visible_relations": receipt["visible_relations"],
        "accessible_relations": receipt["accessible_relations"],
        "status_counts": counts,
        "scope": receipt["scope"],
    }


def decode_sql_receipt(value: JSONValue) -> SqlReceipt:
    """Validate a saved receipt.

    Args:
        value: Parsed JSON value.

    Returns:
        The typed receipt.

    Raises:
        JSONTypeError: When a field is missing or of the wrong type, or a
            status of :data:`ACCESS_STATUSES` has no integer count.
        ValueError: ``WRDS_SQL_STATUS`` for a status
            :data:`ACCESS_STATUSES` does not name.
    """
    fields = narrow_json_to_dict(value)
    counts = require_dict(fields, "status_counts")
    unknown = sorted(set(counts) - set(ACCESS_STATUSES))
    if unknown:
        raise ValueError(f"WRDS_SQL_STATUS: {', '.join(unknown)}")
    return SqlReceipt(
        host=require_str(fields, "host"),
        scheduler=require_str(fields, "scheduler"),
        metadata_job_id=require_str(fields, "metadata_job_id"),
        validation_job_id=require_str(fields, "validation_job_id"),
        query_sha256=require_str(fields, "query_sha256"),
        validation_sha256=require_str(fields, "validation_sha256"),
        catalog_sha256=require_str(fields, "catalog_sha256"),
        remote_catalog=require_str(fields, "remote_catalog"),
        visible_relations=require_int(fields, "visible_relations"),
        accessible_relations=require_int(fields, "accessible_relations"),
        status_counts={status: require_int(counts, status) for status in ACCESS_STATUSES},
        scope=require_str(fields, "scope"),
    )


__all__ = [
    "ACCESSIBLE_STATUSES",
    "ACCESS_STATUSES",
    "SqlReceipt",
    "SqlRelation",
    "SqlRun",
    "decode_sql_receipt",
    "decode_sql_relation",
    "encode_sql_receipt",
    "encode_sql_relation",
]
