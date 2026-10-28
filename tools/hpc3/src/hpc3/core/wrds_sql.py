"""Inventory WRDS PostgreSQL privileges as metadata, then prove them read-only.

Two WRDS Grid Engine jobs, both under ``psql`` with ``ON_ERROR_STOP``:

1. The catalog query reads ``pg_catalog`` in a read-only transaction and
   writes one CSV row per visible relation (:data:`RELATIONS_QUERY`).
2. Every relation the rows classify as accessible is opened with
   ``SELECT * LIMIT 0`` in one read-only transaction. A single refusal fails
   the job, so a completed run means every listed grant is effective.

Neither job returns a dataset row. Authentication is WRDS Cloud's own: the
login environment names an ident-authenticated database host, exported to the
job, so no password is stored, sent or logged.

WRDS libraries are mostly alias views over vendor schemas, each guarded by a
``has_table_privilege('<schema>.<table>', 'SELECT')`` clause. A guard checks
the table without checking its schema's USAGE, so a zero-row query on an alias
can succeed where the data cannot be read: an alias counts as accessible only
when its guard target and every relation it reads carry both privileges.
"""

from __future__ import annotations

import csv
import io
import pathlib

from platform_core.json_utils import dump_json_str

from hpc3.contracts.wrds_sql import (
    ACCESS_STATUSES,
    ACCESSIBLE_STATUSES,
    SqlReceipt,
    SqlRelation,
    SqlRun,
    decode_sql_relation,
    encode_sql_receipt,
    psql_argv,
)
from hpc3.core import _test_hooks, wrds_grid

_GUARD = "'has_table_privilege\\(''([^'']+)''::text, ''SELECT''::text\\)'"

#: The catalog query; its column order is :func:`decode_sql_relation`'s.
RELATIONS_QUERY = f"""BEGIN READ ONLY;
SET LOCAL statement_timeout = '1800s';
COPY (
WITH relation AS (
  SELECT c.oid, n.nspname AS library, c.relname AS relation, c.relkind::text AS kind,
         has_schema_privilege(n.oid, 'USAGE') AS schema_usage,
         has_table_privilege(c.oid, 'SELECT') AS table_select,
         EXISTS (SELECT 1 FROM pg_catalog.pg_depend e
                 WHERE e.classid = 'pg_catalog.pg_class'::regclass AND e.objid = c.oid
                   AND e.deptype = 'e') AS extension_member
  FROM pg_catalog.pg_class c
  JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
  WHERE c.relkind IN ('r', 'v', 'm', 'f', 'p')
    AND n.nspname NOT IN ('pg_catalog', 'information_schema')
    AND n.nspname NOT LIKE 'pg\\_toast%'
    AND n.nspname NOT LIKE 'pg\\_temp\\_%'
), opened AS (
  SELECT r.*, CASE WHEN r.kind = 'v' AND r.schema_usage AND r.table_select
                   THEN pg_catalog.pg_get_viewdef(r.oid) ELSE '' END AS definition
  FROM relation r
), guarded AS (
  SELECT o.*, coalesce(substring(o.definition FROM {_GUARD}), '') AS guard_target
  FROM opened o
)
SELECT o.library, o.relation, o.kind, o.schema_usage, o.table_select, o.extension_member,
       o.definition ~ '\\m[a-z0-9_]+_err\\(' AS error_view,
       o.guard_target,
       g.oid IS NOT NULL AS guard_found,
       coalesce(has_schema_privilege(g.relnamespace, 'USAGE'), false) AS guard_schema_usage,
       coalesce(has_table_privilege(g.oid, 'SELECT'), false) AS guard_table_select,
       coalesce(d.dependencies, 0) AS dependencies,
       coalesce(d.accessible, 0) AS accessible_dependencies
FROM guarded o
LEFT JOIN LATERAL (
  SELECT gc.oid, gc.relnamespace FROM pg_catalog.pg_class gc
  JOIN pg_catalog.pg_namespace gn ON gn.oid = gc.relnamespace
  WHERE gn.nspname = split_part(o.guard_target, '.', 1)
    AND gc.relname = split_part(o.guard_target, '.', 2)
) g ON o.guard_target <> ''
LEFT JOIN LATERAL (
  SELECT count(*) AS dependencies,
         count(*) FILTER (WHERE has_schema_privilege(t.relnamespace, 'USAGE')
                          AND has_table_privilege(t.oid, 'SELECT')) AS accessible
  FROM (SELECT DISTINCT dp.refobjid FROM pg_catalog.pg_rewrite rw
        JOIN pg_catalog.pg_depend dp ON dp.objid = rw.oid
         AND dp.classid = 'pg_catalog.pg_rewrite'::regclass
         AND dp.refclassid = 'pg_catalog.pg_class'::regclass
         AND dp.refobjid <> o.oid
        WHERE rw.ev_class = o.oid) ref
  JOIN pg_catalog.pg_class t ON t.oid = ref.refobjid
) d ON o.definition <> ''
ORDER BY o.library, o.relation
) TO STDOUT WITH (FORMAT CSV);
COMMIT;
"""


def classify(relation: SqlRelation) -> str:
    """Assign the access status a relation's privileges establish.

    Args:
        relation: One catalog row.

    Returns:
        One of :data:`hpc3.contracts.wrds_sql.ACCESS_STATUSES`. A view that
        reads a retired product through an ``*_err`` function, a relation an
        extension owns and a view with no guard are never counted accessible.
    """
    granted = relation["schema_usage"] and relation["table_select"]
    if relation["kind"] != "v":
        return "direct_table_granted" if granted else "direct_access_denied"
    if not granted:
        return "direct_access_denied"
    if relation["guard_target"] != "":
        if not relation["guard_found"]:
            return "alias_guard_target_unresolved"
        guard = relation["guard_schema_usage"] and relation["guard_table_select"]
        reads = relation["accessible_dependencies"] == relation["dependencies"]
        return (
            "alias_and_underlying_granted" if guard and reads else "alias_underlying_access_denied"
        )
    if relation["error_view"]:
        return "retired_error_view"
    return "system_metadata" if relation["extension_member"] else "unguarded_view"


def read_relations(text: str) -> list[SqlRelation]:
    """Decode the catalog query's CSV output.

    Args:
        text: The whole artifact.

    Returns:
        Every relation, in the query's order.

    Raises:
        ValueError: ``WRDS_SQL_EMPTY`` for an artifact with no rows, which a
            truncated or misdirected query produces, and the decoder's
            refusals for a malformed row.
    """
    relations = [decode_sql_relation(row) for row in csv.reader(io.StringIO(text, newline=""))]
    if not relations:
        raise ValueError("WRDS_SQL_EMPTY: the catalog query wrote no relations")
    return relations


def quote_identifier(name: str) -> str:
    """Quote a PostgreSQL identifier exactly as ``quote_ident`` would always quote.

    Args:
        name: Schema or relation name, any characters.

    Returns:
        The name in double quotes with embedded double quotes doubled.
    """
    return '"' + name.replace('"', '""') + '"'


def validation_script(relations: list[SqlRelation]) -> str:
    """Write one zero-row query per relation inside a read-only transaction.

    Args:
        relations: The relations classified accessible.

    Returns:
        The psql script.
    """
    queries = [
        f"SELECT * FROM {quote_identifier(item['library'])}.{quote_identifier(item['relation'])}"
        " LIMIT 0;\n"
        for item in relations
    ]
    return "BEGIN READ ONLY;\n" + "".join(queries) + "COMMIT;\n"


def shortlist_csv(relations: list[SqlRelation], statuses: list[str]) -> str:
    """Render the accessible relations a dataset is chosen from.

    Args:
        relations: Every relation.
        statuses: Each relation's status, in the same order.

    Returns:
        CSV with a header, one row per accessible relation.
    """
    rows: list[list[str]] = [["library", "relation", "kind", "access_status"]]
    rows.extend(
        [item["library"], item["relation"], item["kind"], status]
        for item, status in zip(relations, statuses, strict=True)
        if status in ACCESSIBLE_STATUSES
    )
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def sql_inventory(host: str, remote_dir: str, output: pathlib.Path) -> SqlRun:
    """Catalog and validate SQL access, saving verified metadata locally.

    Writes ``output`` (the catalog CSV exactly as transported), the accessible
    shortlist beside it as ``<output>.accessible.csv`` and the receipt as
    ``<output>.receipt.json``.

    Args:
        host: WRDS SSH alias.
        remote_dir: New absolute directory in the user's WRDS home.
        output: New local CSV path. Its parent must already exist.

    Returns:
        The relations, their statuses and the saved receipt.

    Raises:
        ValueError: For an unsafe remote path, an unreadable scheduler id or
            a malformed catalog.
        FileExistsError: If any local output exists.
        AppError: If transport, the catalog job or any zero-row query fails.
    """
    remote = wrds_grid.remote_directory(remote_dir)
    shortlist = output.with_suffix(output.suffix + ".accessible.csv")
    receipt_file = wrds_grid.receipt_path(output)
    wrds_grid.refuse_existing([output, shortlist, receipt_file])
    wrds_grid.create_directory(host, remote)
    query = str(remote / "relations.sql")
    remote_catalog = str(remote / "relations.csv")
    query_sha = wrds_grid.upload(host, query, RELATIONS_QUERY.encode("utf-8"))
    metadata_job = wrds_grid.submit(
        host, remote, "wrds_sql_catalog", psql_argv(query, remote_catalog), environment=True
    )
    artifact = wrds_grid.download(host, remote_catalog)
    relations = read_relations(artifact["text"])
    statuses = [classify(relation) for relation in relations]
    accessible = [
        relation
        for relation, status in zip(relations, statuses, strict=True)
        if status in ACCESSIBLE_STATUSES
    ]
    script = validation_script(accessible).encode("utf-8")
    validation = str(remote / "validate.sql")
    validation_sha = wrds_grid.upload(host, validation, script)
    validation_job = wrds_grid.submit(
        host, remote, "wrds_sql_validate", psql_argv(validation, "/dev/null"), environment=True
    )
    receipt = SqlReceipt(
        host=host,
        scheduler="grid-engine",
        metadata_job_id=metadata_job,
        validation_job_id=validation_job,
        query_sha256=query_sha,
        validation_sha256=validation_sha,
        catalog_sha256=artifact["sha256"],
        remote_catalog=remote_catalog,
        visible_relations=len(relations),
        accessible_relations=len(accessible),
        status_counts={status: statuses.count(status) for status in ACCESS_STATUSES},
        scope="pg_catalog privileges and read-only zero-row queries; no dataset rows",
    )
    _test_hooks.write_text(output, artifact["text"])
    _test_hooks.write_text(shortlist, shortlist_csv(relations, statuses))
    _test_hooks.write_text(receipt_file, dump_json_str(encode_sql_receipt(receipt)) + "\n")
    return SqlRun(relations=relations, statuses=statuses, receipt=receipt)


__all__ = [
    "RELATIONS_QUERY",
    "classify",
    "quote_identifier",
    "read_relations",
    "shortlist_csv",
    "sql_inventory",
    "validation_script",
]
