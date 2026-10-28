"""The WRDS SQL catalog: its rows, its classification and its two jobs."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str
from platform_core.mcp_testing import sent_arguments

from hpc3.cli import wrds_sql_catalog as cli
from hpc3.contracts.wrds_sql import (
    ACCESS_STATUSES,
    decode_sql_receipt,
    decode_sql_relation,
    encode_sql_receipt,
    encode_sql_relation,
)
from hpc3.core.wrds_sql import (
    RELATIONS_QUERY,
    classify,
    quote_identifier,
    read_relations,
    shortlist_csv,
    sql_inventory,
    validation_script,
)
from tests.conftest import FakeRun
from tests.wrds_board import TASK, accepting_board

#: One row per status, plus the second way an alias can be denied, and a
#: direct table whose name needs quoting. Columns as the query writes them.
CATALOG = (
    "comp_na,funda,r,t,t,f,f,,f,f,f,0,0\n"
    'odd,"we""ird, name",p,t,t,f,f,,f,f,f,0,0\n'
    "secret,deals,r,f,t,f,f,,f,f,f,0,0\n"
    "secret,view,v,t,f,f,f,,f,f,f,0,0\n"
    "comp,funda,v,t,t,f,f,comp_na.funda,t,t,t,2,2\n"
    "comp,gone,v,t,t,f,f,comp_old.gone,f,f,f,1,1\n"
    "ibes,det,v,t,t,f,f,ibes_full.det,t,f,t,1,1\n"
    "ibes,sum,v,t,t,f,f,ibes_full.sum,t,t,t,2,1\n"
    "crspa,acti,v,t,t,f,t,,f,f,f,0,0\n"
    "columnar,chunk,v,t,t,t,f,,f,f,f,0,0\n"
    "misc,plain,v,t,t,f,f,,f,f,f,1,1\n"
)
STATUSES = [
    "direct_table_granted",
    "direct_table_granted",
    "direct_access_denied",
    "direct_access_denied",
    "alias_and_underlying_granted",
    "alias_guard_target_unresolved",
    "alias_underlying_access_denied",
    "alias_underlying_access_denied",
    "retired_error_view",
    "system_metadata",
    "unguarded_view",
]
VALIDATION = (
    "BEGIN READ ONLY;\n"
    'SELECT * FROM "comp_na"."funda" LIMIT 0;\n'
    'SELECT * FROM "odd"."we""ird, name" LIMIT 0;\n'
    'SELECT * FROM "comp"."funda" LIMIT 0;\n'
    "COMMIT;\n"
)
REMOTE = "/home/uci/test"


def _sha(text: str) -> str:
    """The sha256 the transport records for a text."""
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(fake_run: FakeRun, *, validation_exit: int = 0) -> None:
    """Script both jobs, every upload and the catalog download.

    Args:
        fake_run: Process seam fixture.
        validation_exit: What the validation job's ``qsub -sync y`` exits.
    """
    for name, text in (("relations.sql", RELATIONS_QUERY), ("validate.sql", VALIDATION)):
        fake_run.add(f"sha256sum '{REMOTE}/{name}'", stdout=f"{_sha(text)}  {name}\n")
    fake_run.add(f"sha256sum '{REMOTE}/relations.csv'", stdout=f"{_sha(CATALOG)}  csv\n")
    fake_run.add("cat -- ", stdout=CATALOG)
    fake_run.add("wrds_sql_catalog", stdout="11\nJob 11 exited with exit code 0.\n")
    fake_run.add(
        "wrds_sql_validate",
        stdout="12\n",
        stderr="" if validation_exit == 0 else "Job 12 exited with exit code 3.",
        returncode=validation_exit,
    )


def test_classification_covers_every_status() -> None:
    relations = read_relations(CATALOG)
    assert [classify(relation) for relation in relations] == STATUSES
    assert set(STATUSES) == set(ACCESS_STATUSES)


def test_rows_roundtrip_and_quote_inside_names() -> None:
    relations = read_relations(CATALOG)
    assert relations[1]["relation"] == 'we"ird, name'
    for relation in relations:
        assert decode_sql_relation(encode_sql_relation(relation)) == relation


@pytest.mark.parametrize(
    ("fields", "code"),
    [
        (["a"] * 12, "WRDS_SQL_RECORD"),
        (["", "t", "r", "t", "t", "f", "f", "", "f", "f", "f", "0", "0"], "WRDS_SQL_RECORD"),
        (["a", "", "r", "t", "t", "f", "f", "", "f", "f", "f", "0", "0"], "WRDS_SQL_RECORD"),
        (["a", "b", "v", "t", "t", "f", "f", "", "f", "f", "f", "1", "2"], "WRDS_SQL_RECORD"),
        (["a", "b", "r", "true", "t", "f", "f", "", "f", "f", "f", "0", "0"], "WRDS_SQL_VALUE"),
        (["a", "b", "r", "t", "t", "f", "f", "", "f", "f", "f", "-1", "0"], "WRDS_SQL_VALUE"),
    ],
)
def test_malformed_rows_are_refused(fields: list[str], code: str) -> None:
    with pytest.raises(ValueError, match=code):
        decode_sql_relation(fields)


def test_an_empty_catalog_is_refused() -> None:
    with pytest.raises(ValueError, match="WRDS_SQL_EMPTY"):
        read_relations("")


def test_validation_opens_each_accessible_relation_read_only() -> None:
    relations = read_relations(CATALOG)
    accessible = [relations[0], relations[1], relations[4]]
    assert validation_script(accessible) == VALIDATION
    assert validation_script([]) == "BEGIN READ ONLY;\nCOMMIT;\n"
    assert quote_identifier('a"b') == '"a""b"'


def test_shortlist_lists_only_accessible_relations() -> None:
    relations = read_relations(CATALOG)
    assert shortlist_csv(relations, STATUSES) == (
        "library,relation,kind,access_status\n"
        "comp_na,funda,r,direct_table_granted\n"
        'odd,"we""ird, name",p,direct_table_granted\n'
        "comp,funda,v,alias_and_underlying_granted\n"
    )


def test_the_catalog_query_is_read_only_and_metadata_only() -> None:
    assert RELATIONS_QUERY.startswith("BEGIN READ ONLY;\n")
    assert RELATIONS_QUERY.endswith(") TO STDOUT WITH (FORMAT CSV);\nCOMMIT;\n")
    assert "FROM pg_catalog.pg_class c" in RELATIONS_QUERY
    assert "has_table_privilege\\(''([^'']+)''::text, ''SELECT''::text\\)" in RELATIONS_QUERY


def test_inventory_runs_both_jobs_and_saves_verified_metadata(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    prepare(fake_run)
    output = tmp_path / "relations.csv"
    run = sql_inventory("wrds", REMOTE, output)
    assert run["statuses"] == STATUSES
    receipt = run["receipt"]
    assert receipt["metadata_job_id"] == "11"
    assert receipt["validation_job_id"] == "12"
    assert receipt["query_sha256"] == _sha(RELATIONS_QUERY)
    assert receipt["validation_sha256"] == _sha(VALIDATION)
    assert receipt["catalog_sha256"] == _sha(CATALOG)
    assert receipt["visible_relations"] == 11
    assert receipt["accessible_relations"] == 3
    assert receipt["status_counts"]["direct_table_granted"] == 2
    assert receipt["status_counts"]["unguarded_view"] == 1
    assert list(receipt["status_counts"]) == list(ACCESS_STATUSES)
    assert output.read_text() == CATALOG
    assert pathlib.Path(f"{output}.accessible.csv").read_text() == shortlist_csv(
        run["relations"], STATUSES
    )
    saved = load_json_str(pathlib.Path(f"{output}.receipt.json").read_text())
    assert decode_sql_receipt(saved) == receipt
    jobs = [command for command in fake_run.commands() if command.startswith("qsub ")]
    assert len(jobs) == 2
    for job in jobs:
        assert " -V " in job
        assert " psql -X -w -q -v ON_ERROR_STOP=1 -f " in job
    assert jobs[0].endswith(f"-f {REMOTE}/relations.sql -o {REMOTE}/relations.csv")
    assert jobs[1].endswith(f"-f {REMOTE}/validate.sql -o /dev/null")
    uploads = [call.stdin_bytes for call in fake_run.calls if call.stdin_bytes is not None]
    assert uploads == [RELATIONS_QUERY.encode(), VALIDATION.encode()]


def test_a_refused_zero_row_query_fails_the_run(tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
    prepare(fake_run, validation_exit=1)
    with pytest.raises(AppError, match="exit code 3"):
        sql_inventory("wrds", REMOTE, tmp_path / "relations.csv")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", ["out", "out.accessible.csv", "out.receipt.json"])
def test_existing_outputs_are_not_overwritten(
    existing: str, tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    (tmp_path / existing).write_text("keep")
    with pytest.raises(FileExistsError):
        sql_inventory("wrds", REMOTE, tmp_path / "out")
    assert fake_run.commands() == []


def _saved_receipt(counts: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """A receipt as a run saves it, with the given status counts."""
    return {
        "host": "wrds",
        "scheduler": "grid-engine",
        "metadata_job_id": "1",
        "validation_job_id": "2",
        "query_sha256": "q",
        "validation_sha256": "v",
        "catalog_sha256": "c",
        "remote_catalog": "/home/x",
        "visible_relations": 1,
        "accessible_relations": 0,
        "status_counts": counts,
        "scope": "s",
    }


def test_receipt_refuses_unknown_and_missing_statuses() -> None:
    counts: dict[str, JSONValue] = dict.fromkeys(ACCESS_STATUSES, 0)
    receipt = decode_sql_receipt(_saved_receipt(dict(counts)))
    assert decode_sql_receipt(encode_sql_receipt(receipt)) == receipt
    with pytest.raises(ValueError, match="WRDS_SQL_STATUS: invented"):
        decode_sql_receipt(_saved_receipt({**counts, "invented": 1}))
    del counts["unguarded_view"]
    with pytest.raises(JSONTypeError):
        decode_sql_receipt(_saved_receipt(counts))


def test_cli_posts_its_own_load_report(
    tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    prepare(fake_run)
    post = accepting_board(2)
    output = tmp_path / "relations.csv"
    argv = ["--host", "wrds", "--remote-dir", REMOTE, "--output", str(output)]
    assert cli.main([*argv, "--report-task", TASK]) == 0
    report = emitted[0]
    assert report.startswith(
        "WRDS SQL LOAD-REPORT jobs 11 (catalog) and 12 (validation) on wrds (grid-engine): "
        f"catalog sha256 {_sha(CATALOG)}; 11 visible relations, 3 accessible relations "
        "(2 direct tables, 1 aliases), every one opened by SELECT * LIMIT 0 in one "
        "read-only transaction; status counts: direct_table_granted 2, "
        "alias_and_underlying_granted 1, alias_guard_target_unresolved 1, "
        "alias_underlying_access_denied 2, direct_access_denied 2, retired_error_view 1, "
        "system_metadata 1, unguarded_view 1; "
        f"query sha256 {_sha(RELATIONS_QUERY)}, validation sha256 {_sha(VALIDATION)}."
    )
    assert emitted[1] == (
        f"SQL catalog: {output}; accessible shortlist: {output}.accessible.csv; "
        f"scheduler receipt: {output}.receipt.json"
    )
    note = sent_arguments(post.bodies[1])
    assert note["agent"] == "load-report-c4522545"
    assert note["taskId"] == TASK
    assert note["body"] == report


def test_cli_refuses_a_malformed_task_before_submitting(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    argv = ["--host", "wrds", "--remote-dir", REMOTE, "--output", str(tmp_path / "out")]
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        cli.main([*argv, "--report-task", TASK.upper()])
    assert fake_run.commands() == []
