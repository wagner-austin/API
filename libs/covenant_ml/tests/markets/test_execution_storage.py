"""Real SQLite backup, concurrent locking, process death and ownership tamper tests."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from typing import IO

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from covenant_ml.markets.execution_journal import execution_journal
from covenant_ml.markets.execution_records import encode_intent
from covenant_ml.markets.execution_storage import (
    batch_operation,
    claim_batch_owner,
    initialize_batch_storage,
    persist_evidence,
    require_batch_owner,
)
from covenant_ml.markets.validation import read_document

from ._execution_fixtures import intent


def test_backup_is_exact_and_exclusive_evidence_cannot_be_overwritten(tmp_path: Path) -> None:
    source, root = tmp_path / "original.sqlite", tmp_path / "batch"
    execution_journal().initialize(source)
    state = execution_journal().transact(
        source,
        lambda _: {"action": "reserve", "observed_ms": 1000, "data": encode_intent(intent())},
    )
    initialize_batch_storage(root, source)
    assert execution_journal().read(root / "execution.sqlite") == state
    document: JSONObject = {"text": "é"}
    digest = persist_evidence(root / "evidence.json", document)
    assert (root / "evidence.json").read_bytes() == dump_json_str(document).encode("utf-8")
    assert digest == sha256((root / "evidence.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        persist_evidence(root / "evidence.json", {"overwrite": True})
    assert read_document(root / "evidence.json") == document
    with pytest.raises(FileExistsError):
        initialize_batch_storage(root, source)
    with pytest.raises(sqlite3.OperationalError):
        initialize_batch_storage(tmp_path / "bad", tmp_path / "missing.sqlite")


def test_real_process_lock_is_released_after_forced_process_death(tmp_path: Path) -> None:
    source, root = tmp_path / "original.sqlite", tmp_path / "batch"
    execution_journal().initialize(source)
    initialize_batch_storage(root, source)
    script = (
        "from pathlib import Path\nimport sys\n"
        "from covenant_ml.markets.execution_storage import batch_operation\n"
        "with batch_operation(Path(sys.argv[1])):\n"
        " print('locked', flush=True)\n sys.stdin.read()\n"
    )
    process: subprocess.Popen[str] = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with process:
        output: IO[str] | None = process.stdout
        if output is None:
            raise AssertionError("child stdout pipe required")
        assert output.readline() == "locked\n"
        with (
            pytest.raises(sqlite3.OperationalError, match="database is locked"),
            batch_operation(root),
        ):
            pytest.fail("a second process acquired a held operation lock")
        process.kill()
        _, error = process.communicate(timeout=10)
        code: int | None = process.returncode
        assert code in (1, -9)
        assert error == ""
    with batch_operation(root):
        assert execution_journal().read(root / "execution.sqlite")["orders"] == ()
    with closing(sqlite3.connect(root / "operation.sqlite")) as connection, connection:
        connection.execute("UPDATE ownership SET mode = 'wrong'")
    with pytest.raises(ValueError, match="DEMO-BATCH-008"), batch_operation(root):
        pytest.fail("a corrupt lock identity was accepted")


@pytest.mark.parametrize("field", ["root", "credential_fingerprint", "plan_sha256"])
def test_permanent_owner_refuses_second_campaign_and_tampering(tmp_path: Path, field: str) -> None:
    root, guards = tmp_path / "batch", tmp_path / "guards"
    root.mkdir()
    persist_evidence(root / "plan.json", {"attempt_limit": 20})
    fingerprint = "a" * 64
    claim_batch_owner(guards, root, fingerprint)
    require_batch_owner(guards, root, fingerprint)
    with pytest.raises(FileExistsError):
        claim_batch_owner(guards, root, fingerprint)
    owner = guards / (fingerprint + ".json")
    document = read_document(owner)
    document[field] = "changed"
    owner.write_text(dump_json_str(document), encoding="utf-8")
    with pytest.raises(ValueError, match="DEMO-BATCH-005"):
        require_batch_owner(guards, root, fingerprint)
