from __future__ import annotations

from pathlib import Path

from monorepo_guards.httpx_rules import HttpxRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_httpx_rule_flags_direct_import(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "api.py"
    _write(bad, "import httpx\n\ndef fetch(): pass\n")

    rule = HttpxRule()
    violations = rule.run([bad])
    assert len(violations) == 1
    v = violations[0]
    assert v.kind == "httpx-direct-import"
    assert v.file == bad


def test_httpx_rule_flags_from_import(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "demo" / "client.py"
    _write(bad, "from httpx import Client\n\ndef fetch(): pass\n")

    rule = HttpxRule()
    violations = rule.run([bad])
    assert len(violations) == 1
    v = violations[0]
    assert v.kind == "httpx-direct-import"
    assert v.file == bad


def test_httpx_rule_allows_canonical_path(tmp_path: Path) -> None:
    canonical = (
        tmp_path / "libs" / "platform_core" / "src" / "platform_core" / "data_bank_client.py"
    )
    _write(canonical, "import httpx\n\nclass DataBankClient: pass\n")

    rule = HttpxRule()
    violations = rule.run([canonical])
    assert len(violations) == 0


def test_httpx_rule_allows_test_files(tmp_path: Path) -> None:
    test_file = tmp_path / "services" / "demo" / "tests" / "test_api.py"
    _write(test_file, "import httpx\n\ndef test_something(): pass\n")

    rule = HttpxRule()
    violations = rule.run([test_file])
    assert len(violations) == 0


def test_httpx_rule_allows_scripts(tmp_path: Path) -> None:
    script_file = tmp_path / "services" / "demo" / "scripts" / "run.py"
    _write(script_file, "import httpx\n\ndef main(): pass\n")

    rule = HttpxRule()
    violations = rule.run([script_file])
    assert len(violations) == 0


def test_httpx_rule_allows_non_httpx_imports(tmp_path: Path) -> None:
    ok = tmp_path / "services" / "demo" / "api.py"
    _write(ok, "import json\nfrom platform_core.data_bank_client import DataBankClient\n")

    rule = HttpxRule()
    violations = rule.run([ok])
    assert len(violations) == 0


def test_httpx_rule_ignores_relative_imports(tmp_path: Path) -> None:
    ok = tmp_path / "services" / "demo" / "api.py"
    _write(ok, "from . import utils\nfrom .. import helpers\n")

    rule = HttpxRule()
    violations = rule.run([ok])
    assert len(violations) == 0


def test_httpx_rule_flags_turkic_main_disallowed(tmp_path: Path) -> None:
    # turkic-api main now must use DataBankClient, not httpx
    bad = tmp_path / "services" / "turkic-api" / "src" / "turkic_api" / "api" / "main.py"
    _write(bad, "import httpx\n\ndef stream_data(): pass\n")

    rule = HttpxRule()
    violations = rule.run([bad])
    assert len(violations) == 1


def test_httpx_rule_flags_turkic_jobs_disallowed(tmp_path: Path) -> None:
    bad = tmp_path / "services" / "turkic-api" / "src" / "turkic_api" / "api" / "jobs.py"
    _write(bad, "import httpx\n\ndef upload_result(): pass\n")

    rule = HttpxRule()
    violations = rule.run([bad])
    assert len(violations) == 1
