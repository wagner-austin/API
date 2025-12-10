from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.standardization_rules import StandardizationRule


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_standardization_rule_flags_expected_duplicates(tmp_path: Path) -> None:
    rule = StandardizationRule()

    files: list[Path] = []
    files.append(_write(tmp_path / "clients/DiscordBot/src/clubbot/utils/validators.py", "x"))
    files.append(
        _write(
            tmp_path / "services/foo/src/foo/request_context.py",
            "class RequestIdMiddleware: pass",
        )
    )
    files.append(
        _write(
            tmp_path / "services/foo/src/foo/model_trainer_client.py",
            "class HTTPModelTrainerClient: pass",
        )
    )
    files.append(
        _write(tmp_path / "services/foo/src/foo/data_bank_client.py", "class DataBankClient: pass")
    )
    files.append(
        _write(
            tmp_path / "services/foo/src/foo/fastapi_adapter.py", "class FastAPIAppAdapter: pass"
        )
    )

    violations = rule.run(files)
    kinds = {v.kind for v in violations}
    assert "qr-legacy-validators" in kinds
    assert "duplicate-request-id-middleware" in kinds
    assert "duplicate-model-trainer-client" in kinds
    assert "duplicate-data-bank-client" in kinds
    assert "duplicate-fastapi-adapter" in kinds


def test_standardization_rule_skips_allowed_locations(tmp_path: Path) -> None:
    rule = StandardizationRule()
    files: list[Path] = []
    files.append(
        _write(
            tmp_path / "libs/platform_core/src/platform_core/request_context.py",
            "class RequestIdMiddleware: pass",
        )
    )
    files.append(
        _write(
            tmp_path / "libs/platform_core/src/platform_core/model_trainer_client.py",
            "class HTTPModelTrainerClient: pass",
        )
    )
    files.append(
        _write(
            tmp_path / "libs/platform_core/src/platform_core/data_bank_client.py",
            "class DataBankClient: pass",
        )
    )
    files.append(
        _write(
            tmp_path / "libs/platform_core/src/platform_core/fastapi.py",
            "class FastAPIAppAdapter: pass",
        )
    )
    files.append(
        _write(
            tmp_path / "libs/monorepo_guards/src/monorepo_guards/request_context.py",
            "class RequestIdMiddleware: pass",
        )
    )
    files.append(_write(tmp_path / "notes/readme.txt", "not python"))

    violations = rule.run(files)
    assert violations == []


def test_standardization_rule_flags_missing_request_id_middleware(tmp_path: Path) -> None:
    rule = StandardizationRule()
    service_path = _write(
        tmp_path / "services/foo/src/foo/app.py", "from fastapi import FastAPI\napp = FastAPI()"
    )
    client_path = _write(
        tmp_path / "clients/foo/src/foo/app.py", "from fastapi import FastAPI\napp = FastAPI()"
    )
    violations = rule.run([service_path, client_path])
    missing = [v.file for v in violations if v.kind == "missing-request-id-middleware"]
    assert service_path in missing
    assert client_path in missing


def test_standardization_rule_allows_fastapi_in_tests_and_libs(tmp_path: Path) -> None:
    rule = StandardizationRule()
    lib_path = _write(
        tmp_path / "libs/platform_core/tests/test_fastapi.py",
        "from fastapi import FastAPI\napp = FastAPI()",
    )
    service_test_path = _write(
        tmp_path / "services/foo/tests/test_app.py",
        "from fastapi import FastAPI\napp = FastAPI()",
    )
    violations = rule.run([lib_path, service_test_path])
    kinds = {v.kind for v in violations}
    assert "missing-request-id-middleware" not in kinds


def test_standardization_rule_allows_install_helper(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/app.py",
        (
            "from fastapi import FastAPI\n"
            "from platform_core.request_context import install_request_id_middleware\n"
            "app = FastAPI()\n"
            "install_request_id_middleware(app)\n"
        ),
    )
    assert rule.run([path]) == []


def test_standardization_rule_allows_add_middleware_call(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/app.py",
        (
            "from fastapi import FastAPI\n"
            "from platform_core.request_context import RequestIdMiddleware\n"
            "app = FastAPI()\n"
            "app.add_middleware(RequestIdMiddleware)\n"
        ),
    )
    assert rule.run([path]) == []


def test_standardization_rule_allows_attr_install_call(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/app.py",
        (
            "from fastapi import FastAPI\n"
            "import platform_core.request_context as rc\n"
            "app = FastAPI()\n"
            "rc.install_request_id_middleware(app)\n"
        ),
    )
    assert rule.run([path]) == []


def test_standardization_rule_flags_factory_call_without_middleware(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/asgi.py",
        "from foo import create_app\napp = create_app()\n",
    )
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "missing-request-id-middleware" in kinds


def test_standardization_rule_allows_local_factory_install(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/app.py",
        (
            "from fastapi import FastAPI\n"
            "from platform_core.request_context import install_request_id_middleware\n\n"
            "def create_app() -> FastAPI:\n"
            "    app = FastAPI()\n"
            "    install_request_id_middleware(app)\n"
            "    return app\n\n"
            "app = create_app()\n"
        ),
    )
    assert rule.run([path]) == []


def test_standardization_rule_flags_local_factory_without_install(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/app.py",
        (
            "from fastapi import FastAPI\n\n"
            "def create_app() -> FastAPI:\n"
            "    app = FastAPI()\n"
            "    return app\n\n"
            "app = create_app()\n"
        ),
    )
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "missing-request-id-middleware" in kinds


def test_standardization_rule_allows_pure_factory_without_module_app(tmp_path: Path) -> None:
    """Pure factory functions are allowed - the caller (asgi.py) handles middleware."""
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/factory.py",
        ("from fastapi import FastAPI\n\ndef build() -> FastAPI:\n    return FastAPI()\n"),
    )
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "missing-request-id-middleware" not in kinds


def test_standardization_rule_allows_module_install_with_factory_call(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/asgi.py",
        (
            "from platform_core.request_context import install_request_id_middleware\n"
            "from foo import create_app\n"
            "app = create_app()\n"
            "install_request_id_middleware(app)\n"
        ),
    )
    assert rule.run([path]) == []


def test_standardization_rule_respects_non_app_assignments(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/asgi.py",
        "from foo import create_app\nfoo = create_app()\n",
    )
    assert rule.run([path]) == []


def test_standardization_rule_handles_annotated_factory_call(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/asgi.py",
        (
            "import foo.factory\n"
            "from fastapi import FastAPI\n\n"
            "app: FastAPI = foo.factory.create_app()\n"
        ),
    )
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "missing-request-id-middleware" in kinds


def test_standardization_rule_allows_annotated_declaration_without_value(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/asgi.py",
        "from fastapi import FastAPI\napp: FastAPI\n",
    )
    assert rule.run([path]) == []


def test_standardization_rule_ignores_nonname_factory_call(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/asgi.py",
        "app = (lambda: object())()\n",
    )
    assert rule.run([path]) == []


def test_standardization_rule_ignores_annotated_non_app_target(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/asgi.py",
        "from fastapi import FastAPI\nfoo: FastAPI = create_app()\n",
    )
    assert rule.run([path]) == []


def test_standardization_rule_ignores_annotated_alias_to_existing_app(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/asgi.py",
        "from fastapi import FastAPI\nexisting_app = FastAPI()\napp: FastAPI = existing_app\n",
    )
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "missing-request-id-middleware" in kinds


def test_standardization_rule_flags_assignment_from_existing_app_without_middleware(
    tmp_path: Path,
) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/asgi.py",
        "from fastapi import FastAPI\nexisting_app = FastAPI()\napp = existing_app\n",
    )
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "missing-request-id-middleware" in kinds


def test_standardization_rule_raises_on_ast_error(tmp_path: Path) -> None:
    rule = StandardizationRule()
    bad = _write(
        tmp_path / "services/foo/src/foo/app.py",
        "from fastapi import FastAPI\napp = FastAPI(\n",  # invalid syntax
    )
    with pytest.raises(RuntimeError):
        rule.run([bad])


def test_standardization_rule_ignores_install_without_fastapi(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/app.py",
        (
            "from platform_core.request_context import install_request_id_middleware\n"
            "install_request_id_middleware(object())\n"
        ),
    )
    assert rule.run([path]) == []


def test_standardization_rule_flags_fastapi_without_middleware_even_with_other_calls(
    tmp_path: Path,
) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/app.py",
        "from fastapi import FastAPI\napp = FastAPI()\nprint('noop')\n",
    )
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "missing-request-id-middleware" in kinds


def test_standardization_rule_flags_fastapi_with_wrong_middleware(tmp_path: Path) -> None:
    rule = StandardizationRule()
    path = _write(
        tmp_path / "services/foo/src/foo/app.py",
        (
            "from fastapi import FastAPI\n"
            "from starlette.middleware.cors import CORSMiddleware\n"
            "app = FastAPI()\n"
            "app.add_middleware(CORSMiddleware)\n"
        ),
    )
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "missing-request-id-middleware" in kinds
