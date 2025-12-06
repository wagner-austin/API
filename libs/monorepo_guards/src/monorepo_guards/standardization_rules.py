from __future__ import annotations

import ast
from pathlib import Path

from monorepo_guards import Rule, Violation


class StandardizationRule(Rule):
    name = "standardization"

    _PLATFORM_CORE_PATH = "libs/platform_core/src/platform_core/"
    _LEGACY_BOT_VALIDATORS = "clients/DiscordBot/src/clubbot/utils/validators.py"

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            as_posix = path.as_posix()
            if not as_posix.endswith(".py"):
                continue
            if "libs/monorepo_guards" in as_posix:
                continue

            # Block reintroduction of legacy QR validators in the bot.
            if as_posix.endswith(self._LEGACY_BOT_VALIDATORS):
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="qr-legacy-validators",
                        line=as_posix,
                    )
                )
                continue

            # RequestIdMiddleware must come from platform_core (allow adapters with adapter names).
            text = self._read_text(path, as_posix)
            if self._has_duplicate_request_id_middleware(text, as_posix):
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="duplicate-request-id-middleware",
                        line=as_posix,
                    )
                )

            # Model Trainer client must come from platform_core.
            if (
                "class HTTPModelTrainerClient" in text or "class ModelTrainerClient" in text
            ) and self._PLATFORM_CORE_PATH not in as_posix:
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="duplicate-model-trainer-client",
                        line=as_posix,
                    )
                )

            # DataBank client must come from platform_core.
            if "class DataBankClient" in text and self._PLATFORM_CORE_PATH not in as_posix:
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="duplicate-data-bank-client",
                        line=as_posix,
                    )
                )

            # FastAPIAppAdapter must come from platform_core.
            if "class FastAPIAppAdapter" in text and self._PLATFORM_CORE_PATH not in as_posix:
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="duplicate-fastapi-adapter",
                        line=as_posix,
                    )
                )

            # FastAPI apps must add RequestIdMiddleware (except tests/mocks).
            is_service_src = "/services/" in as_posix and "/src/" in as_posix
            is_client_src = "/clients/" in as_posix and "/src/" in as_posix
            enforce_request_id = (is_service_src or is_client_src) and "/tests/" not in as_posix
            if enforce_request_id and self._needs_request_id_middleware(path, text):
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="missing-request-id-middleware",
                        line=as_posix,
                    )
                )

        return out

    def _read_text(self, path: Path, as_posix: str) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="strict")
        except OSError as exc:
            raise RuntimeError(f"failed to read {as_posix}: {exc}") from exc

    def _has_duplicate_request_id_middleware(self, text: str, as_posix: str) -> bool:
        return (
            "class RequestIdMiddleware" in text
            and "class RequestIdMiddlewareAdapter" not in text
            and self._PLATFORM_CORE_PATH not in as_posix
        )

    def _needs_request_id_middleware(self, path: Path, text: str) -> bool:
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            raise RuntimeError(f"failed to parse {path.as_posix()}: {exc}") from exc

        functions: dict[str, tuple[bool, bool]] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node.name] = self._analyze_scope(node.body)

        # NOTE: We intentionally do NOT check functions individually here.
        # A "pure factory" function (def create_app(): return FastAPI()) is valid
        # as long as the caller (e.g., asgi.py) installs middleware at module level.
        # The module-level check already handles local factories called at module level.
        return self._module_missing_middleware(tree.body, functions)

    def _analyze_scope(self, body: list[ast.stmt]) -> tuple[bool, bool]:
        creates_app = False
        has_install = False
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if self._contains_fastapi_call(stmt) or self._app_factory_call(stmt) is not None:
                creates_app = True
            if self._contains_middleware_install(stmt):
                has_install = True
        return creates_app, has_install

    def _module_missing_middleware(
        self, body: list[ast.stmt], functions: dict[str, tuple[bool, bool]]
    ) -> bool:
        module_has_install = self._analyze_scope(body)[1]

        for stmt in self._iter_statements(body):
            if self._contains_fastapi_call(stmt):
                if not module_has_install:
                    return True
                continue

            factory_name = self._app_factory_call(stmt)
            if factory_name is None:
                continue

            if module_has_install:
                continue

            fn_info = functions.get(factory_name)
            if fn_info is None:
                # External factory call without module-level middleware install.
                return True
            if self._factory_has_middleware(factory_name, functions):
                continue
            return True

        return False

    def _contains_fastapi_call(self, node: ast.stmt) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and self._is_fastapi_call(child):
                return True
        return False

    def _contains_middleware_install(self, node: ast.stmt) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and self._is_middleware_install(child):
                return True
        return False

    def _is_fastapi_call(self, node: ast.Call) -> bool:
        func = node.func
        return (isinstance(func, ast.Name) and func.id == "FastAPI") or (
            isinstance(func, ast.Attribute) and func.attr == "FastAPI"
        )

    def _app_factory_call(self, node: ast.stmt) -> str | None:
        call: ast.Call | None = None
        if isinstance(node, ast.Assign):
            if not any(
                isinstance(target, ast.Name) and target.id == "app" for target in node.targets
            ):
                return None
            if isinstance(node.value, ast.Call):
                call = node.value
        elif isinstance(node, ast.AnnAssign):
            if not (isinstance(node.target, ast.Name) and node.target.id == "app"):
                return None
            if isinstance(node.value, ast.Call):
                call = node.value
        if call is None:
            return None
        func = call.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _factory_has_middleware(
        self, factory_name: str, functions: dict[str, tuple[bool, bool]]
    ) -> bool:
        creates_app, has_install = functions[factory_name]
        return creates_app and has_install

    def _iter_statements(self, body: list[ast.stmt]) -> list[ast.stmt]:
        return [
            stmt
            for stmt in body
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

    def _is_middleware_install(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_middleware":
            args = node.args
            if args and isinstance(args[0], ast.Name) and args[0].id == "RequestIdMiddleware":
                return True
        if isinstance(func, ast.Attribute) and func.attr == "install_request_id_middleware":
            return True
        return isinstance(func, ast.Name) and func.id == "install_request_id_middleware"


__all__ = ["StandardizationRule"]
