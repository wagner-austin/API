from __future__ import annotations

from pathlib import Path

from monorepo_guards import Rule, Violation
from monorepo_guards.util import read_lines


class ValidationRule(Rule):
    name = "validation"

    _PLATFORM_VALIDATORS = "libs/platform_core/src/platform_core/validators.py"
    _REQUIRED_IMPORT = "from platform_core.validators import"
    _DUPLICATE_DEFS = (
        "def _load_json_dict",
        "def _decode_optional_literal",
        "def _decode_required_literal",
        "def _decode_int_range",
        "def _decode_float_range",
        "def _decode_bool",
        "def _decode_str",
    )

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            as_posix = path.as_posix()
            if not as_posix.endswith(".py"):
                continue
            if "libs/monorepo_guards" in as_posix:
                continue
            if "/libs/platform_core/" in as_posix:
                # platform_core is canonical for shared validators; allow local helpers there
                continue

            lines = read_lines(path)
            text = "\n".join(lines)
            is_platform_validators = self._PLATFORM_VALIDATORS in as_posix

            # Prevent redefinition of validator helpers outside platform_core.
            if not is_platform_validators and any(defn in text for defn in self._DUPLICATE_DEFS):
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="duplicate-validators",
                        line=as_posix,
                    )
                )

            # Service validators modules must import shared validators.
            is_service_validator = (
                "/services/" in as_posix
                and as_posix.endswith("validators.py")
                and "/tests/" not in as_posix
            )
            if is_service_validator and self._REQUIRED_IMPORT not in text:
                out.append(
                    Violation(
                        file=path,
                        line_no=1,
                        kind="missing-platform-validators-import",
                        line=as_posix,
                    )
                )

        return out


__all__ = ["ValidationRule"]
