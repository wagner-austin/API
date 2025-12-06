from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path

from monorepo_guards._types import UnknownJson
from monorepo_guards.config import GuardConfig


def _decode_string_tuple(
    data: Mapping[str, UnknownJson], key: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Configuration key '{key}' must be a list or tuple of strings")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"All items in '{key}' must be strings")
        result.append(item)
    return tuple(result)


def _decode_boolean(data: Mapping[str, UnknownJson], key: str, default: bool) -> bool:
    value = data.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Configuration key '{key}' must be a boolean")
    return value


def _decode_dataclass_ban_segments(
    data: Mapping[str, UnknownJson], key: str, default: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, ...], ...]:
    value = data.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        msg = f"Configuration key '{key}' must be a list or tuple of lists/tuples of strings"
        raise ValueError(msg)

    result: list[tuple[str, ...]] = []
    for outer_item in value:
        if not isinstance(outer_item, (list, tuple)):
            raise ValueError(f"All items in '{key}' must be lists or tuples of strings")

        inner_list: list[str] = []
        for inner_item in outer_item:
            if not isinstance(inner_item, str):
                raise ValueError(f"All sub-items in '{key}' must be strings")
            inner_list.append(inner_item)
        result.append(tuple(inner_list))
    return tuple(result)


def _decode_monorepo_guard_config(monorepo_root: Path) -> GuardConfig:
    config_path = monorepo_root / "monorepo-guards.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Monorepo guard config not found at {config_path}")

    with config_path.open("rb") as f:
        # TOML always parses to dict at root level per spec
        config_data: dict[str, UnknownJson] = tomllib.load(f)

    guards_section_raw = config_data.get("guards")
    if guards_section_raw is None:
        guards_section: Mapping[str, UnknownJson] = {}
    elif not isinstance(guards_section_raw, dict):
        raise ValueError("The 'guards' section in monorepo-guards.toml must be a mapping.")
    else:
        guards_section = guards_section_raw

    directories = _decode_string_tuple(guards_section, "directories", ("src", "scripts", "tests"))
    exclude_parts = _decode_string_tuple(guards_section, "exclude_parts", ())
    forbid_pyi = _decode_boolean(guards_section, "forbid_pyi", True)
    allow_print_in_tests = _decode_boolean(guards_section, "allow_print_in_tests", False)
    dataclass_ban_segments = _decode_dataclass_ban_segments(
        guards_section, "dataclass_ban_segments", ()
    )

    return GuardConfig(
        root=monorepo_root,  # Temporarily, this will be overwritten
        directories=directories,
        exclude_parts=exclude_parts,
        forbid_pyi=forbid_pyi,
        allow_print_in_tests=allow_print_in_tests,
        dataclass_ban_segments=dataclass_ban_segments,
    )


__all__ = ["_decode_monorepo_guard_config"]
