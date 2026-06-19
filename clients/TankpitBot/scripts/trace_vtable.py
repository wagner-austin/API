"""Trace V table handler field assignments from beautified tpclient.js.

Reads tpclient.pretty.js and traces the full chain for each message handler:
  wire a[N] -> parse local var -> constructor param -> this.X -> b.Y on tank

Usage: poetry run python -m scripts.trace_vtable [tpclient.pretty.js]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from scripts import _test_hooks

XC_SEMANTICS: dict[str, str] = {
    "h": "team",
    "l": "rank",
    "u": "damage_state (dual: rank_category on init)",
    "s": "lb_score",
    "aa": "persistent_tank_id",
    "name": "name",
    "v": "decoration_state",
    "id": "tank_id",
    "direction": "direction",
    "W": "carry_direction",
    "Y": "carrying_flag",
    "la": "is_carrying",
    "j": "viewport_col",
    "i": "viewport_row",
    "P": "hover_col",
    "U": "hover_row",
}

_VTABLE_SINGLE = re.compile(r"""V\["(.)"\]\s*=\s*(\$?\w+)""")
_VTABLE_DOT = re.compile(r"""V\.([A-Z])\s*=\s*(\$?\w+)""")
_FIELD_ASSIGN = re.compile(r"\s+this\.(\w+)\s*=\s*(\w+)")
_EXEC_ASSIGN = re.compile(r"\s+(\w)\.(\w+)\s*=\s*this\.(\w+)")


def _find_v_table(lines: list[str]) -> dict[str, tuple[str, int]]:
    """Find all V[char] = ClassName assignments.

    Args:
        lines: Beautified JS lines.

    Returns:
        Dict mapping message char to (class_name, line_number).
    """
    result: dict[str, tuple[str, int]] = {}
    for i, line in enumerate(lines):
        for m in _VTABLE_SINGLE.finditer(line):
            ch = str(m.group(1))
            if len(ch) == 1 and ch.isprintable():
                result[ch] = (str(m.group(2)), i)
        for m in _VTABLE_DOT.finditer(line):
            result[str(m.group(1))] = (str(m.group(2)), i)
    return result


def _find_constructor(cls: str, lines: list[str]) -> tuple[int, list[str], dict[str, str]]:
    """Find constructor, return (line, param_names, {field: param}).

    Args:
        cls: Class name to find.
        lines: Beautified JS lines.

    Returns:
        Tuple of (line_number, param_list, field_to_param_map).
        Returns (-1, [], {}) if not found.
    """
    pat = re.compile(rf"^\s*function {re.escape(cls)}\(([^)]*)\)")
    for i, line in enumerate(lines):
        m = pat.match(line)
        if m:
            params = [p.strip() for p in str(m.group(1)).split(",") if p.strip()]
            fields: dict[str, str] = {}
            for j in range(i + 1, min(i + 30, len(lines))):
                fm = _FIELD_ASSIGN.match(lines[j])
                if fm:
                    fields[str(fm.group(1))] = str(fm.group(2))
                elif lines[j].strip().startswith("}"):
                    break
            return i, params, fields
    return -1, [], {}


def _split_args_respecting_parens(raw: str) -> list[str]:
    """Split a comma-separated argument string respecting nested parens.

    Args:
        raw: Raw argument string (e.g. "a[0], X(a[1], a[2]), a[3]").

    Returns:
        List of individual argument expressions.
    """
    args: list[str] = []
    depth = 0
    current = ""
    for char in raw:
        if char == "(":
            depth += 1
            current += char
        elif char == ")":
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        args.append(current.strip())
    return args


def _extract_return_args(cls: str, lines: list[str], start: int) -> list[str]:
    """Extract arguments from 'return new Cls(...)' statement.

    Handles multi-line return statements by joining lines until the
    closing paren is found.

    Args:
        cls: Class name in the return statement.
        lines: Beautified JS lines.
        start: Line to start searching from.

    Returns:
        List of argument expressions.
    """
    buf = ""
    for j in range(start, min(start + 50, len(lines))):
        buf += " " + lines[j]
        prefix = f"return new {cls}("
        idx = buf.find(prefix)
        if idx >= 0:
            inner_start = idx + len(prefix)
            depth = 1
            pos = inner_start
            while pos < len(buf) and depth > 0:
                if buf[pos] == "(":
                    depth += 1
                elif buf[pos] == ")":
                    depth -= 1
                pos += 1
            if depth != 0:
                continue
            raw = buf[inner_start : pos - 1]
            return _split_args_respecting_parens(raw)
    return []


def _resolve_wire_expr(arg: str, wire_map: dict[str, str]) -> str:
    """Resolve a parse argument to its wire byte expression.

    Args:
        arg: Argument expression from return new statement.
        wire_map: Map of local variable names to wire expressions.

    Returns:
        Resolved wire expression, or the original arg if no resolution.
    """
    stripped = arg.strip()
    if stripped in wire_map:
        return wire_map[stripped]
    if "a[" in stripped:
        return stripped
    return stripped


_WIRE_PATTERNS: list[str] = [
    r"\b(\w+)\s*=\s*(X\(a\[\d+\]\s*,\s*a\[\d+\]\))",
    r"\b(\w+)\s*=\s*(256\s*\*\s*\(256\s*\*\s*a\[\d+\][^,;)]+\))",
    r"\b(\w+)\s*=\s*(1\s*===\s*a\[\d+\])",
    r"\b(\w+)\s*=\s*(a\[\d+\]\s*>>\s*\d+\s*&\s*\d+)",
    r"\b(\w+)\s*=\s*(a\[\d+\]\s*&\s*\d+)",
]


def _extract_wire_map(block: str) -> dict[str, str]:
    """Extract local var -> wire byte expression mappings from a code block.

    Args:
        block: Concatenated JS source for the parse function.

    Returns:
        Dict mapping local variable names to wire byte expressions.
    """
    wire_map: dict[str, str] = {}
    for pattern in _WIRE_PATTERNS:
        for vm in re.finditer(pattern, block):
            wire_map[str(vm.group(1))] = str(vm.group(2))
    for vm in re.finditer(r"\b(\w+)\s*=\s*(a\[(\d+)\])(?![,\]])", block):
        key = str(vm.group(1))
        if key not in wire_map:
            wire_map[key] = str(vm.group(2))
    return wire_map


def _find_parse(cls: str, lines: list[str]) -> tuple[int, dict[str, str], list[str]]:
    """Find .h static parse function.

    Args:
        cls: Class name.
        lines: Beautified JS lines.

    Returns:
        Tuple of (line_number, wire_map, return_args).
        wire_map maps local var names to wire byte expressions.
    """
    pat = re.compile(rf"^\s*{re.escape(cls)}\.h\s*=\s*function\s*\(a\)")
    for i, line in enumerate(lines):
        if pat.match(line):
            block = ""
            for j in range(i, min(i + 50, len(lines))):
                block += " " + lines[j]
                if lines[j].strip().startswith("};"):
                    break

            wire_map = _extract_wire_map(block)
            ret_args = _extract_return_args(cls, lines, i)
            return i, wire_map, ret_args
    return -1, {}, []


def _find_execute(cls: str, lines: list[str]) -> list[tuple[str, str, str, int]]:
    """Find .prototype.h execute handler field assignments.

    Args:
        cls: Class name.
        lines: Beautified JS lines.

    Returns:
        List of (target_var, target_field, source_field, line_number).
    """
    pat = re.compile(rf"^\s*{re.escape(cls)}\.prototype\.h\s*=\s*function\s*\(a\)")
    results: list[tuple[str, str, str, int]] = []
    for i, line in enumerate(lines):
        if pat.match(line):
            for j in range(i + 1, min(i + 60, len(lines))):
                em = _EXEC_ASSIGN.match(lines[j])
                if em:
                    results.append((str(em.group(1)), str(em.group(2)), str(em.group(3)), j + 1))
                if lines[j].strip().startswith("};"):
                    break
            break
    return results


def _format_handler(
    char: str,
    cls: str,
    lines: list[str],
    out: list[str],
) -> None:
    """Format one V table handler trace.

    Args:
        char: Message type character.
        cls: Handler class name.
        lines: Beautified JS lines.
        out: Output lines accumulator.
    """
    hex_code = f"0x{ord(char):02X}"
    out.append(f"{'=' * 60}")
    out.append(f"V['{char}'] ({hex_code}) = {cls}")
    out.append(f"{'=' * 60}")

    _, ctor_params, ctor_fields = _find_constructor(cls, lines)
    if not ctor_params and not ctor_fields:
        out.append("  [constructor not found or empty]")
        out.append("")
        return

    _parse_line, wire_map, ret_args = _find_parse(cls, lines)

    resolved: dict[str, str] = {}
    for field, param in ctor_fields.items():
        if param in ctor_params:
            idx = ctor_params.index(param)
            if idx < len(ret_args):
                resolved[field] = _resolve_wire_expr(ret_args[idx], wire_map)

    args_preview = ", ".join(ret_args[:8])
    suffix = "..." if len(ret_args) > 8 else ""
    out.append(f"  Parse: {cls}.h(a) -> new {cls}({args_preview}{suffix})")
    for field, param in ctor_fields.items():
        idx = ctor_params.index(param) if param in ctor_params else -1
        wire = resolved.get(field, "?")
        arg_str = f"arg {idx}" if idx >= 0 else "?"
        out.append(f"    this.{field} = {param} ({arg_str}) <- {wire}")

    execute_assigns = _find_execute(cls, lines)
    if execute_assigns:
        out.append("  Execute: tank entity assignments")
        for tvar, tfield, sfield, _line_no in execute_assigns:
            wire = resolved.get(sfield, "?")
            semantic = XC_SEMANTICS.get(tfield, "")
            sem_str = f"  [{semantic}]" if semantic else ""
            out.append(f"    {tvar}.{tfield} = this.{sfield} <- {wire}{sem_str}")

    out.append("")


def trace_vtable(js_path: Path) -> str:
    """Trace all V table handlers and return formatted report.

    Args:
        js_path: Path to beautified tpclient.pretty.js.

    Returns:
        Formatted trace report as string.

    Raises:
        FileNotFoundError: If js_path does not exist.
    """
    text = _test_hooks.read_text(js_path)
    lines = text.split("\n")
    v_table = _find_v_table(lines)

    out: list[str] = [f"V table: {len(v_table)} handlers", ""]
    for char in sorted(v_table, key=ord):
        cls, _ = v_table[char]
        _format_handler(char, cls, lines, out)

    return "\n".join(out)


def main() -> None:
    """Trace V table handlers from beautified tpclient.js."""
    _test_hooks.setup_rich_logging(level="INFO")

    js_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tpclient.pretty.js")

    if not _test_hooks.path_exists(js_path):
        sys.stdout.write(f"File not found: {js_path}\n")
        raise SystemExit(1)

    report = trace_vtable(js_path)
    sys.stdout.write(report)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()


__all__ = [
    "main",
    "trace_vtable",
]
