"""Tests for scripts.trace_vtable."""

from __future__ import annotations

from pathlib import Path

from scripts.trace_vtable import (
    _extract_return_args,
    _extract_wire_map,
    _find_constructor,
    _find_execute,
    _find_parse,
    _find_v_table,
    _format_handler,
    _resolve_wire_expr,
    trace_vtable,
)

from scripts import _test_hooks


class TestFindVTable:
    """Tests for V table discovery."""

    def test_finds_quoted_assignment(self) -> None:
        """Finds V["x"] = ClassName."""
        lines = ['    V["!"] = Tf;']
        result = _find_v_table(lines)
        assert "!" in result
        assert result["!"] == ("Tf", 0)

    def test_finds_dot_assignment(self) -> None:
        """Finds V.S = ClassName."""
        lines = ["    V.S = Gg;"]
        result = _find_v_table(lines)
        assert "S" in result
        assert result["S"] == ("Gg", 0)

    def test_ignores_non_printable(self) -> None:
        """Skips entries where matched char is not printable."""
        lines = ['    V["\x01"] = Bad;']
        result = _find_v_table(lines)
        assert len(result) == 0

    def test_multiple_handlers(self) -> None:
        """Finds multiple handlers on separate lines."""
        lines = ['    V["!"] = Tf;', '    V["."] = Og;', "    V.G = Lg;"]
        result = _find_v_table(lines)
        assert len(result) == 3


class TestFindConstructor:
    """Tests for constructor discovery."""

    def test_finds_constructor_with_fields(self) -> None:
        """Extracts params and field assignments."""
        lines = [
            "    function Tf(a, b, c) {",
            "        this.i = a;",
            "        this.j = b;",
            "        this.k = c",
            "    }",
        ]
        line, params, fields = _find_constructor("Tf", lines)
        assert line == 0
        assert params == ["a", "b", "c"]
        assert fields == {"i": "a", "j": "b", "k": "c"}

    def test_finds_constructor_without_closing_brace(self) -> None:
        """Extracts fields even when no closing brace in scan window."""
        lines = [
            "    function Tf(a, b) {",
            "        this.i = a;",
            "        this.j = b;",
            "        // no closing brace in next 30 lines",
        ]
        _, params, fields = _find_constructor("Tf", lines)
        assert params == ["a", "b"]
        assert fields == {"i": "a", "j": "b"}

    def test_skips_non_field_lines(self) -> None:
        """Skips lines that are neither field assignments nor closing brace."""
        lines = [
            "    function Tf(a, b) {",
            "        var x = 5;",
            "        this.i = a;",
            "        some_call();",
            "    }",
        ]
        _, _, fields = _find_constructor("Tf", lines)
        assert fields == {"i": "a"}

    def test_returns_empty_when_not_found(self) -> None:
        """Returns (-1, [], {}) when class not found."""
        lines = ["    function Other(a) { this.x = a }"]
        line, params, fields = _find_constructor("Missing", lines)
        assert line == -1
        assert params == []
        assert fields == {}


class TestExtractReturnArgs:
    """Tests for return new Cls(...) argument extraction."""

    def test_single_line_return(self) -> None:
        """Extracts args from single-line return."""
        lines = [
            "    Tf.h = function(a) {",
            "        return new Tf(a[0], X(a[1], a[2]), a[3])",
            "    };",
        ]
        args = _extract_return_args("Tf", lines, 0)
        assert len(args) == 3
        assert args[0] == "a[0]"
        assert "X(a[1]" in args[1]
        assert args[2] == "a[3]"

    def test_nested_parens(self) -> None:
        """Handles nested parentheses in arguments."""
        lines = [
            "    Og.h = function(a) {",
            "        return new Og(a[0], 256 * (256 * a[1] + a[2]) + a[3])",
            "    };",
        ]
        args = _extract_return_args("Og", lines, 0)
        assert len(args) == 2

    def test_returns_empty_when_no_return(self) -> None:
        """Returns [] when no return new found."""
        lines = ["    Tf.h = function(a) {", "    };"]
        args = _extract_return_args("Tf", lines, 0)
        assert args == []

    def test_unbalanced_parens_skips(self) -> None:
        """Skips when parens don't balance within the scan window."""
        lines = [
            "    Tf.h = function(a) {",
            "        return new Tf(a[0], (((",
        ]
        args = _extract_return_args("Tf", lines, 0)
        assert args == []


class TestExtractWireMap:
    """Tests for wire byte expression extraction."""

    def test_extracts_x_call(self) -> None:
        """Extracts X(a[N], a[N]) patterns."""
        block = "var b = X(a[0], a[1]);"
        result = _extract_wire_map(block)
        assert result["b"] == "X(a[0], a[1])"

    def test_extracts_bit_shift(self) -> None:
        """Extracts bit shift patterns."""
        block = "var c = a[3] >> 4 & 15;"
        result = _extract_wire_map(block)
        assert result["c"] == "a[3] >> 4 & 15"

    def test_extracts_direct_byte(self) -> None:
        """Extracts direct a[N] references."""
        block = "var d = a[5];"
        result = _extract_wire_map(block)
        assert result["d"] == "a[5]"

    def test_extracts_boolean_check(self) -> None:
        """Extracts 1 === a[N] patterns."""
        block = "var e = 1 === a[11];"
        result = _extract_wire_map(block)
        assert result["e"] == "1 === a[11]"


class TestResolveWireExpr:
    """Tests for wire expression resolution."""

    def test_resolves_known_var(self) -> None:
        """Resolves a mapped variable."""
        wire_map = {"b": "X(a[0], a[1])"}
        assert _resolve_wire_expr("b", wire_map) == "X(a[0], a[1])"

    def test_passes_through_direct_ref(self) -> None:
        """Passes through expressions containing a[."""
        assert _resolve_wire_expr("a[5]", {}) == "a[5]"

    def test_passes_through_unknown(self) -> None:
        """Returns original for unknown expressions."""
        assert _resolve_wire_expr("t", {}) == "t"


class TestFindParse:
    """Tests for parse function discovery."""

    def test_finds_parse_with_wire_map(self) -> None:
        """Finds .h function and extracts wire mappings."""
        lines = [
            "    Tf.h = function(a) {",
            "        var b = X(a[0], a[1]);",
            "        return new Tf(b, a[2])",
            "    };",
        ]
        line, wire_map, ret_args = _find_parse("Tf", lines)
        assert line == 0
        assert "b" in wire_map
        assert len(ret_args) == 2

    def test_parse_without_terminator(self) -> None:
        """Handles parse function that lacks }; within scan window."""
        lines = [
            "    Tf.h = function(a) {",
            "        var b = a[0];",
            "        return new Tf(b)",
        ]
        line, wire_map, ret_args = _find_parse("Tf", lines)
        assert line == 0
        assert "b" in wire_map
        assert ret_args == ["b"]

    def test_returns_empty_when_not_found(self) -> None:
        """Returns (-1, {}, []) when not found."""
        lines = ["    Other.h = function(a) { };"]
        line, wire_map, ret_args = _find_parse("Missing", lines)
        assert line == -1
        assert wire_map == {}
        assert ret_args == []


class TestFindExecute:
    """Tests for execute handler discovery."""

    def test_finds_field_assignments(self) -> None:
        """Finds b.field = this.field assignments."""
        lines = [
            "    Tf.prototype.h = function(a) {",
            "        var b = ud(a.P, this.o);",
            "        b.h = this.i;",
            "        b.l = this.s;",
            "    };",
        ]
        assigns = _find_execute("Tf", lines)
        assert len(assigns) == 2
        assert assigns[0] == ("b", "h", "i", 3)
        assert assigns[1] == ("b", "l", "s", 4)

    def test_execute_without_terminator(self) -> None:
        """Finds assignments even without }; terminator."""
        lines = [
            "    Tf.prototype.h = function(a) {",
            "        b.h = this.i;",
        ]
        assigns = _find_execute("Tf", lines)
        assert len(assigns) == 1

    def test_returns_empty_when_not_found(self) -> None:
        """Returns [] when prototype.h not found."""
        assigns = _find_execute("Missing", ["no match"])
        assert assigns == []


class TestFormatHandler:
    """Tests for handler formatting."""

    def test_formats_complete_handler(self) -> None:
        """Formats a handler with constructor, parse, and execute."""
        lines = [
            "    function Tf(a, b) {",
            "        this.i = a;",
            "        this.j = b",
            "    }",
            "    Tf.h = function(a) {",
            "        return new Tf(a[0], a[1])",
            "    };",
            "    Tf.prototype.h = function(a) {",
            "        var b = ud(a.P, 1);",
            "        b.h = this.i;",
            "    };",
        ]
        out: list[str] = []
        _format_handler("!", "Tf", lines, out)
        text = "\n".join(out)
        assert "V['!']" in text
        assert "0x21" in text
        assert "this.i = a" in text
        assert "b.h = this.i" in text

    def test_formats_handler_with_unresolvable_params(self) -> None:
        """Formats handler where param unknown and ret_args shorter than fields."""
        lines = [
            "    function Tf(a, b, c) {",
            "        this.i = a;",
            "        this.j = b;",
            "        this.k = z;",
            "    }",
            "    Tf.h = function(a) {",
            "        return new Tf(a[0])",
            "    };",
        ]
        out: list[str] = []
        _format_handler("!", "Tf", lines, out)
        text = "\n".join(out)
        assert "this.j = b (arg 1) <- ?" in text
        assert "this.k = z (?) <- ?" in text

    def test_formats_missing_constructor(self) -> None:
        """Handles missing constructor gracefully."""
        out: list[str] = []
        _format_handler("X", "Missing", ["no match"], out)
        assert any("not found" in line for line in out)


class TestSplitArgs:
    """Tests for paren-aware arg splitting."""

    def test_simple_split(self) -> None:
        """Splits simple comma-separated args."""
        from scripts.trace_vtable import _split_args_respecting_parens

        result = _split_args_respecting_parens("a, b, c")
        assert result == ["a", "b", "c"]

    def test_nested_parens(self) -> None:
        """Preserves nested function calls."""
        from scripts.trace_vtable import _split_args_respecting_parens

        result = _split_args_respecting_parens("X(a[0], a[1]), a[2]")
        assert result == ["X(a[0], a[1])", "a[2]"]

    def test_empty_string(self) -> None:
        """Returns empty list for empty input."""
        from scripts.trace_vtable import _split_args_respecting_parens

        assert _split_args_respecting_parens("") == []

    def test_single_arg(self) -> None:
        """Handles single argument."""
        from scripts.trace_vtable import _split_args_respecting_parens

        assert _split_args_respecting_parens("a[0]") == ["a[0]"]


class TestTraceVtable:
    """Integration test for full trace."""

    def setup_method(self) -> None:
        """Store original hooks."""
        self._original_read_text = _test_hooks.read_text
        self._original_path_exists = _test_hooks.path_exists
        self._original_setup_logging = _test_hooks.setup_rich_logging

    def teardown_method(self) -> None:
        """Restore original hooks."""
        _test_hooks.read_text = self._original_read_text
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.setup_rich_logging = self._original_setup_logging

    def _make_fake_js(self) -> str:
        """Create minimal fake JS with a V table handler."""
        return "\n".join(
            [
                "    function Rg(a, b) {",
                "        this.j = a;",
                "        this.i = b",
                "    }",
                "    Rg.h = function(a) {",
                "        return new Rg(X(a[0], a[1]), a[2])",
                "    };",
                "    Rg.prototype.h = function(a) {",
                "    };",
                '    V["D"] = Rg;',
            ]
        )

    def test_traces_simple_vtable(self) -> None:
        """Traces a minimal V table from fake JS content."""

        def fake_read_text(path: Path) -> str:
            return self._make_fake_js()

        _test_hooks.read_text = fake_read_text
        result = trace_vtable(Path("fake.js"))
        assert "V table: 1 handlers" in result
        assert "V['D']" in result
        assert "0x44" in result

    def test_main_file_not_found(self) -> None:
        """main() exits with SystemExit(1) when file not found."""
        import io
        import sys as _sys

        import pytest
        from scripts.trace_vtable import main

        def fake_path_exists(path: Path) -> bool:
            return False

        def fake_logging(level: _test_hooks.LogLevel) -> None:
            pass

        _test_hooks.path_exists = fake_path_exists
        _test_hooks.setup_rich_logging = fake_logging

        old_argv = _sys.argv
        old_stdout = _sys.stdout
        try:
            _sys.argv = ["trace_vtable"]
            _sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
            with pytest.raises(SystemExit, match="1"):
                main()
        finally:
            _sys.argv = old_argv
            _sys.stdout = old_stdout

    def test_main_success(self) -> None:
        """main() writes report to stdout."""
        import io
        import sys as _sys

        from scripts.trace_vtable import main

        fake_js = self._make_fake_js()

        def fake_path_exists(path: Path) -> bool:
            return True

        def fake_read_text(path: Path) -> str:
            return fake_js

        def fake_logging(level: _test_hooks.LogLevel) -> None:
            pass

        _test_hooks.path_exists = fake_path_exists
        _test_hooks.read_text = fake_read_text
        _test_hooks.setup_rich_logging = fake_logging

        old_argv = _sys.argv
        old_stdout = _sys.stdout
        buf = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        try:
            _sys.argv = ["trace_vtable"]
            _sys.stdout = buf
            main()
            buf.seek(0)
            output = buf.read()
            assert "V table:" in output
        finally:
            _sys.argv = old_argv
            _sys.stdout = old_stdout

    def test_runpy_main_guard(self) -> None:
        """Covers if __name__ == '__main__' via runpy.

        ``scripts.trace_vtable`` is already imported at module top, so
        ``run_module`` would emit a RuntimeWarning about re-executing
        an imported module. Drop it from ``sys.modules`` first so
        runpy executes a clean module under ``__main__``.
        """
        import runpy
        import sys as _sys

        fake_js = self._make_fake_js()

        def fake_path_exists(path: Path) -> bool:
            return True

        def fake_read_text(path: Path) -> str:
            return fake_js

        def fake_logging(level: _test_hooks.LogLevel) -> None:
            pass

        _test_hooks.path_exists = fake_path_exists
        _test_hooks.read_text = fake_read_text
        _test_hooks.setup_rich_logging = fake_logging

        old_argv = _sys.argv
        saved = _sys.modules.pop("scripts.trace_vtable", None)
        try:
            _sys.argv = ["trace_vtable"]
            runpy.run_module("scripts.trace_vtable", run_name="__main__")
        finally:
            _sys.argv = old_argv
            if saved is not None:
                _sys.modules["scripts.trace_vtable"] = saved
