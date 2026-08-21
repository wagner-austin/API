"""WeakAssertionRule edge cases: branch coverage and key-in-dict detection.

The rule's core detections are pinned in ``test_weak_assertion_rule.py``.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.test_quality_rules import WeakAssertionRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestBranchCoverage:
    """Additional tests for branch coverage."""

    def test_async_non_test_function_ignored(self, tmp_path: Path) -> None:
        """Async helper functions should be ignored."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "async def helper():\n    assert x is None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_is_not_none_with_non_none_constant(self, tmp_path: Path) -> None:
        """assert x is not 1 should not trigger is_not_none check."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert x is not 1\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        not_none_violations = [v for v in violations if v.kind == "weak-assertion-is-not-none"]
        assert len(not_none_violations) == 0

    def test_len_with_non_call_left(self, tmp_path: Path) -> None:
        """assert x > 0 (not len(x)) should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert x > 0\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_len_with_non_len_func(self, tmp_path: Path) -> None:
        """assert size(x) > 0 should not trigger len check."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert size(x) > 0\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_len_with_multiple_ops(self, tmp_path: Path) -> None:
        """assert 0 < len(x) < 10 should not trigger len check."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 0 < len(x) < 10\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_string_in_stdout(self, tmp_path: Path) -> None:
        """assert 'x' in result.stdout should trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'x' in result.stdout\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert any(v.kind == "weak-assertion-in-output" for v in violations)

    def test_string_in_non_output_attr(self, tmp_path: Path) -> None:
        """assert 'x' in result.data should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'x' in result.data\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        output_violations = [v for v in violations if v.kind == "weak-assertion-in-output"]
        assert len(output_violations) == 0

    def test_is_not_none_with_non_constant_comparator(self, tmp_path: Path) -> None:
        """assert x is not y should not trigger (non-constant)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert x is not y\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        not_none_violations = [v for v in violations if v.kind == "weak-assertion-is-not-none"]
        assert len(not_none_violations) == 0

    def test_len_with_non_constant_comparator(self, tmp_path: Path) -> None:
        """assert len(x) > y should not trigger (non-constant)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert len(x) > y\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_string_in_non_attribute(self, tmp_path: Path) -> None:
        """assert 'x' in result should not trigger (not attribute)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'x' in result\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        output_violations = [v for v in violations if v.kind == "weak-assertion-in-output"]
        assert len(output_violations) == 0

    def test_len_with_chained_comparison(self, tmp_path: Path) -> None:
        """assert len(x) > 0 < y should not trigger (chained comparison)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert len(x) > 0 < y\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0


class TestKeyInDictRule:
    """Tests for weak-assertion-key-in-dict detection.

    The rule only flags `assert "key" in d` when `d` is also accessed via
    subscript elsewhere in the function (proving it's a dict, not a set).
    """

    def test_detects_key_in_dict_without_value_check(self, tmp_path: Path) -> None:
        """assert 'key' in d when d is used as dict should trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"key": "value", "other": "x"}
    assert "key" in d
    x = d["other"]  # proves d is a dict
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 1
        assert '"key"' in key_violations[0].line

    def test_allows_key_in_dict_with_value_check(self, tmp_path: Path) -> None:
        """assert 'key' in d followed by assert d['key'] == value should pass."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"key": "value"}
    assert "key" in d
    assert d["key"] == "value"
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_detects_multiple_unverified_keys(self, tmp_path: Path) -> None:
        """Multiple unverified key checks should all be flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"a": 1, "b": 2, "c": 3}
    assert "a" in d
    assert "b" in d
    x = d["c"]  # proves d is a dict
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 2

    def test_allows_partial_verification(self, tmp_path: Path) -> None:
        """Only unverified keys should be flagged, verified ones pass."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"a": 1, "b": 2}
    assert "a" in d
    assert "b" in d
    assert d["a"] == 1
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 1
        assert '"b"' in key_violations[0].line

    def test_ignores_set_membership_check(self, tmp_path: Path) -> None:
        """assert 'key' in s when s is only used as set should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    kinds = {"error-type-a", "error-type-b"}
    assert "error-type-a" in kinds
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_ignores_non_string_key(self, tmp_path: Path) -> None:
        """assert 1 in d (integer key) should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {1: "value", 2: "other"}
    assert 1 in d
    x = d[2]
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_ignores_variable_key(self, tmp_path: Path) -> None:
        """assert key in d (variable key) should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    key = "foo"
    d = {"foo": "value"}
    assert key in d
    x = d["foo"]
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_ignores_attribute_dict(self, tmp_path: Path) -> None:
        """assert 'key' in obj.d should not trigger (attribute access)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    assert "key" in obj.data
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_ignores_chained_in_check(self, tmp_path: Path) -> None:
        """assert 'a' in 'b' in d (chained) should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    assert "a" in "abc" in some_list
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_nested_dict_access_verifies_outer_key(self, tmp_path: Path) -> None:
        """assert d['outer']['inner'] == x should verify 'outer' key."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"outer": {"inner": "value"}}
    assert "outer" in d
    assert d["outer"]["inner"] == "value"
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_real_world_pattern_request_in_payload(self, tmp_path: Path) -> None:
        """Real-world pattern: assert 'request' in payload without value check."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_rq_enqueuer_methods():
    payload = {"run_id": "r1", "request": {"device": "cpu"}}
    assert "request" in payload
    assert "run_id" in payload
    assert payload["run_id"] == "r1"
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        # "request" is not verified, "run_id" is verified
        # payload is used as dict (via payload["run_id"])
        assert len(key_violations) == 1
        assert '"request"' in key_violations[0].line

    def test_separate_functions_independent(self, tmp_path: Path) -> None:
        """Keys verified in one function don't satisfy checks in another."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_first():
    d = {"key": "value", "other": "x"}
    assert "key" in d
    x = d["other"]  # proves d is a dict

def test_second():
    d = {"key": "value"}
    assert d["key"] == "value"
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        # test_first should have violation, test_second should not
        assert len(key_violations) == 1
        assert "test_first" in key_violations[0].line

    def test_ignores_chained_comparisons_multiple_comparators(self, tmp_path: Path) -> None:
        """Chained comparisons with multiple comparators should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"a": 1}
    x = d["a"]
    assert "key" in d in some_other  # multiple comparators
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_variable_subscript_in_assert_does_not_verify(self, tmp_path: Path) -> None:
        """assert d[var] == x doesn't verify string key assertions."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"key": "value"}
    var = "key"
    assert d[var] == "value"  # variable subscript in assert
    assert "key" in d
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        # d is dict-like (via d[var] subscript)
        # "key" in d is not verified by d[var] (var is not a string constant)
        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 1

    def test_ignores_attribute_subscript_access(self, tmp_path: Path) -> None:
        """Attribute subscript like obj.attr["key"] should handle edge cases."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    x = obj.data["key"]  # attr.subscript pattern
    assert "other" in d
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        # Should not crash, d is not used as dict
        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0
