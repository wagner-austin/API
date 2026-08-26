"""Tests for the renaming-alias guard rule.

The cases that matter are the ones that separate a rename from a re-export,
because getting that wrong in either direction is what makes a guard useless:
too strict and authors route around it, too loose and the thing it was written
for walks straight past.

The three real shapes it was written from are asserted verbatim -- they were
deleted on 2026-08-26 and this is what stops them coming back.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.passthrough_rules import PassthroughRule


def _write_src(tmp_path: Path, content: str, name: str = "module.py") -> Path:
    """Write content to a file inside a src directory.

    Args:
        tmp_path: Pytest temporary directory.
        content: Python source to write.
        name: File name to use.

    Returns:
        Path to the written file.
    """
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True, exist_ok=True)
    path = src / name
    path.write_text(content, encoding="utf-8")
    return path


def _kinds(path: Path) -> list[str]:
    """Run the rule over one file and return the violation kinds.

    Args:
        path: File to check.

    Returns:
        One kind string per violation, in order.
    """
    return [violation.kind for violation in PassthroughRule().run([path])]


def _lines(path: Path) -> list[str]:
    """Run the rule over one file and return the rendered violation lines.

    Args:
        path: File to check.

    Returns:
        One rendered line per violation, in order.
    """
    return [violation.line for violation in PassthroughRule().run([path])]


class TestTheThreeAliasesThisRuleWasWrittenFrom:
    """Each was deleted on 2026-08-26 after being found by its own comment."""

    def test_it_catches_the_platform_calendar_token_response(self, tmp_path: Path) -> None:
        path = _write_src(
            tmp_path,
            "from platform_core.oauth_types import OAuthTokenResponse\n"
            "GoogleTokenResponse = OAuthTokenResponse\n",
        )
        assert _kinds(path) == ["passthrough-alias"]

    def test_it_catches_the_music_wrapped_generate_request(self, tmp_path: Path) -> None:
        path = _write_src(
            tmp_path,
            "class GenerateRequest:\n    pass\n\nLastFmGenerate = GenerateRequest\n",
        )
        assert _kinds(path) == ["passthrough-alias"]

    def test_it_catches_a_config_facade_binding_many_at_once(self, tmp_path: Path) -> None:
        """The Model-Trainer settings module bound ten in a row."""
        path = _write_src(
            tmp_path,
            "from platform_core.config import ModelTrainerAppConfig, ModelTrainerSettings\n"
            "AppConfig = ModelTrainerAppConfig\n"
            "Settings = ModelTrainerSettings\n",
        )
        assert _kinds(path) == ["passthrough-alias", "passthrough-alias"]

    def test_the_message_names_what_to_use_instead(self, tmp_path: Path) -> None:
        """An author who cannot see the replacement edits the guard instead."""
        path = _write_src(
            tmp_path,
            "from platform_core.oauth_types import OAuthTokenResponse\n"
            "GoogleTokenResponse = OAuthTokenResponse\n",
        )
        assert _lines(path) == [
            "GoogleTokenResponse = OAuthTokenResponse -- use OAuthTokenResponse at the call sites"
        ]


class TestWhatIsNotARename:
    def test_a_constant_bound_to_a_constant_is_out_of_scope(self, tmp_path: Path) -> None:
        """A duplicated value is a different problem with a different fix."""
        path = _write_src(tmp_path, "from pkg.limits import MAX_RETRIES\nRETRY_CAP = MAX_RETRIES\n")
        assert _kinds(path) == []

    def test_a_lowercase_binding_is_out_of_scope(self, tmp_path: Path) -> None:
        path = _write_src(tmp_path, "from pkg.io import loader\nread = loader\n")
        assert _kinds(path) == []

    def test_a_same_name_re_export_is_not_a_rename(self, tmp_path: Path) -> None:
        """``import X as X`` is the sanctioned re-export idiom, and it is an
        import rather than an assignment, so there is no second spelling."""
        path = _write_src(
            tmp_path,
            "from platform_core.oauth_types import OAuthTokenResponse as OAuthTokenResponse\n",
        )
        assert _kinds(path) == []

    def test_a_value_this_module_never_bound_is_not_flagged(self, tmp_path: Path) -> None:
        """Without this the rule fires on any capitalised assignment, which is
        how a guard starts reporting things nobody can act on."""
        path = _write_src(tmp_path, "Alpha = Beta\n")
        assert _kinds(path) == []

    def test_an_assignment_from_a_call_is_not_an_alias(self, tmp_path: Path) -> None:
        path = _write_src(tmp_path, "class Thing:\n    pass\n\nDefault = Thing()\n")
        assert _kinds(path) == []

    def test_a_tuple_unpack_is_not_an_alias(self, tmp_path: Path) -> None:
        path = _write_src(tmp_path, "class Thing:\n    pass\n\nA = B = Thing\n")
        assert _kinds(path) == []

    def test_an_alias_nested_inside_a_function_is_a_local(self, tmp_path: Path) -> None:
        """Only module level ships a second public name."""
        path = _write_src(
            tmp_path,
            "class Thing:\n    pass\n\ndef build() -> None:\n    Local = Thing\n    del Local\n",
        )
        assert _kinds(path) == []


class TestScope:
    def test_a_test_file_is_not_checked(self, tmp_path: Path) -> None:
        """A local name inside a test is a fixture, not a shipped surface."""
        tests = tmp_path / "tests"
        tests.mkdir(parents=True, exist_ok=True)
        path = tests / "test_thing.py"
        path.write_text("class Thing:\n    pass\n\nAlias = Thing\n", encoding="utf-8")
        assert _kinds(path) == []

    def test_a_class_defined_here_counts_as_bindable(self, tmp_path: Path) -> None:
        path = _write_src(tmp_path, "class Thing:\n    pass\n\nAlias = Thing\n")
        assert _kinds(path) == ["passthrough-alias"]

    def test_an_aliased_import_counts_as_bindable(self, tmp_path: Path) -> None:
        path = _write_src(tmp_path, "from pkg.types import Thing as Renamed\nAlias = Renamed\n")
        assert _kinds(path) == ["passthrough-alias"]

    def test_a_plain_module_import_counts_as_bindable(self, tmp_path: Path) -> None:
        path = _write_src(tmp_path, "import Thing\nAlias = Thing\n")
        assert _kinds(path) == ["passthrough-alias"]

    def test_several_files_are_reported_together(self, tmp_path: Path) -> None:
        first = _write_src(tmp_path, "class Thing:\n    pass\n\nAlias = Thing\n", name="a.py")
        second = _write_src(tmp_path, "class Other:\n    pass\n\nAka = Other\n", name="b.py")
        assert len(PassthroughRule().run([first, second])) == 2

    def test_the_rule_reports_the_line_it_found(self, tmp_path: Path) -> None:
        path = _write_src(tmp_path, "class Thing:\n    pass\n\n\nAlias = Thing\n")
        assert [v.line_no for v in PassthroughRule().run([path])] == [5]

    def test_the_rule_is_named_for_its_kind(self) -> None:
        assert PassthroughRule().name == "passthrough"
