"""Tests for the launcher-to-payload export audit.

The filesystem cases write real modules under ``tmp_path`` and read them
back. Nothing here fakes a file: the module's whole job is deciding whether
source names a variable, so a test that handed it prepared strings would be
testing the caller's idea of the tree rather than the tree.
"""

from __future__ import annotations

import pathlib

import pytest

from platform_core.exported_env_audit import (
    EXPORT_KEYWORD,
    VENDORED_DIRECTORIES,
    ExportedVariable,
    ReaderSite,
    VariableAudit,
    audit_variables,
    decode_exported_variable,
    decode_reader_site,
    decode_variable_audit,
    describe_inert,
    encode_exported_variable,
    encode_reader_site,
    encode_variable_audit,
    inert_variables,
    is_vendored,
    modifies_existing,
    originated_names,
    payload_modules,
    reader_sites,
    shell_export_assignments,
    shell_exported_names,
    site_order,
    string_literals,
)
from platform_core.json_utils import JSONTypeError

_VARIABLE = ExportedVariable(name="HPC3_JOB_NAME", purpose="name the run")
_SITE = ReaderSite(path="libs/x/src/x/reader.py", line=7)


def _write(root: pathlib.Path, relative: str, source: str) -> pathlib.Path:
    """Write a module under a root, creating parents.

    Args:
        root: The tree to write into.
        relative: Path relative to the root.
        source: File contents.

    Returns:
        The path written.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


class TestExportKeyword:
    def test_it_is_the_shell_form_with_its_trailing_space(self) -> None:
        assert EXPORT_KEYWORD == "export "

    def test_virtualenvs_and_caches_are_vendored(self) -> None:
        assert ".venv" in VENDORED_DIRECTORIES
        assert "site-packages" in VENDORED_DIRECTORIES


class TestEncodeDecodeExportedVariable:
    def test_a_variable_survives_the_round_trip(self) -> None:
        assert decode_exported_variable(encode_exported_variable(_VARIABLE)) == _VARIABLE

    def test_encoding_carries_both_fields(self) -> None:
        assert encode_exported_variable(_VARIABLE) == {
            "name": "HPC3_JOB_NAME",
            "purpose": "name the run",
        }

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_exported_variable(["HPC3_JOB_NAME"])

    def test_a_missing_purpose_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_exported_variable({"name": "HPC3_JOB_NAME"})

    def test_a_non_string_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_exported_variable({"name": 3, "purpose": "name the run"})


class TestEncodeDecodeReaderSite:
    def test_a_site_survives_the_round_trip(self) -> None:
        assert decode_reader_site(encode_reader_site(_SITE)) == _SITE

    def test_a_non_integer_line_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_reader_site({"path": "a.py", "line": "seven"})

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_reader_site("a.py:7")


class TestEncodeDecodeVariableAudit:
    def test_an_audit_survives_the_round_trip(self) -> None:
        audit = VariableAudit(variable=_VARIABLE, readers=(_SITE,))
        assert decode_variable_audit(encode_variable_audit(audit)) == audit

    def test_an_audit_with_no_readers_survives_the_round_trip(self) -> None:
        audit = VariableAudit(variable=_VARIABLE, readers=())
        assert decode_variable_audit(encode_variable_audit(audit)) == audit

    def test_a_non_list_readers_field_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_variable_audit(
                {"variable": encode_exported_variable(_VARIABLE), "readers": "none"}
            )


class TestIsVendored:
    def test_a_virtualenv_path_is_vendored(self, tmp_path: pathlib.Path) -> None:
        assert is_vendored(tmp_path / ".venv" / "lib" / "mod.py")

    def test_first_party_source_is_not(self, tmp_path: pathlib.Path) -> None:
        assert not is_vendored(tmp_path / "src" / "pkg" / "mod.py")


class TestPayloadModules:
    def test_it_finds_modules_and_skips_vendored_trees(self, tmp_path: pathlib.Path) -> None:
        kept = _write(tmp_path, "pkg/mod.py", "X = 1\n")
        _write(tmp_path, ".venv/lib/vendored.py", "X = 1\n")
        _write(tmp_path, "pkg/notes.txt", "not python")
        assert payload_modules(tmp_path) == (kept,)

    def test_an_empty_tree_yields_nothing(self, tmp_path: pathlib.Path) -> None:
        assert payload_modules(tmp_path) == ()


class TestStringLiterals:
    def test_it_reports_literals_with_their_lines(self) -> None:
        assert string_literals('X = "ALPHA"\n') == (("ALPHA", 1),)

    def test_non_string_constants_are_ignored(self) -> None:
        assert string_literals("X = 3\n") == ()

    def test_a_comment_is_not_a_literal(self) -> None:
        assert string_literals("# ALPHA\nX = 1\n") == ()

    def test_unparsable_source_raises(self) -> None:
        with pytest.raises(SyntaxError):
            string_literals("def (:\n")


class TestShellExportParsing:
    def test_it_reads_name_and_value(self) -> None:
        assert shell_export_assignments('export A="1"\n') == (("A", "1"),)

    def test_a_line_that_is_not_an_export_is_skipped(self) -> None:
        assert shell_export_assignments("echo hello\n") == ()

    def test_an_export_without_an_assignment_is_skipped(self) -> None:
        assert shell_export_assignments("export A\n") == ()

    def test_names_are_sorted_and_deduplicated(self) -> None:
        assert shell_exported_names('export B="1"\nexport A="2"\nexport A="3"\n') == ("A", "B")


class TestModifiesExisting:
    def test_a_bare_self_reference_is_a_modification(self) -> None:
        assert modifies_existing("PATH", "/opt/bin:$PATH")

    def test_a_braced_self_reference_is_a_modification(self) -> None:
        assert modifies_existing("PATH", "/opt/bin:${PATH}")

    def test_referencing_a_different_variable_originates(self) -> None:
        assert not modifies_existing("HPC3_RESTART_COUNT", "${SLURM_RESTART_COUNT:-0}")

    def test_a_literal_value_originates(self) -> None:
        assert not modifies_existing("HPC3_JOB_NAME", "abl.arm-b-42")


class TestOriginatedNames:
    def test_a_self_extending_export_is_excluded(self) -> None:
        script = 'export PATH="/opt/bin:$PATH"\nexport HPC3_JOB_NAME="abl"\n'
        assert originated_names(script) == ("HPC3_JOB_NAME",)


class TestSiteOrder:
    def test_it_orders_by_path_then_line(self) -> None:
        assert site_order(_SITE) == ("libs/x/src/x/reader.py", 7)


class TestReaderSites:
    def test_it_finds_a_literal_naming_the_variable(self, tmp_path: pathlib.Path) -> None:
        module = _write(tmp_path, "pkg/reader.py", 'V = get_env("HPC3_JOB_NAME")\n')
        assert reader_sites("HPC3_JOB_NAME", (tmp_path,)) == (
            ReaderSite(path=module.as_posix(), line=1),
        )

    def test_a_module_not_mentioning_the_name_is_skipped(self, tmp_path: pathlib.Path) -> None:
        _write(tmp_path, "pkg/other.py", "X = 1\n")
        assert reader_sites("HPC3_JOB_NAME", (tmp_path,)) == ()

    def test_a_prose_mention_is_not_a_reader(self, tmp_path: pathlib.Path) -> None:
        """The name appears, so the cheap substring gate passes and the parse
        still refuses it -- which is the distinction the audit exists to make."""
        _write(tmp_path, "pkg/doc.py", '"""Set HPC3_JOB_NAME before running."""\n')
        assert reader_sites("HPC3_JOB_NAME", (tmp_path,)) == ()

    def test_results_are_ordered_across_roots(self, tmp_path: pathlib.Path) -> None:
        first = _write(tmp_path / "a", "pkg/one.py", 'N = "V"\n')
        second = _write(tmp_path / "b", "pkg/two.py", '\nN = "V"\n')
        found = reader_sites("V", (tmp_path / "b", tmp_path / "a"))
        assert [site["path"] for site in found] == sorted([first.as_posix(), second.as_posix()])


class TestAuditVariables:
    def test_a_read_variable_reports_its_reader(self, tmp_path: pathlib.Path) -> None:
        _write(tmp_path, "pkg/reader.py", 'V = get_env("HPC3_JOB_NAME")\n')
        audits = audit_variables((_VARIABLE,), (tmp_path,))
        assert len(audits[0]["readers"]) == 1
        assert inert_variables(audits) == ()

    def test_an_unread_variable_is_inert(self, tmp_path: pathlib.Path) -> None:
        audits = audit_variables((_VARIABLE,), (tmp_path,))
        assert audits[0]["readers"] == ()
        assert inert_variables(audits) == ("HPC3_JOB_NAME",)


class TestDescribeInert:
    def test_it_names_the_variable_and_the_missing_behaviour(self) -> None:
        audits = (VariableAudit(variable=_VARIABLE, readers=()),)
        described = describe_inert(audits)
        assert "HPC3_JOB_NAME" in described
        assert "name the run" in described

    def test_a_fully_read_audit_describes_nothing(self) -> None:
        audits = (VariableAudit(variable=_VARIABLE, readers=(_SITE,)),)
        assert describe_inert(audits) == ""
