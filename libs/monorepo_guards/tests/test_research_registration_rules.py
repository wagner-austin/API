"""An entry point that produces a research record must be in the registry.

The rule under test exists because ``cartridge_qa_benchmark`` produced a
programme headline while appearing in no run document, no registry entry and
no tracked artifact. These tests pin the predicate that would have caught it,
and -- more importantly -- pin the three ways such a predicate goes wrong: by
convicting a report module that only reads records back, by crediting one
measurement with another's registry entry because one name contains the other,
and by accepting a module because its bare stem happens to be an English word
the registry uses in prose.
"""

from __future__ import annotations

import pathlib

from monorepo_guards.config import GuardConfig
from monorepo_guards.research_registration_rules import (
    REGISTRY_RELATIVE_PATH,
    ResearchRegistrationRule,
)

#: A module that brings a record into existence: it imports the factory and
#: the encoder, which is how ``cartridge_qa_benchmark`` and ``score_baseline``
#: both read.
_PRODUCER_SOURCE = (
    "from platform_core.run_record import Observation, RunRecord, encode_run_record, run_record\n"
    "\n"
    "def main() -> int: ...\n"
)

#: A module that READS records back and annotates what it read. The rule must
#: not fire on this: ``sdpa_benchmark_report`` is exactly this shape.
_REPORTER_SOURCE = (
    "from platform_core.run_record import RunRecord\n"
    "from model_trainer.cli.record_reports import read_run_records\n"
    "\n"
    "def main() -> int: ...\n"
)


def _write(root: pathlib.Path, relative: str, body: str) -> pathlib.Path:
    """Write a source file inside a package layout.

    Args:
        root: Temporary root standing in for the monorepo.
        relative: Path under the root, e.g. ``"services/T/src/t/cli/a.py"``.
        body: File contents.

    Returns:
        The path written.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _write_registry(root: pathlib.Path, body: str) -> pathlib.Path:
    """Write the research registry at the location the rule reads.

    Args:
        root: Temporary monorepo root.
        body: Registry contents.

    Returns:
        The path written.
    """
    path = root.joinpath(*REGISTRY_RELATIVE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _config(root: pathlib.Path) -> GuardConfig:
    """Build a config whose monorepo root is the temporary tree.

    Args:
        root: Temporary monorepo root.

    Returns:
        A config the rule can read the registry through.
    """
    return GuardConfig(
        root=root,
        monorepo_root=root,
        directories=("src", "tests", "scripts"),
        exclude_parts=(),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )


class TestAProducerThatNobodyRegistered:
    """The defect the rule is named for."""

    def test_an_unregistered_producing_entry_point_is_a_violation(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A surface that makes no registry claim cannot be reviewed.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/cartridge_qa_benchmark.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "# Research index\n\nNothing here names it.\n")

        found = ResearchRegistrationRule(_config(tmp_path)).run([path])

        assert [v.kind for v in found] == ["research-entry-point-unregistered"]

    def test_the_violation_names_the_qualified_path_and_points_at_its_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A message that names neither is a message nobody acts on.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/cartridge_qa_benchmark.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "# Research index\n")

        found = ResearchRegistrationRule(_config(tmp_path)).run([path])

        assert len(found) == 1
        assert found[0].file == path
        assert found[0].line_no == 1
        assert "t.cli.cartridge_qa_benchmark" in found[0].line

    def test_naming_the_qualified_path_clears_it(self, tmp_path: pathlib.Path) -> None:
        """The remedy must be a registry entry, and it must actually work.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/cartridge_qa_benchmark.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "- **Runs:** `t.cli.cartridge_qa_benchmark`\n")

        assert ResearchRegistrationRule(_config(tmp_path)).run([path]) == []

    def test_the_brace_notation_the_registry_actually_uses_is_expanded(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``pkg.cli.{a, b}`` is how the registry groups commands.

        A reader treating that as one opaque token would demand every entry be
        rewritten, so the rule reads the notation the file already uses.

        Args:
            tmp_path: Temporary monorepo root.
        """
        first = _write(tmp_path, "services/T/src/t/cli/gemm_benchmark.py", _PRODUCER_SOURCE)
        second = _write(tmp_path, "services/T/src/t/cli/probe_ladder.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "- **Runs:** `t.cli.{gemm_benchmark, probe_ladder}`\n")

        assert ResearchRegistrationRule(_config(tmp_path)).run([first, second]) == []


class TestTheThingsTheRuleMustNotConvict:
    """A rule that fires on correct code is one an operator learns to skip."""

    def test_a_report_module_that_only_reads_records_is_not_a_producer(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Annotating a record you read back does not bring one into existence.

        This is the false positive that keying on the ``RunRecord`` TYPE would
        have produced across every ``*_report`` module in the workspace.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/sdpa_benchmark_report.py", _REPORTER_SOURCE)
        _write_registry(tmp_path, "# Research index\n")

        assert ResearchRegistrationRule(_config(tmp_path)).run([path]) == []

    def test_a_library_module_that_builds_records_is_not_an_entry_point(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A registry entry names the command, not every module beneath it.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "services/T/src/t/core/services/training/run_records.py",
            _PRODUCER_SOURCE,
        )
        _write_registry(tmp_path, "# Research index\n")

        assert ResearchRegistrationRule(_config(tmp_path)).run([path]) == []

    def test_a_private_module_under_cli_is_not_an_entry_point(self, tmp_path: pathlib.Path) -> None:
        """Hook seams are not invoked by a run document and cannot be named.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/_measurement_hooks.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "# Research index\n")

        assert ResearchRegistrationRule(_config(tmp_path)).run([path]) == []

    def test_the_package_defining_the_vocabulary_is_exempt(self, tmp_path: pathlib.Path) -> None:
        """``platform_core`` owns the record types rather than doing research.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path, "libs/platform_core/src/platform_core/cli/emit.py", _PRODUCER_SOURCE
        )
        _write_registry(tmp_path, "# Research index\n")

        assert ResearchRegistrationRule(_config(tmp_path)).run([path]) == []

    def test_an_entry_point_outside_the_src_layout_still_gets_a_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A tool laid out without ``src`` must still be nameable.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "scripts/cli/batch.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "- **Runs:** `cli.batch`\n")

        assert ResearchRegistrationRule(_config(tmp_path)).run([path]) == []


class TestANameThatIsAlsoAnOrdinaryWord:
    """The false negative that shipped in this rule's first version.

    ``code_style_eval.cli.compare`` passed because the word "compare" occurs
    twice in the real registry as English prose, and ``scripts.batch`` had the
    same exposure on "batch". A module that satisfies a check by sharing a
    spelling with a sentence has not been checked at all.
    """

    def test_the_registry_using_the_stem_as_prose_does_not_register_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Prose is not a declaration.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "tools/E/src/e/cli/compare.py", _PRODUCER_SOURCE)
        _write_registry(
            tmp_path,
            "### code-style\n\nThe entry emits a table so a reader can compare\n"
            "two arms, and compare them again after a rebuild.\n",
        )

        found = ResearchRegistrationRule(_config(tmp_path)).run([path])

        assert [v.kind for v in found] == ["research-entry-point-unregistered"]
        assert "e.cli.compare" in found[0].line

    def test_the_qualified_path_registers_it_where_the_bare_word_did_not(
        self, tmp_path: pathlib.Path
    ) -> None:
        """And the remedy is to name the command, which prose cannot fake.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "tools/E/src/e/cli/compare.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "### code-style\n\n- **Runs:** `e.cli.compare`\n")

        assert ResearchRegistrationRule(_config(tmp_path)).run([path]) == []


class TestOneNameInsideAnother:
    """The bug a substring test would have shipped."""

    def test_a_longer_registered_name_does_not_register_the_shorter_one(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``gemm_probe`` is a substring of ``legacy_gemm_probe``.

        Crediting the first with the second's entry would silently give one
        measurement another measurement's paperwork.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/gemm_probe.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "- **Runs:** `t.cli.legacy_gemm_probe`\n")

        found = ResearchRegistrationRule(_config(tmp_path)).run([path])

        assert [v.kind for v in found] == ["research-entry-point-unregistered"]

    def test_the_shorter_registered_name_does_not_register_the_longer_one(
        self, tmp_path: pathlib.Path
    ) -> None:
        """And the containment runs the other way too.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/legacy_gemm_probe.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "- **Runs:** `t.cli.gemm_probe`\n")

        found = ResearchRegistrationRule(_config(tmp_path)).run([path])

        assert [v.kind for v in found] == ["research-entry-point-unregistered"]

    def test_another_package_with_the_same_command_name_does_not_register_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Two projects may both ship a ``compare``; the package disambiguates.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "tools/E/src/e/cli/compare.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "- **Runs:** `other_package.cli.compare`\n")

        found = ResearchRegistrationRule(_config(tmp_path)).run([path])

        assert [v.kind for v in found] == ["research-entry-point-unregistered"]


class TestTheBraceNotationsEdges:
    """Malformed grouping must not silently register anything."""

    def test_an_unclosed_group_registers_nothing_from_it(self, tmp_path: pathlib.Path) -> None:
        """A truncated entry is not a declaration.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/gemm_benchmark.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "- **Runs:** `t.cli.{gemm_benchmark, probe_ladder\n")

        found = ResearchRegistrationRule(_config(tmp_path)).run([path])

        assert [v.kind for v in found] == ["research-entry-point-unregistered"]

    def test_a_brace_not_preceded_by_a_dotted_prefix_registers_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``word{a}`` is prose punctuation, not the registry's notation.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/gemm_benchmark.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "see runs{gemm_benchmark} for the arm\n")

        found = ResearchRegistrationRule(_config(tmp_path)).run([path])

        assert [v.kind for v in found] == ["research-entry-point-unregistered"]

    def test_an_empty_member_in_a_group_is_skipped_and_the_rest_register(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A stray comma must not register the prefix on its own.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/gemm_benchmark.py", _PRODUCER_SOURCE)
        _write_registry(tmp_path, "- **Runs:** `t.cli.{, gemm_benchmark, }`\n")

        assert ResearchRegistrationRule(_config(tmp_path)).run([path]) == []


class TestWhenTheRegistryItselfIsAbsent:
    """A missing registry must fail loudly rather than pass vacuously."""

    def test_producers_with_no_registry_file_is_a_violation(self, tmp_path: pathlib.Path) -> None:
        """Otherwise deleting the registry would turn every check green.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/cartridge_qa_benchmark.py", _PRODUCER_SOURCE)

        found = ResearchRegistrationRule(_config(tmp_path)).run([path])

        assert [v.kind for v in found] == ["research-registry-missing"]
        assert found[0].file == path

    def test_no_producers_and_no_registry_is_not_a_violation(self, tmp_path: pathlib.Path) -> None:
        """A package doing no research owes the registry nothing.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "services/T/src/t/cli/sdpa_benchmark_report.py", _REPORTER_SOURCE)

        assert ResearchRegistrationRule(_config(tmp_path)).run([path]) == []
