"""Tests for the literal-set drift rule, over both sets it is registered for.

The motivating case is ``_flags_a_literal_left_behind_by_a_widened_tuple``:
that is what the source looks like the moment after a third member is added to
the declaring tuple and one of the dozen inline ``Literal`` annotations is not
updated with it. mypy accepts that state, because the annotations are
independent of each other and of the tuple.

Most cases below drive the ``corpus_format`` instance, because that is the set
the AST handling was written against. ``TestTheStrategyNameSet`` at the end
drives the second registration, so the parameterisation is covered by a set
with a different declaring module, tuple name and field name -- not merely
asserted to exist.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.literal_set_rules import CORPUS_FORMAT_SET, LiteralSetRule
from tests._literal_set_support import DECLARING_PACKAGE, config_for, write_source

# Taken from the rule rather than restated. These fixtures exercise the
# REAL declaring path, so a change to it -- such as qualifying it by
# package, which is what stopped this rule firing on Art-Trainer's
# unrelated core/contracts/dataset.py -- moves the fixtures with it
# instead of breaking fourteen tests that were never about the path.
_DECLARING = CORPUS_FORMAT_SET.defining_module


def _declaring(tmp_path: Path, members: str) -> Path:
    """Write a stand-in declaring module binding the tuple.

    Args:
        tmp_path: Directory to write into.
        members: The tuple's contents, as source.

    Returns:
        The path written.
    """
    return write_source(
        tmp_path / DECLARING_PACKAGE / _DECLARING,
        f'from typing import Literal\n\nCORPUS_FORMATS: tuple[Literal["lines", '
        f'"documents"], ...] = ({members})\n',
    )


class TestDrift:
    """A Literal that no longer names what the tuple declares."""

    def test_it_flags_a_literal_left_behind_by_a_widened_tuple(self, tmp_path: Path) -> None:
        """The exact state mypy accepts and this rule exists to refuse."""
        declaring = _declaring(tmp_path, '"lines", "documents", "fim"')
        stale = write_source(
            tmp_path / "contracts" / "queue.py",
            "from typing import Literal\n\n\nclass P:\n"
            '    corpus_format: Literal["lines", "documents"]\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, stale])

        kinds = [v.kind for v in violations]
        assert kinds == ["corpus-format-literal-drift"]
        assert "documents, fim, lines" in violations[0].line

    def test_it_flags_a_stale_parameter_annotation(self, tmp_path: Path) -> None:
        """Signatures drift the same way field annotations do."""
        declaring = _declaring(tmp_path, '"lines", "documents", "fim"')
        stale = write_source(
            tmp_path / "reader.py",
            "from typing import Literal\n\n\n"
            'def read(corpus_format: Literal["lines"]) -> None:\n    return None\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, stale])

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_it_flags_a_stale_return_annotation(self, tmp_path: Path) -> None:
        """A narrowing function's return is the third drifting shape."""
        declaring = _declaring(tmp_path, '"lines", "documents", "fim"')
        stale = write_source(
            tmp_path / "narrow.py",
            "from typing import Literal\n\n\n"
            'def as_corpus_format(raw: str) -> Literal["lines", "documents"]:\n'
            '    return "lines"\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, stale])

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_the_message_names_both_sets(self, tmp_path: Path) -> None:
        """A drift report is only actionable if it says what disagrees."""
        declaring = _declaring(tmp_path, '"lines", "documents", "fim"')
        stale = write_source(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    corpus_format: Literal["lines"]\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, stale])

        assert "names lines" in violations[0].line
        assert "documents, fim, lines" in violations[0].line


class TestAgreement:
    """The rule stays silent when nothing has drifted."""

    def test_a_matching_literal_passes(self, tmp_path: Path) -> None:
        """The state the repository is actually in."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        matching = write_source(
            tmp_path / "contracts" / "queue.py",
            "from typing import Literal\n\n\nclass P:\n"
            '    corpus_format: Literal["lines", "documents"]\n',
        )

        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, matching]) == []
        )

    def test_member_order_does_not_matter(self, tmp_path: Path) -> None:
        """The comparison is over sets, so reordering is not drift."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        reordered = write_source(
            tmp_path / "q.py",
            "from typing import Literal\n\n\nclass P:\n"
            '    corpus_format: Literal["documents", "lines"]\n',
        )

        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, reordered])
            == []
        )

    def test_a_single_member_literal_subscript_is_read(self, tmp_path: Path) -> None:
        """``Literal["lines"]`` has no ast.Tuple slice, and must still parse.

        The declaring tuple needs its trailing comma for the same reason:
        ``("lines")`` is a string in parentheses, not a one-member tuple.
        """
        declaring = _declaring(tmp_path, '"lines",')
        single = write_source(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    corpus_format: Literal["lines"]\n',
        )

        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, single]) == []
        )

    def test_a_qualified_literal_is_read(self, tmp_path: Path) -> None:
        """``typing.Literal[...]`` is an Attribute base, not a Name."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        qualified = write_source(
            tmp_path / "q.py",
            "import typing\n\n\nclass P:\n"
            '    corpus_format: typing.Literal["lines", "documents"]\n',
        )

        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, qualified])
            == []
        )

    def test_a_non_literal_annotation_is_ignored(self, tmp_path: Path) -> None:
        """The manifest carries the format as a plain ``str`` by convention."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        plain = write_source(
            tmp_path / "models.py",
            "class M:\n    corpus_format: str\n",
        )

        assert LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, plain]) == []

    def test_a_subscripted_non_literal_annotation_is_ignored(self, tmp_path: Path) -> None:
        """``Sequence[...]`` subscripts too, and names no format."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        sequence = write_source(
            tmp_path / "q.py",
            "from collections.abc import Sequence\n\n\nclass P:\n"
            "    corpus_format: Sequence[str]\n",
        )

        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, sequence]) == []
        )

    def test_a_non_string_literal_is_ignored(self, tmp_path: Path) -> None:
        """A Literal of ints names no corpus format."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        numeric = write_source(
            tmp_path / "q.py",
            "from typing import Literal\n\n\nclass P:\n    corpus_format: Literal[1, 2]\n",
        )

        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, numeric]) == []
        )

    def test_an_unrelated_field_is_ignored(self, tmp_path: Path) -> None:
        """Only annotations on the format's own name are compared."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        other = write_source(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    optimizer: Literal["adamw", "sgd"]\n',
        )

        assert LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, other]) == []

    def test_an_attribute_target_is_compared(self, tmp_path: Path) -> None:
        """``self.corpus_format: Literal[...]`` is an ast.Attribute target."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        attribute = write_source(
            tmp_path / "q.py",
            "from typing import Literal\n\n\nclass P:\n    def __init__(self) -> None:\n"
            '        self.corpus_format: Literal["lines"] = "lines"\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run(
            [declaring, attribute]
        )

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_an_async_function_parameter_is_compared(self, tmp_path: Path) -> None:
        """Async definitions are a separate AST node and must be walked too."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        asynchronous = write_source(
            tmp_path / "q.py",
            "from typing import Literal\n\n\n"
            'async def read(corpus_format: Literal["lines"]) -> None:\n    return None\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run(
            [declaring, asynchronous]
        )

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_a_keyword_only_parameter_is_compared(self, tmp_path: Path) -> None:
        """Keyword-only args live in their own list on ast.arguments."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        kwonly = write_source(
            tmp_path / "q.py",
            "from typing import Literal\n\n\n"
            'def read(*, corpus_format: Literal["lines"]) -> None:\n    return None\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run(
            [declaring, kwonly]
        )

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_a_positional_only_parameter_is_compared(self, tmp_path: Path) -> None:
        """Positional-only args live in a third list again."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        posonly = write_source(
            tmp_path / "q.py",
            "from typing import Literal\n\n\n"
            'def read(corpus_format: Literal["lines"], /) -> None:\n    return None\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run(
            [declaring, posonly]
        )

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_an_unannotated_parameter_is_ignored(self, tmp_path: Path) -> None:
        """A bare parameter has no annotation to compare."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        bare = write_source(
            tmp_path / "q.py",
            "def read(corpus_format) -> None:\n    return None\n",
        )

        assert LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, bare]) == []

    def test_a_return_on_an_unrelated_function_is_ignored(self, tmp_path: Path) -> None:
        """Only a function whose name says it yields a format is compared."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        unrelated = write_source(
            tmp_path / "q.py",
            "from typing import Literal\n\n\n"
            'def pick_optimizer() -> Literal["adamw"]:\n    return "adamw"\n',
        )

        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, unrelated])
            == []
        )

    def test_a_subscript_target_is_ignored(self, tmp_path: Path) -> None:
        """``registry["corpus_format"]: X`` names no field of that name."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        subscript = write_source(
            tmp_path / "q.py",
            "from typing import Literal\n\nregistry = {}\n"
            'registry["corpus_format"]: Literal["lines"]\n',
        )

        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, subscript])
            == []
        )

    def test_a_subscript_whose_base_is_not_a_name_is_ignored(self, tmp_path: Path) -> None:
        """A computed annotation base is not the typing Literal."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        computed = write_source(
            tmp_path / "q.py",
            "def factory():\n    return dict\n\n\nclass P:\n"
            '    corpus_format: factory()["lines"]\n',
        )

        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring, computed]) == []
        )


class TestTheRuleKnowsWhenItIsInert:
    """A guard that silently checks nothing is worse than no guard."""

    def test_a_package_that_does_not_declare_the_tuple_is_still_checked(
        self, tmp_path: Path
    ) -> None:
        """THE DEFECT THIS RULE ITSELF HAD.

        It used to look for the declaring module among the files it was
        handed, and return no findings when it was not there. So for every
        package except the one owning the tuple it reported "0 violations"
        and read as checked -- forty-three packages out of forty-four, and
        every set shared through a library. The declaring module is now
        resolved from the monorepo root, so a stale Literal is caught in the
        package that actually holds it.
        """
        _declaring(tmp_path, '"lines", "documents"')
        stale = write_source(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    corpus_format: Literal["anything"]\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([stale])

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_a_declaring_module_that_cannot_be_found_is_reported(self, tmp_path: Path) -> None:
        """Nothing declares the tuple anywhere, so the rule can read no set.

        Reported rather than skipped: silence here is indistinguishable from a
        clean tree, which is the state this rule was in for every package that
        did not own the declaration.
        """
        stale = write_source(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    corpus_format: Literal["x"]\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([stale])

        assert [v.kind for v in violations] == ["corpus-format-declaration-unresolved"]

    def test_two_packages_declaring_the_tuple_is_reported(self, tmp_path: Path) -> None:
        """Which of them is authoritative is not the guard's to guess."""
        _declaring(tmp_path, '"lines", "documents"')
        write_source(
            tmp_path / "libs" / "second_pkg" / "src" / _DECLARING,
            'from typing import Literal\n\nCORPUS_FORMATS: tuple[Literal["lines"], ...]'
            ' = ("lines",)\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([])

        assert [v.kind for v in violations] == ["corpus-format-declaration-unresolved"]
        assert "expected exactly one" in violations[0].line

    def test_a_tuple_that_is_no_longer_a_tuple_is_reported(self, tmp_path: Path) -> None:
        """Rebinding the name to a call would make the rule read nothing."""
        declaring = write_source(
            tmp_path / DECLARING_PACKAGE / _DECLARING,
            "from typing import Literal\n\nCORPUS_FORMATS: "
            'tuple[Literal["lines"], ...] = tuple(["lines"])\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring])

        assert [v.kind for v in violations] == ["corpus-format-tuple-missing"]

    def test_a_tuple_of_non_strings_is_reported(self, tmp_path: Path) -> None:
        """A tuple whose members are not literals cannot be compared."""
        declaring = write_source(
            tmp_path / DECLARING_PACKAGE / _DECLARING,
            "from typing import Literal\n\nLINES = 'lines'\nCORPUS_FORMATS: "
            'tuple[Literal["lines"], ...] = (LINES,)\n',
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring])

        assert [v.kind for v in violations] == ["corpus-format-tuple-missing"]

    def test_a_declaring_module_binding_no_tuple_at_all_is_reported(self, tmp_path: Path) -> None:
        """The name could be deleted outright by a careless rename."""
        declaring = write_source(tmp_path / DECLARING_PACKAGE / _DECLARING, "OTHER: int = 1\n")

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring])

        assert [v.kind for v in violations] == ["corpus-format-tuple-missing"]

    def test_an_unannotated_assignment_does_not_count_as_the_tuple(self, tmp_path: Path) -> None:
        """The rule reads the annotated declaration, which is the real one."""
        declaring = write_source(
            tmp_path / DECLARING_PACKAGE / _DECLARING, 'CORPUS_FORMATS = ("lines", "documents")\n'
        )

        violations = LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).run([declaring])

        assert [v.kind for v in violations] == ["corpus-format-tuple-missing"]
