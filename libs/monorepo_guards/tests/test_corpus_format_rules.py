"""Tests for the corpus-format literal-drift rule.

The motivating case is ``_flags_a_literal_left_behind_by_a_widened_tuple``:
that is what the source looks like the moment after a third format is added to
``CORPUS_FORMATS`` and one of the dozen inline ``Literal`` annotations is not
updated with it. mypy accepts that state, because the annotations are
independent of each other and of the tuple.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.corpus_format_rules import CorpusFormatLiteralRule

_DECLARING = "core/contracts/dataset.py"


def _write(path: Path, text: str) -> Path:
    """Write a source file for the rule to scan.

    Args:
        path: Where to write it.
        text: The source.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _declaring(tmp_path: Path, members: str) -> Path:
    """Write a stand-in declaring module binding the tuple.

    Args:
        tmp_path: Directory to write into.
        members: The tuple's contents, as source.

    Returns:
        The path written.
    """
    return _write(
        tmp_path / _DECLARING,
        f'from typing import Literal\n\nCORPUS_FORMATS: tuple[Literal["lines", '
        f'"documents"], ...] = ({members})\n',
    )


class TestDrift:
    """A Literal that no longer names what the tuple declares."""

    def test_it_flags_a_literal_left_behind_by_a_widened_tuple(self, tmp_path: Path) -> None:
        """The exact state mypy accepts and this rule exists to refuse."""
        declaring = _declaring(tmp_path, '"lines", "documents", "fim"')
        stale = _write(
            tmp_path / "contracts" / "queue.py",
            "from typing import Literal\n\n\nclass P:\n"
            '    corpus_format: Literal["lines", "documents"]\n',
        )

        violations = CorpusFormatLiteralRule().run([declaring, stale])

        kinds = [v.kind for v in violations]
        assert kinds == ["corpus-format-literal-drift"]
        assert "documents, fim, lines" in violations[0].line

    def test_it_flags_a_stale_parameter_annotation(self, tmp_path: Path) -> None:
        """Signatures drift the same way field annotations do."""
        declaring = _declaring(tmp_path, '"lines", "documents", "fim"')
        stale = _write(
            tmp_path / "reader.py",
            "from typing import Literal\n\n\n"
            'def read(corpus_format: Literal["lines"]) -> None:\n    return None\n',
        )

        violations = CorpusFormatLiteralRule().run([declaring, stale])

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_it_flags_a_stale_return_annotation(self, tmp_path: Path) -> None:
        """A narrowing function's return is the third drifting shape."""
        declaring = _declaring(tmp_path, '"lines", "documents", "fim"')
        stale = _write(
            tmp_path / "narrow.py",
            "from typing import Literal\n\n\n"
            'def as_corpus_format(raw: str) -> Literal["lines", "documents"]:\n'
            '    return "lines"\n',
        )

        violations = CorpusFormatLiteralRule().run([declaring, stale])

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_the_message_names_both_sets(self, tmp_path: Path) -> None:
        """A drift report is only actionable if it says what disagrees."""
        declaring = _declaring(tmp_path, '"lines", "documents", "fim"')
        stale = _write(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    corpus_format: Literal["lines"]\n',
        )

        violations = CorpusFormatLiteralRule().run([declaring, stale])

        assert "names lines" in violations[0].line
        assert "documents, fim, lines" in violations[0].line


class TestAgreement:
    """The rule stays silent when nothing has drifted."""

    def test_a_matching_literal_passes(self, tmp_path: Path) -> None:
        """The state the repository is actually in."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        matching = _write(
            tmp_path / "contracts" / "queue.py",
            "from typing import Literal\n\n\nclass P:\n"
            '    corpus_format: Literal["lines", "documents"]\n',
        )

        assert CorpusFormatLiteralRule().run([declaring, matching]) == []

    def test_member_order_does_not_matter(self, tmp_path: Path) -> None:
        """The comparison is over sets, so reordering is not drift."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        reordered = _write(
            tmp_path / "q.py",
            "from typing import Literal\n\n\nclass P:\n"
            '    corpus_format: Literal["documents", "lines"]\n',
        )

        assert CorpusFormatLiteralRule().run([declaring, reordered]) == []

    def test_a_single_member_literal_subscript_is_read(self, tmp_path: Path) -> None:
        """``Literal["lines"]`` has no ast.Tuple slice, and must still parse.

        The declaring tuple needs its trailing comma for the same reason:
        ``("lines")`` is a string in parentheses, not a one-member tuple.
        """
        declaring = _declaring(tmp_path, '"lines",')
        single = _write(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    corpus_format: Literal["lines"]\n',
        )

        assert CorpusFormatLiteralRule().run([declaring, single]) == []

    def test_a_qualified_literal_is_read(self, tmp_path: Path) -> None:
        """``typing.Literal[...]`` is an Attribute base, not a Name."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        qualified = _write(
            tmp_path / "q.py",
            "import typing\n\n\nclass P:\n"
            '    corpus_format: typing.Literal["lines", "documents"]\n',
        )

        assert CorpusFormatLiteralRule().run([declaring, qualified]) == []

    def test_a_non_literal_annotation_is_ignored(self, tmp_path: Path) -> None:
        """The manifest carries the format as a plain ``str`` by convention."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        plain = _write(
            tmp_path / "models.py",
            "class M:\n    corpus_format: str\n",
        )

        assert CorpusFormatLiteralRule().run([declaring, plain]) == []

    def test_a_subscripted_non_literal_annotation_is_ignored(self, tmp_path: Path) -> None:
        """``Sequence[...]`` subscripts too, and names no format."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        sequence = _write(
            tmp_path / "q.py",
            "from collections.abc import Sequence\n\n\nclass P:\n"
            "    corpus_format: Sequence[str]\n",
        )

        assert CorpusFormatLiteralRule().run([declaring, sequence]) == []

    def test_a_non_string_literal_is_ignored(self, tmp_path: Path) -> None:
        """A Literal of ints names no corpus format."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        numeric = _write(
            tmp_path / "q.py",
            "from typing import Literal\n\n\nclass P:\n    corpus_format: Literal[1, 2]\n",
        )

        assert CorpusFormatLiteralRule().run([declaring, numeric]) == []

    def test_an_unrelated_field_is_ignored(self, tmp_path: Path) -> None:
        """Only annotations on the format's own name are compared."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        other = _write(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    optimizer: Literal["adamw", "sgd"]\n',
        )

        assert CorpusFormatLiteralRule().run([declaring, other]) == []

    def test_an_attribute_target_is_compared(self, tmp_path: Path) -> None:
        """``self.corpus_format: Literal[...]`` is an ast.Attribute target."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        attribute = _write(
            tmp_path / "q.py",
            "from typing import Literal\n\n\nclass P:\n    def __init__(self) -> None:\n"
            '        self.corpus_format: Literal["lines"] = "lines"\n',
        )

        violations = CorpusFormatLiteralRule().run([declaring, attribute])

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_an_async_function_parameter_is_compared(self, tmp_path: Path) -> None:
        """Async definitions are a separate AST node and must be walked too."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        asynchronous = _write(
            tmp_path / "q.py",
            "from typing import Literal\n\n\n"
            'async def read(corpus_format: Literal["lines"]) -> None:\n    return None\n',
        )

        violations = CorpusFormatLiteralRule().run([declaring, asynchronous])

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_a_keyword_only_parameter_is_compared(self, tmp_path: Path) -> None:
        """Keyword-only args live in their own list on ast.arguments."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        kwonly = _write(
            tmp_path / "q.py",
            "from typing import Literal\n\n\n"
            'def read(*, corpus_format: Literal["lines"]) -> None:\n    return None\n',
        )

        violations = CorpusFormatLiteralRule().run([declaring, kwonly])

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_a_positional_only_parameter_is_compared(self, tmp_path: Path) -> None:
        """Positional-only args live in a third list again."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        posonly = _write(
            tmp_path / "q.py",
            "from typing import Literal\n\n\n"
            'def read(corpus_format: Literal["lines"], /) -> None:\n    return None\n',
        )

        violations = CorpusFormatLiteralRule().run([declaring, posonly])

        assert [v.kind for v in violations] == ["corpus-format-literal-drift"]

    def test_an_unannotated_parameter_is_ignored(self, tmp_path: Path) -> None:
        """A bare parameter has no annotation to compare."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        bare = _write(
            tmp_path / "q.py",
            "def read(corpus_format) -> None:\n    return None\n",
        )

        assert CorpusFormatLiteralRule().run([declaring, bare]) == []

    def test_a_return_on_an_unrelated_function_is_ignored(self, tmp_path: Path) -> None:
        """Only a function whose name says it yields a format is compared."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        unrelated = _write(
            tmp_path / "q.py",
            "from typing import Literal\n\n\n"
            'def pick_optimizer() -> Literal["adamw"]:\n    return "adamw"\n',
        )

        assert CorpusFormatLiteralRule().run([declaring, unrelated]) == []

    def test_a_subscript_target_is_ignored(self, tmp_path: Path) -> None:
        """``registry["corpus_format"]: X`` names no field of that name."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        subscript = _write(
            tmp_path / "q.py",
            "from typing import Literal\n\nregistry = {}\n"
            'registry["corpus_format"]: Literal["lines"]\n',
        )

        assert CorpusFormatLiteralRule().run([declaring, subscript]) == []

    def test_a_subscript_whose_base_is_not_a_name_is_ignored(self, tmp_path: Path) -> None:
        """A computed annotation base is not the typing Literal."""
        declaring = _declaring(tmp_path, '"lines", "documents"')
        computed = _write(
            tmp_path / "q.py",
            "def factory():\n    return dict\n\n\nclass P:\n"
            '    corpus_format: factory()["lines"]\n',
        )

        assert CorpusFormatLiteralRule().run([declaring, computed]) == []


class TestTheRuleKnowsWhenItIsInert:
    """A guard that silently checks nothing is worse than no guard."""

    def test_packages_without_the_declaring_module_are_skipped(self, tmp_path: Path) -> None:
        """Forty of the forty-one packages never declare the tuple."""
        elsewhere = _write(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    corpus_format: Literal["anything"]\n',
        )

        assert CorpusFormatLiteralRule().run([elsewhere]) == []

    def test_a_tuple_that_is_no_longer_a_tuple_is_reported(self, tmp_path: Path) -> None:
        """Rebinding the name to a call would make the rule read nothing."""
        declaring = _write(
            tmp_path / _DECLARING,
            "from typing import Literal\n\nCORPUS_FORMATS: "
            'tuple[Literal["lines"], ...] = tuple(["lines"])\n',
        )

        violations = CorpusFormatLiteralRule().run([declaring])

        assert [v.kind for v in violations] == ["corpus-format-tuple-missing"]

    def test_a_tuple_of_non_strings_is_reported(self, tmp_path: Path) -> None:
        """A tuple whose members are not literals cannot be compared."""
        declaring = _write(
            tmp_path / _DECLARING,
            "from typing import Literal\n\nLINES = 'lines'\nCORPUS_FORMATS: "
            'tuple[Literal["lines"], ...] = (LINES,)\n',
        )

        violations = CorpusFormatLiteralRule().run([declaring])

        assert [v.kind for v in violations] == ["corpus-format-tuple-missing"]

    def test_a_declaring_module_binding_no_tuple_at_all_is_reported(self, tmp_path: Path) -> None:
        """The name could be deleted outright by a careless rename."""
        declaring = _write(tmp_path / _DECLARING, "OTHER: int = 1\n")

        violations = CorpusFormatLiteralRule().run([declaring])

        assert [v.kind for v in violations] == ["corpus-format-tuple-missing"]

    def test_an_unannotated_assignment_does_not_count_as_the_tuple(self, tmp_path: Path) -> None:
        """The rule reads the annotated declaration, which is the real one."""
        declaring = _write(tmp_path / _DECLARING, 'CORPUS_FORMATS = ("lines", "documents")\n')

        violations = CorpusFormatLiteralRule().run([declaring])

        assert [v.kind for v in violations] == ["corpus-format-tuple-missing"]
