"""The retired machine name scan, over any corpus: the repository's tracked
files in test_retired_machine_names.py, or a handful of texts held in memory,
so each kept class can be seen to admit a line and refuse its near miss (MCPs
board task 3f37b2a7).

This is MCPs ``mcp-shared/tests/architecture/_retired-name-scan.ts`` (board
task 88f5676c) in this repository's language. Its classes are predicates, as
that room's lead set them on 88f5676c's thread (045b032c, 3b9908f8, 635e9594,
42650bd5, e55b336e), never a list of files, and they keep MCPs' numbers:

  0 and 1 (applied migration SQL, and a migrate test seeding its migration's
     literal) have no subject here. This repository tracks one ``.sql`` file,
     a schema with no migration chain, and no migrate test, so a migration
     naming the name would be code written after the machine was gone, and
     no class admits it.
  2. A capture. (a) A string literal naming one, the name inside the
     capture's base name there; (b) a string literal found byte for byte in a
     capture its module names; (c) a line inside a capture; (d) a string
     literal in a ``_*.py`` test module whose header says, with the day, that
     its text was copied verbatim. A capture is a fixture file of data a
     program wrote (.xml, .json, .jsonl, .txt or .csv) named
     <machine>-<kind>-<date>, or one the module loading it declares, in the
     comment on the lines above the one naming it or in its header, with the
     day and the word for how it was captured. (e) A string literal in which
     every occurrence of the name is one whole path segment directly under
     an archive mount, ``/mnt/archive-<letters>/<name>/``: the directory a
     live host keeps a wiped machine's bytes in is named for the machine
     they were copied from, which is provenance, and the line naming it says
     where those bytes are (lead's answer 4f481261 on this task's thread).
  3. The scan's needle: the one line declaring :data:`RETIRED_NAME`, in the
     module that defines :func:`scan_retired_name`, known by the path the
     caller passes (the repository run passes :data:`NEEDLE_MODULE`'s). A
     check that names its subject must hold it once; everything else here
     takes the name as an argument.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Final, NamedTuple, Protocol

from tests._retired_name_lexing import Lexed, Lexer, Span, lex_all, line_starts_of, within


class KeptClass(StrEnum):
    """The kept classes, as the report names them."""

    CAPTURE_PATH = "2a a path to a capture"
    CAPTURE_BYTES = "2b a capture's own bytes"
    INSIDE_CAPTURE = "2c inside a capture"
    COPIED_RECORD = "2d a copied record's text"
    ARCHIVE_PATH = "2e a path to an archive named for its source machine"
    NEEDLE = "3 the scan's own needle"


class Corpus(Protocol):
    """What the scan reads."""

    def tracked(self) -> tuple[str, ...]:
        """Every tracked path."""
        ...

    def read(self, path: str) -> str:
        """One tracked file's text."""
        ...

    def naming(self, needles: tuple[str, ...], ignore_case: bool) -> tuple[str, ...]:
        """The tracked text files holding any of the needles."""
        ...


class Hit(NamedTuple):
    """One line naming the retired machine.

    Attributes:
        path: The file.
        line: The 1-based line.
        code: Offsets of its occurrences outside comments.
    """

    path: str
    line: int
    code: tuple[int, ...]


class Verdict(NamedTuple):
    """A non-comment line and the class that kept it, or None."""

    hit: Hit
    kept: KeptClass | None


class Scan(NamedTuple):
    """What the scan found.

    Attributes:
        name: The name it looked for.
        tracked: How many tracked files the corpus holds.
        files: The files naming it.
        lexers: How many of them each lexer read.
        hits: Every line naming it.
        verdicts: Each line naming it outside comments, with its class.
        captures: Each capture, and what declared it.
        candidates: How many fixture files could have been captures.
    """

    name: str
    tracked: int
    files: tuple[str, ...]
    lexers: dict[Lexer, int]
    hits: tuple[Hit, ...]
    verdicts: tuple[Verdict, ...]
    captures: dict[str, str]
    candidates: int


#: The name this repository retired, and the one line of code that holds it:
#: the scan's needle, which class 3 admits by finding this declaration. Every
#: other piece of the scan reads the name from its argument, so the cases in
#: memory run it under a name no repository holds.
RETIRED_NAME: Final = "austinpc"

#: The declaration class 3 admits, as it is written above, for a given name.
NEEDLE_DECLARATION: Final = 'RETIRED_NAME: Final = "{name}"'

#: This module's own absolute path: the one file class 3 admits the needle
#: in, once the caller makes it repository-relative.
NEEDLE_MODULE: Final = Path(__file__).resolve()

DATED_CAPTURE: Final = re.compile(
    r"(^|/)tests/(.+/)?fixtures/[a-z0-9]+-[a-z0-9-]+-\d{4}-\d{2}-\d{2}\.[a-z0-9]+$"
)
FIXTURE_FILE: Final = re.compile(r"(^|/)tests/(.+/)?fixtures/")
DATA_EXTENSION: Final = re.compile(r"\.(xml|json|jsonl|txt|csv)$")
TEST_MODULE: Final = re.compile(r"(^|/)tests/.+\.py$")
ISO_DAY: Final = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
CAPTURE_WORD: Final = re.compile(
    r"\b(captur\w*|copied|verbatim|written|recorded|read from)\b", re.IGNORECASE
)
COPIED_WORD: Final = re.compile(r"\b(verbatim|copied)\b", re.IGNORECASE)
#: An archive mount, as the text before a class 2e occurrence must end.
ARCHIVE_MOUNT: Final = re.compile(r"/mnt/archive-[a-z]+/$")
QUOTE_OPEN: Final = re.compile(r"^[rbfutRBFUT]*(\"\"\"|'''|\"|')")
QUOTE_CLOSE: Final = re.compile(r"(\"\"\"|'''|\"|')$")


def base_name(path: str) -> str:
    """The base name of a path.

    Args:
        path: The path.

    Returns:
        Everything after its last slash.
    """
    return path[path.rfind("/") + 1 :]


def _line_of(starts: tuple[int, ...], offset: int) -> int:
    """The 1-based line an offset is on."""
    return sum(1 for start in starts if start <= offset)


def _line_text(text: str, starts: tuple[int, ...], line: int) -> str:
    """One 1-based line's text, without its newline."""
    start = starts[line - 1]
    end = starts[line] - 1 if line < len(starts) else len(text)
    return text[start:end]


def header_of(text: str, lexed: Lexed) -> str:
    """A module's header: every comment before its first code.

    Args:
        text: The module's text.
        lexed: Its lexing.

    Returns:
        The header's comments, joined by newlines; empty when code comes first.
    """
    parts: list[str] = []
    at = 0
    while True:
        while at < len(text) and text[at].isspace():
            at += 1
        span = next((s for s in lexed.comments if s.start == at), None)
        if span is None:
            return "\n".join(parts)
        parts.append(text[span.start : span.end])
        at = span.end


def comment_above(text: str, lexed: Lexed, line: int) -> str:
    """The comment-only lines directly above a 1-based line, joined.

    Args:
        text: The module's text.
        lexed: Its lexing.
        line: The line.

    Returns:
        Those lines, nearest first.
    """
    starts = line_starts_of(text)
    above: list[str] = []
    for row in range(line - 1, 0, -1):
        content = _line_text(text, starts, row)
        start = starts[row - 1]
        comment_only = content.strip() != "" and all(
            ch.isspace() or within(lexed.comments, start + i) for i, ch in enumerate(content)
        )
        if not comment_only:
            break
        above.append(content)
    return "\n".join(above)


def literal_at(lexed: Lexed, offset: int) -> Span | None:
    """The string literal holding an offset.

    Args:
        lexed: The file's lexing.
        offset: The offset.

    Returns:
        The literal, or None when the offset is in none.
    """
    return next((s for s in lexed.strings if s.start <= offset < s.end), None)


def unquoted(literal: str) -> str:
    """A literal's value, without its prefix and quotes.

    Args:
        literal: The literal as written.

    Returns:
        Its text between the quotes, escapes as written.
    """
    return QUOTE_CLOSE.sub("", QUOTE_OPEN.sub("", literal, count=1), count=1)


def _inside_base_name(text: str, span: Span, offset: int, base: str) -> bool:
    """Whether an occurrence lies inside one of a base name's spans in its literal."""
    at = text.find(base, span.start)
    while at != -1 and at + len(base) <= span.end:
        if at <= offset < at + len(base):
            return True
        at = text.find(base, at + 1)
    return False


def _archive_segment(text: str, span: Span, offset: int, length: int) -> bool:
    """Whether an occurrence is a whole path segment directly under an archive mount.

    Args:
        text: The file's text.
        span: The literal holding the occurrence.
        offset: The occurrence's offset.
        length: The name's length.

    Returns:
        True when the literal reads ``/mnt/archive-<letters>/`` just before it
        and ``/`` just after it.
    """
    after = offset + length
    return (
        ARCHIVE_MOUNT.search(text[span.start : offset]) is not None
        and after < span.end
        and text[after] == "/"
    )


def _hits(texts: dict[str, str], lexed: dict[str, Lexed], name: str) -> tuple[Hit, ...]:
    """Every line naming the name, in any case, with its code occurrences."""
    retired = re.compile(re.escape(name), re.IGNORECASE)
    hits: list[Hit] = []
    for path, text in texts.items():
        starts = line_starts_of(text)
        by_line: dict[int, list[int]] = {}
        for match in retired.finditer(text):
            code = by_line.setdefault(_line_of(starts, match.start()), [])
            if not within(lexed[path].comments, match.start()):
                code.append(match.start())
        hits.extend(Hit(path, line, tuple(code)) for line, code in by_line.items())
    return tuple(hits)


def _declarer(modules: dict[str, str], lexed: dict[str, Lexed]) -> Callable[[str], str | None]:
    """What declares a fixture file a capture, read from the modules naming it."""

    def declared_by(path: str) -> str | None:
        if DATED_CAPTURE.search(path):
            return "its name"
        for module, text in modules.items():
            spans = lexed[module]
            starts = line_starts_of(text)
            for span in spans.strings:
                if base_name(path) not in text[span.start : span.end]:
                    continue
                line = _line_of(starts, span.start)
                above = comment_above(text, spans, line)
                if ISO_DAY.search(above) and CAPTURE_WORD.search(above):
                    return f"the comment above {module}:{line}"
                header = header_of(text, spans)
                if ISO_DAY.search(header) and CAPTURE_WORD.search(header):
                    return f"the header of {module}:1, naming it at line {line}"
        return None

    return declared_by


def scan_retired_name(corpus: Corpus, name: str, needle_home: str) -> Scan:
    """Scan a corpus for a retired name.

    Args:
        corpus: What to read.
        name: The name, lower case, matched in any case.
        needle_home: The corpus path of the module declaring the needle, the
            one file class 3 admits it in.

    Returns:
        Every line naming it, and each non-comment line's kept class or None.
    """
    files = corpus.naming((name,), True)
    texts = {path: corpus.read(path) for path in files}
    lexed = lex_all(texts)
    hits = _hits(texts, lexed, name)

    candidates = tuple(
        path
        for path in corpus.tracked()
        if FIXTURE_FILE.search(path)
        and DATA_EXTENSION.search(path)
        and (path in texts or name in base_name(path).lower())
    )
    naming_modules = (
        corpus.naming(tuple(base_name(p) for p in candidates), False) if candidates else ()
    )
    modules = {
        path: corpus.read(path)
        for path in naming_modules
        if TEST_MODULE.search(path) and not FIXTURE_FILE.search(path)
    }
    declared_by = _declarer(modules, lex_all(modules))
    captures = {path: by for path in candidates if (by := declared_by(path)) is not None}

    def class_of(hit: Hit) -> KeptClass | None:
        spans = lexed[hit.path]
        text = texts[hit.path]
        literals = tuple(literal_at(spans, offset) for offset in hit.code)
        if hit.path in captures:
            return KeptClass.INSIDE_CAPTURE
        named = tuple(capture for capture in captures if base_name(capture) in text)
        if all(
            span is not None
            and any(_inside_base_name(text, span, offset, base_name(c)) for c in named)
            for offset, span in zip(hit.code, literals, strict=True)
        ):
            return KeptClass.CAPTURE_PATH
        if all(
            span is not None
            and any(unquoted(text[span.start : span.end]) in corpus.read(c) for c in named)
            for span in literals
        ):
            return KeptClass.CAPTURE_BYTES
        if all(
            span is not None and _archive_segment(text, span, offset, len(name))
            for offset, span in zip(hit.code, literals, strict=True)
        ):
            return KeptClass.ARCHIVE_PATH
        header = header_of(text, spans)
        copied = base_name(hit.path).startswith("_") and TEST_MODULE.search(hit.path) is not None
        if (
            copied
            and all(span is not None for span in literals)
            and ISO_DAY.search(header)
            and COPIED_WORD.search(header)
        ):
            return KeptClass.COPIED_RECORD
        line = _line_text(text, line_starts_of(text), hit.line)
        declares = line == NEEDLE_DECLARATION.format(name=name)
        if hit.path == needle_home and declares and "def scan_retired_name(" in text:
            return KeptClass.NEEDLE
        return None

    return Scan(
        name=name,
        tracked=len(corpus.tracked()),
        files=files,
        lexers={lexer: sum(1 for p in files if lexed[p].lexer is lexer) for lexer in Lexer},
        hits=hits,
        verdicts=tuple(Verdict(hit, class_of(hit)) for hit in hits if hit.code),
        captures=captures,
        candidates=len(candidates),
    )


def unkept_lines(scan: Scan) -> list[str]:
    """The lines no class kept.

    Args:
        scan: The scan.

    Returns:
        Each as path:line.
    """
    return [f"{v.hit.path}:{v.hit.line}" for v in scan.verdicts if v.kept is None]


def scan_report(scan: Scan) -> str:
    """What the scan read and what each class admitted, file by file.

    Args:
        scan: The scan.

    Returns:
        The report, ending with "N of M non-comment line(s) kept".
    """
    tally: dict[KeptClass, dict[str, int]] = {}
    for verdict in scan.verdicts:
        if verdict.kept is not None:
            per_file = tally.setdefault(verdict.kept, {})
            per_file[verdict.hit.path] = per_file.get(verdict.hit.path, 0) + 1
    classes = [
        f"class {kept.value}: {sum(tally[kept].values())} line(s) in {len(tally[kept])} "
        "file(s): " + ", ".join(f"{path} ({n})" for path, n in tally[kept].items())
        for kept in sorted(tally)
    ]
    comment_lines = len(scan.hits) - len(scan.verdicts)
    declared = "; ".join(f"{path} by {by}" for path, by in scan.captures.items()) or "none"
    kept = len(scan.verdicts) - len(unkept_lines(scan))
    return "\n".join(
        [
            f"{scan.name}: {len(scan.hits)} line(s) in {len(scan.files)} of {scan.tracked} "
            f"tracked file(s); {comment_lines} in comments, docstrings or prose, "
            f"{len(scan.verdicts)} outside them",
            "files by lexer (data is every language read without one, TypeScript and "
            "Rust among them): "
            + ", ".join(f"{lexer.value} {n}" for lexer, n in scan.lexers.items()),
            *classes,
            f"captures: {len(scan.captures)} of {scan.candidates} candidate fixture file(s): "
            + declared,
            f"{kept} of {len(scan.verdicts)} non-comment line(s) kept",
        ]
    )
