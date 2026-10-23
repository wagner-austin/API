"""The retired machine name is gone from code and config, and every line still
holding it is a comment or a record of a past day (MCPs board task 3f37b2a7).

The Windows install the fleet called by that name was wiped on 2026-10-22,
and the machine came back as demure (``registry/machines/demure.json``). The
scan (``_retired_name_scan.py``) reads every tracked line naming it, in any
case, with its own language's lexer (``_retired_name_lexing.py``): a line
whose every occurrence sits in a comment, a docstring or prose is history,
which needs no change, and each other line must fall in a kept class. The
run over the repository records what each class admitted, file by file, and
"N of M non-comment line(s) kept", which the run's summary prints
(``tests/_host.py``). The cases after it hand the same scan a few texts held
in memory, under a made-up name, so each class is seen to admit one line and
refuse its near miss: a guard seen only on clean code has never been seen to
work. MCPs runs the same scan over its own tree (``mcp-shared/tests/
architecture/retired-machine-names.test.ts``, board task 88f5676c).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Final

from tests._host import REPORT_PROPERTY
from tests._retired_name_lexing import Lexer, lexer_for
from tests._retired_name_scan import (
    NEEDLE_MODULE,
    RETIRED_NAME,
    KeptClass,
    scan_report,
    scan_retired_name,
    unkept_lines,
)

#: The repository root: tools/fleet/tests is three levels below it.
ROOT: Final = Path(__file__).resolve().parents[3]

#: The name the cases in memory retire: one no repository holds.
NAME: Final = "oldhub"

#: Where the cases in memory declare their needle.
MEMORY_NEEDLE_HOME: Final = "tools/fleet/tests/_scan.py"


def _git(*args: str) -> tuple[str, ...]:
    """Run git at the repository root, reading NUL-separated entries.

    Args:
        args: Its arguments, with ``-z``.

    Returns:
        The entries; none when grep matched nothing.

    Raises:
        RuntimeError: When git fails.
    """
    done = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False, timeout=120
    )
    nothing_found = done.returncode == 1 and done.stdout == "" and done.stderr == ""
    if done.returncode != 0 and not nothing_found:
        raise RuntimeError(f"git {' '.join(args)} exited {done.returncode}: {done.stderr}")
    return tuple(entry for entry in done.stdout.split("\0") if entry)


class GitCorpus:
    """The repository's tracked files, as git ls-files and git grep give them."""

    def tracked(self) -> tuple[str, ...]:
        """Every tracked path.

        Returns:
            The paths, repository-relative.
        """
        return _git("ls-files", "-z")

    def read(self, path: str) -> str:
        """One tracked file's text.

        Args:
            path: Its repository-relative path.

        Returns:
            Its text.
        """
        return (ROOT / path).read_text(encoding="utf-8")

    def naming(self, needles: tuple[str, ...], ignore_case: bool) -> tuple[str, ...]:
        """The tracked text files holding any of the needles.

        Args:
            needles: Fixed strings.
            ignore_case: Whether case is ignored.

        Returns:
            The paths.
        """
        flags = ("-i",) if ignore_case else ()
        patterns = tuple(arg for needle in needles for arg in ("-e", needle))
        return _git("grep", "-I", "-l", "-z", "-F", *flags, *patterns)


class MemoryCorpus:
    """A corpus of texts held in memory."""

    def __init__(self, files: dict[str, str]) -> None:
        """Hold the texts.

        Args:
            files: Each path's text.
        """
        self._files = files

    def tracked(self) -> tuple[str, ...]:
        """Every path held.

        Returns:
            The paths.
        """
        return tuple(self._files)

    def read(self, path: str) -> str:
        """One held text.

        Args:
            path: Its path.

        Returns:
            Its text.
        """
        return self._files[path]

    def naming(self, needles: tuple[str, ...], ignore_case: bool) -> tuple[str, ...]:
        """The held texts holding any of the needles.

        Args:
            needles: Fixed strings.
            ignore_case: Whether case is ignored.

        Returns:
            The paths.
        """

        def holds(text: str, needle: str) -> bool:
            return needle.lower() in text.lower() if ignore_case else needle in text

        return tuple(
            path
            for path, text in self._files.items()
            if any(holds(text, needle) for needle in needles)
        )


def _verdicts(files: dict[str, str]) -> dict[str, KeptClass | None]:
    """Scan texts held in memory for :data:`NAME`, keyed by path:line.

    Args:
        files: Each path's text.

    Returns:
        Each non-comment line's verdict; comment lines carry none.
    """
    scan = scan_retired_name(MemoryCorpus(files), NAME, MEMORY_NEEDLE_HOME)
    return {f"{v.hit.path}:{v.hit.line}": v.kept for v in scan.verdicts}


def test_the_repository_names_it_only_in_comments_and_kept_classes(
    record_property: Callable[[str, str], None],
) -> None:
    needle_home = NEEDLE_MODULE.relative_to(ROOT).as_posix()
    scan = scan_retired_name(GitCorpus(), RETIRED_NAME, needle_home)
    record_property(REPORT_PROPERTY, scan_report(scan))
    needles = [v.hit.path for v in scan.verdicts if v.kept is KeptClass.NEEDLE]
    assert needles == [needle_home]
    assert unkept_lines(scan) == []
    assert len(scan.hits) >= len(scan.files) >= 1
    assert scan.tracked > len(scan.files)


def test_each_lexer_reads_its_languages_comments() -> None:
    seen = _verdicts(
        {
            "tools/x.py": '"""Docs naming oldhub."""\n# oldhub\nHOST = "oldhub"\n',
            "tools/y.py": 'def f() -> str:\n    """oldhub."""\n    return f"{1}oldhub"\n',
            "tools/x.json": '{\n  "_comment": ["oldhub here"],\n  "notes": "was oldhub",\n'
            '  "host": "oldhub",\n  "oldhub": 1\n}\n',
            "tools/x.sh": "# oldhub\necho 'a # oldhub'\n",
            "tools/x.sql": "-- oldhub\nSELECT 'oldhub';\n/* oldhub */\n",
            "tools/x.ps1": "<#\n oldhub\n#>\nWrite-Host 'oldhub'\n",
            "tools/y.ps1": '# oldhub\n$x = @"\n# oldhub\n"@\n',
            "tools/x.toml": '# oldhub\nkey = "a # oldhub"\n',
            "tools/x.ini": '; oldhub\nhost = "oldhub"\n',
            "tools/.gitattributes": "# oldhub\n*.bin oldhub\n",
            "tools/x.md": "oldhub\n",
            "tools/wiki/log.txt": "oldhub\n",
            "tools/x.txt": "oldhub\n",
            "tools/x.ts": "// oldhub\n",
        }
    )
    assert seen == {
        "tools/x.py:3": None,
        "tools/y.py:3": None,
        "tools/x.json:4": None,
        "tools/x.json:5": None,
        "tools/x.sh:2": None,
        "tools/x.sql:2": None,
        "tools/x.ps1:4": None,
        "tools/y.ps1:3": None,
        "tools/x.toml:2": None,
        "tools/x.ini:2": None,
        "tools/.gitattributes:2": None,
        "tools/x.txt:1": None,
        "tools/x.ts:1": None,
    }


def test_a_lexer_is_chosen_by_the_path() -> None:
    chosen = {
        path: lexer_for(path)
        for path in (
            "a/b.py",
            "a/registry/m.json",
            "a/s.psm1",
            "a/q.sql",
            "a/c.ini",
            "a/Makefile",
            "a/.env.example",
            "a/.envrc",
            "wiki/pages/p.txt",
            "a/README.md",
            "a/k.rs",
            "a/.hidden",
        )
    }
    assert chosen == {
        "a/b.py": Lexer.PYTHON,
        "a/registry/m.json": Lexer.JSON,
        "a/s.psm1": Lexer.POWERSHELL,
        "a/q.sql": Lexer.SQL,
        "a/c.ini": Lexer.INI,
        "a/Makefile": Lexer.HASH,
        "a/.env.example": Lexer.HASH,
        "a/.envrc": Lexer.HASH,
        "wiki/pages/p.txt": Lexer.PROSE,
        "a/README.md": Lexer.PROSE,
        "a/k.rs": Lexer.DATA,
        "a/.hidden": Lexer.DATA,
    }


def test_class_2_admits_a_captures_path_bytes_and_lines_and_refuses_the_near_misses() -> None:
    seen = _verdicts(
        {
            "tools/fleet/tests/fixtures/oldhub-boot-2026-09-26.txt": '{"hostname":"OLDHUB"}\n',
            "tools/fleet/tests/_boot_fixtures.py": "\n".join(
                [
                    '"""The hub\'s boot probe."""',
                    'STDOUT = read_capture("oldhub-boot-2026-09-26.txt")',
                    'HOSTNAME = "OLDHUB"',
                    'OTHER = "OldHub-x"',
                    'LABEL = "oldhub as oldhub-boot-2026-09-26.txt saw it"',
                    "",
                ]
            ),
            "tools/fleet/tests/fixtures/oldhub-boot-2026-09-27.py": 'X = "oldhub"\n',
            "tools/fleet/tests/fixtures/stream.jsonl": '{"x":"oldhub"}\n',
            "tools/fleet/tests/fixtures/calls.json": '{"result": "oldhub"}\n',
            "tools/fleet/tests/_fake.py": "\n".join(
                [
                    '"""Running a fake claude."""',
                    "",
                    "import subprocess",
                    "",
                    "#: Captured from claude on 2026-10-06.",
                    'STREAM = "stream.jsonl"',
                    "",
                    "#: The calls of the 23:40Z turn, read from its transcript.",
                    'CALLS = "calls.json"',
                    "",
                ]
            ),
        }
    )
    assert seen == {
        "tools/fleet/tests/_boot_fixtures.py:2": KeptClass.CAPTURE_PATH,
        "tools/fleet/tests/_boot_fixtures.py:3": KeptClass.CAPTURE_BYTES,
        "tools/fleet/tests/_boot_fixtures.py:4": None,
        "tools/fleet/tests/_boot_fixtures.py:5": None,
        "tools/fleet/tests/fixtures/oldhub-boot-2026-09-26.txt:1": KeptClass.INSIDE_CAPTURE,
        "tools/fleet/tests/fixtures/oldhub-boot-2026-09-27.py:1": None,
        "tools/fleet/tests/fixtures/stream.jsonl:1": KeptClass.INSIDE_CAPTURE,
        "tools/fleet/tests/fixtures/calls.json:1": None,
    }


def test_class_2_reads_a_capture_declared_by_its_loaders_header() -> None:
    scan = scan_retired_name(
        MemoryCorpus(
            {
                "tools/fleet/tests/fixtures/junit.xml": "<testsuite hostname='oldhub'/>\n",
                "tools/fleet/tests/_junit.py": '"""Reports written on 2026-10-07, as written."""\n'
                '\nREPORT = "junit.xml"\n',
            }
        ),
        NAME,
        MEMORY_NEEDLE_HOME,
    )
    assert scan.captures == {
        "tools/fleet/tests/fixtures/junit.xml": "the header of tools/fleet/tests/_junit.py:1, "
        "naming it at line 3"
    }
    assert [v.kept for v in scan.verdicts] == [KeptClass.INSIDE_CAPTURE]


def test_class_2d_admits_a_literal_of_a_module_copied_verbatim_and_refuses_the_rest() -> None:
    seen = _verdicts(
        {
            "tools/fleet/tests/x/_real_2026_09_30.py": "\n".join(
                [
                    '"""The real threads, copied verbatim from the board on 2026-09-30."""',
                    'BODY = "measured on oldhub"',
                    "oldhub_helper = 1",
                    "",
                ]
            ),
            "tools/fleet/tests/x/_written.py": '"""Written by hand on 2026-09-30."""\n'
            'BODY = "on oldhub"\n',
            "tools/fleet/tests/x/real.py": '"""Copied verbatim on 2026-09-30."""\n'
            'BODY = "oldhub"\n',
        }
    )
    assert seen == {
        "tools/fleet/tests/x/_real_2026_09_30.py:2": KeptClass.COPIED_RECORD,
        "tools/fleet/tests/x/_real_2026_09_30.py:3": None,
        "tools/fleet/tests/x/_written.py:2": None,
        "tools/fleet/tests/x/real.py:2": None,
    }


def test_class_2e_admits_a_segment_under_an_archive_mount_and_refuses_the_near_misses() -> None:
    seen = _verdicts(
        {
            "tools/a.py": "\n".join(
                [
                    'A = "/mnt/archive-a/oldhub/x/y"',
                    'B = "/mnt/archive-a/x/oldhub/"',
                    'C = "/mnt/oldhub/"',
                    'D = "/home/oldhub/"',
                    'E = "C:/Users/oldhub/"',
                    'F = "/mnt/archive-a/oldhub/x " + "oldhub"',
                    'G = "/mnt/archive-a/oldhubx/"',
                    "oldhub = '/mnt/archive-a/oldhub/x'",
                    "",
                ]
            ),
        }
    )
    assert seen == {
        "tools/a.py:1": KeptClass.ARCHIVE_PATH,
        "tools/a.py:2": None,
        "tools/a.py:3": None,
        "tools/a.py:4": None,
        "tools/a.py:5": None,
        "tools/a.py:6": None,
        "tools/a.py:7": None,
        "tools/a.py:8": None,
    }


def test_class_3_admits_the_needle_in_the_scans_own_module_and_nowhere_else() -> None:
    declaration = 'RETIRED_NAME: Final = "oldhub"'
    scan_module = f"{declaration}\ndef scan_retired_name() -> None: ...\n"
    seen = _verdicts(
        {
            MEMORY_NEEDLE_HOME: f'{scan_module}AGAIN = "oldhub"\n',
            "tools/_copied_scan.py": scan_module,
            "tools/elsewhere.py": f"{declaration}\n",
        }
    )
    assert seen == {
        f"{MEMORY_NEEDLE_HOME}:1": KeptClass.NEEDLE,
        f"{MEMORY_NEEDLE_HOME}:3": None,
        "tools/_copied_scan.py:1": None,
        "tools/elsewhere.py:1": None,
    }


def test_the_report_prints_each_class_and_the_count() -> None:
    scan = scan_retired_name(
        MemoryCorpus(
            {
                "tools/fleet/tests/fixtures/oldhub-boot-2026-09-26.txt": "oldhub\n",
                "tools/a.py": "# oldhub\nX = 'oldhub'\n",
            }
        ),
        NAME,
        MEMORY_NEEDLE_HOME,
    )
    assert scan_report(scan).splitlines() == [
        "oldhub: 3 line(s) in 2 of 2 tracked file(s); 1 in comments, docstrings or prose, "
        "2 outside them",
        "files by lexer (data is every language read without one, TypeScript and Rust among "
        "them): python 1, json 0, hash 0, ini 0, powershell 0, sql 0, prose 0, data 1",
        "class 2c inside a capture: 1 line(s) in 1 file(s): "
        "tools/fleet/tests/fixtures/oldhub-boot-2026-09-26.txt (1)",
        "captures: 1 of 1 candidate fixture file(s): "
        "tools/fleet/tests/fixtures/oldhub-boot-2026-09-26.txt by its name",
        "1 of 2 non-comment line(s) kept",
    ]
    assert unkept_lines(scan) == ["tools/a.py:2"]
