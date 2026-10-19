"""Every queue close is made by fleet.cli.job_close (MCPs board task af8b5662).

The lift of af8b5662 moved the node lanes', the agent's and the hub jobs'
closes into :mod:`fleet.cli.job_close`, and check review 678e4470 found two
it missed: the node collect's refusal of a claim that launched nothing and
its settle's close of an ended run, each building ``queue.report_close`` by
hand. A file check of the lifted module could not see them. This suite reads
every module of this package's source from its syntax tree and fails on any
reference to ``report_close`` (a call, an attribute read or an import) in a
module but :data:`CLOSER`, so a close built anywhere else fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

#: This package's source.
SOURCE: Final[Path] = Path(__file__).resolve().parents[1] / "src" / "fleet"

#: The queue's one close, defined in fleet.core.queue.
CLOSE: Final[str] = "report_close"

#: The one module that may make it.
CLOSER: Final[Path] = SOURCE / "cli" / "job_close.py"

#: How many references the closer holds, refuse's and close's, so a scan
#: that found none cannot pass.
CLOSER_SITES: Final[int] = 2


def close_sites(source: str, where: str) -> list[str]:
    """Every reference to :data:`CLOSE` in one module, as ``<where>:<line>``.

    Args:
        source: The module's text.
        where: Its name, for the sites.

    Returns:
        One site per reference, in the order the tree is walked.
    """
    sites: list[str] = []
    for node in ast.walk(ast.parse(source)):
        match node:
            case ast.Name(id=name) | ast.Attribute(attr=name) if name == CLOSE:
                sites.append(f"{where}:{node.lineno}")
            case ast.ImportFrom(names=aliases) if any(alias.name == CLOSE for alias in aliases):
                sites.append(f"{where}:{node.lineno}")
            case _:
                pass
    return sites


def test_only_job_close_closes_a_queue_job() -> None:
    modules = sorted(SOURCE.rglob("*.py"))
    elsewhere = [
        site
        for module in modules
        if module != CLOSER
        for site in close_sites(module.read_text(encoding="utf-8"), str(module))
    ]

    assert len(modules) > 100
    assert len(close_sites(CLOSER.read_text(encoding="utf-8"), str(CLOSER))) == CLOSER_SITES
    assert elsewhere == []


def test_a_call_an_attribute_and_an_import_are_each_found() -> None:
    source = (
        "from fleet.core.queue import report_close as shut\n"
        "def f(creds):\n"
        "    queue.report_close(creds, job_id='j')\n"
        "    report_close(creds, job_id='j')\n"
        "    later = queue.report_close\n"
        "    queue.report_start(creds, job_id='j')\n"
        "    return 'report_close'\n"
    )

    assert sorted(close_sites(source, "m.py")) == ["m.py:1", "m.py:3", "m.py:4", "m.py:5"]
