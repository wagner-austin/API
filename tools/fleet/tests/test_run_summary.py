"""The build's last line, executed and read back (MCPs board tasks abb0a10e, 74d10d85).

The generation lines, two phases and the summary line run under this host's
bash against a cache root under ``tmp_path``, as a build does, and the
transcript they leave is read by the verdict: so a run that begins its
generation, one that reads it after, and the phases each took are what
``sh`` produced, not what the text looks like.
"""

from __future__ import annotations

import pathlib
import subprocess

from fleet.core import names, verdict
from fleet.core.phase_markers import sh_phase_lines
from fleet.core.run_summary import (
    RUN_SUMMARY_MARKER,
    UNREAD,
    RunSummary,
    read_run_summary,
    sh_run_summary_line,
)
from fleet.core.tool_caches import sh_build_lines
from tests._host_bash import host_bash


def _build(tmp_path: pathlib.Path, run: str) -> str:
    """Run one build's generation, deps and check phases and its summary.

    Args:
        tmp_path: Holds the cache root and the run directories.
        run: The run directory's name.

    Returns:
        The transcript the build left.
    """
    target = tmp_path / run
    target.mkdir()
    log = names.log_path(target.as_posix())
    lines = [
        "set -eu",
        *sh_build_lines(
            tools=f"'{(tmp_path / 'cache' / 'tools').as_posix()}'",
            marker=f"'{target.as_posix()}/{names.GENERATION_MARKER_NAME}'",
            run_as="",
            says_began=True,
        ),
        *sh_phase_lines(phase="deps", command="true", log=log),
        *sh_phase_lines(phase="check", command="echo suite", log=log),
        sh_run_summary_line(log),
    ]
    ran = subprocess.run(
        [host_bash(), "-c", "\n".join(lines) + "\n"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert ran.returncode == 0, ran.stderr
    return pathlib.Path(log).read_text(encoding="utf-8")


def test_the_first_build_says_it_began_the_generation_and_a_later_one_that_it_did_not(
    tmp_path: pathlib.Path,
) -> None:
    first = read_run_summary(_build(tmp_path, "run-1"))
    second = read_run_summary(_build(tmp_path, "run-2"))

    current = (tmp_path / "cache" / "tools" / names.CURRENT_GENERATION_NAME).read_text(
        encoding="utf-8"
    )
    assert first["generation"] == current.strip()
    assert first["began"] == "yes"
    assert second["generation"] == first["generation"]
    assert second["began"] == "no"
    assert [entry.split(":")[0] for entry in first["phases"].split(",")] == ["deps", "check"]
    assert all(entry.endswith("s") for entry in first["phases"].split(","))


def test_the_summary_is_the_transcript_s_last_line(tmp_path: pathlib.Path) -> None:
    transcript = _build(tmp_path, "run-1").splitlines()

    assert transcript[-1].startswith(f"{RUN_SUMMARY_MARKER}: generation=")
    assert transcript[-2].startswith("fleet-phase check ended ")


def test_the_verdict_carries_the_three_fields_last() -> None:
    tail = (
        "=== ALL CHECKS PASSED ===\n"
        f"{RUN_SUMMARY_MARKER}: generation=1793525277 began=yes "
        "phases=deps:7s,deps:46s,check:214s\n"
    )
    judged = verdict.judge(
        job_id="147f0920-db7b-4785-a70c-6a6b7a6a5ed6",
        project="libs/covenant_ml",
        sha="fd98be1b885b29a3efd86fde35b944f250fc4616",
        node="serendipity",
        exit_code=0,
        tail=tail,
        ended_unix=1793525544,
        log_path="/home/corvis/fleet/stage/logs/libs-covenant_ml-serendipity-1793525263.log",
        run_id="libs-covenant_ml-serendipity-1793525263",
        workers=3,
    )

    assert verdict.render_verdict(judged).endswith(
        " ci=unread dispatch=unread vmtests=unread generation=1793525277 began=yes"
        " phases=deps:7s,deps:46s,check:214s steps=unread registry=unread"
    )


def test_a_transcript_without_the_line_reads_unread() -> None:
    assert read_run_summary("fleet-phase check ended\n") == RunSummary(
        generation=UNREAD, began=UNREAD, phases=UNREAD
    )


def test_a_build_with_no_phase_says_none(tmp_path: pathlib.Path) -> None:
    log = (tmp_path / "result.txt.log").as_posix()
    script = "\n".join(["set -eu", "generation=1 generation_began=no", sh_run_summary_line(log)])
    ran = subprocess.run(
        [host_bash(), "-c", script + "\n"], capture_output=True, text=True, check=False, timeout=60
    )
    assert ran.returncode == 0, ran.stderr
    assert read_run_summary(pathlib.Path(log).read_text(encoding="utf-8"))["phases"] == "none"
