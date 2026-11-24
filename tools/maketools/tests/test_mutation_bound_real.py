"""The real lifted step held to its bound: real git, cosmic-ray and pytest (MCPs task 3b386c0d).

MCPs fleet job 24f79dd5 ran the step 33 minutes past its 900 s bound until
the fleet's lease ended the check, and it printed nothing. This is MCPs'
case for it (``mcp-shared-py/tests/test_mutation_bound_real.py``) through
API's boundary: maketools' real hooks (:func:`maketools.mutation.launch.mutation_hooks`)
and API's codec, over the fixture ``test_mutation_real.py`` builds. A bound
of 0 s has come before the first trial can start, so the run makes the
seven mutants of ``value >= floor`` and runs none, whatever the machine's
speed: the step must name each one as not run, refuse
``MUTATION_OVER_BOUND`` saying how many it ran, still record its span, put
the source's bytes back and remove its scratch directory. That the
launcher hands the step ``check-budget``'s own bound is
``test_mutation_launch.py``'s case.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from maketools import _test_hooks
from maketools.mutation.context import MutationErrorCode, MutationStepError
from maketools.mutation.launch import API_CODEC, mutation_hooks
from maketools.mutation.step import run_in_scratch
from tests.test_mutation_real import (
    PYPROJECT,
    SOURCE,
    WALL_SECONDS,
    WEAK_TESTS,
    _commit,
    _git,
)

#: The comparison's mutant ``test_mutation_real.py`` sees survive, named here as not run.
COMPARISON = (
    "MUTANT NOT RUN, THE BOUND CAME src/compare.py:2:18 ReplaceComparisonOperator_GtE_Gt -> "
    '"return value >= floor"'
)

#: pytest-cov's options recording which test ran each line, as ``run.py test`` passes them.
COVERAGE_CONTEXTS = ("--cov=src", "--cov-context=test")


def test_a_run_whose_bound_has_come_runs_no_trial_names_each_mutant_and_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "fixture"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, {"pyproject.toml": PYPROJECT, ".gitignore": "runs/\n.coverage\n"}, "start")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _commit(repo, {"src/compare.py": SOURCE, "tests/test_compare.py": WEAK_TESTS}, "weak")
    env = {"PATH": _test_hooks.environ()["PATH"], "HOME": str(tmp_path)}
    # The suite with the per-test coverage contexts the step reads, as
    # ``run.py test`` runs it; none of this suite's own coverage settings.
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *COVERAGE_CONTEXTS],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        timeout=WALL_SECONDS,
    )
    (repo / "runs").mkdir()
    scratch = repo / "runs" / ".mutation-bound"
    spans: list[int] = []
    with pytest.raises(MutationStepError) as refused:
        run_in_scratch(
            mutation_hooks(repo, scratch),
            API_CODEC,
            repo,
            sys.executable,
            "",
            env,
            spans.append,
            lambda: 0,
        )
    assert refused.value.code is MutationErrorCode.OVER_BOUND
    assert str(refused.value) == (
        "MUTATION_OVER_BOUND: 0 of the 7 mutant(s) the tests run in 1 source file(s) had been "
        "run when the step's 0 s bound came, so 7 were not run, beside 0 already uncaught, each "
        "named above; HEAD's bytes are back in each file it mutates and its span is recorded. "
        "A change too large to mutate inside the bound is one to split"
    )
    lines = capsys.readouterr().out.splitlines()
    mutants = [line for line in lines if line.startswith("MUTANT ")]
    assert len(mutants) == 7
    assert all(
        line.startswith("MUTANT NOT RUN, THE BOUND CAME src/compare.py:2:") for line in mutants
    )
    assert COMPARISON in mutants
    # No trial ran, so every timing total counts none, and no verdict line follows.
    totals = [line for line in lines if line.startswith("mutation timing:")]
    assert len(totals) == 4
    assert all(" runs=0 " in line for line in totals)
    assert not any(line.startswith(("mutation: passed:", "mutation: FAILED:")) for line in lines)
    assert len(spans) == 1
    assert (repo / "src" / "compare.py").read_text(encoding="utf-8") == SOURCE
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert list((repo / "runs").iterdir()) == []
