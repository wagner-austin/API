"""Tests for the CI job time-bound rule.

The rule exists because an unbounded job on a frozen runner held CI for
about three hours on 2026-09-09 while every job still read ``in_progress``.
The cases below drive the real parser and the real rule; the last class
runs both against THIS repository, which is what stops the rule from being
a definition nothing exercises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from monorepo_guards.config import GuardConfig
from monorepo_guards.util import find_monorepo_root
from monorepo_guards.workflow_timeout_rules import (
    MAX_BOUND_MINUTES,
    JobBound,
    WorkflowTimeoutRule,
    parse_workflow_jobs,
    workflow_files,
)

#: This repository's root, as a plain path.
#:
#: Derived from this file's own location rather than from
#: :func:`find_monorepo_root`, whose ``Path | None`` would force either an
#: ``assert ... is not None`` -- which the weak-assertion guard correctly
#: refuses -- or a ``None`` arm that can never be taken and so can never be
#: covered. The two are cross-checked by
#: :meth:`TestThisRepository.test_the_root_matches_the_finder`.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

BOUNDED = """\
name: Example

"on":
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read

jobs:
  check:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - run: echo hi
"""

UNBOUNDED = """\
name: Example

"on":
  push:

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
"""


def _config(root: Path) -> GuardConfig:
    """Build a configuration whose monorepo root is the given tree.

    Args:
        root: The tree to treat as the repository root.

    Returns:
        The configuration.
    """
    return GuardConfig(
        root=root,
        monorepo_root=root,
        directories=("src",),
        exclude_parts=(),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )


def _workflow(root: Path, name: str, body: str) -> Path:
    """Write one workflow file into a tree.

    Args:
        root: The repository root.
        name: The workflow's file name.
        body: Its contents.

    Returns:
        The path written.
    """
    directory = root / ".github" / "workflows"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


class TestParsing:
    """What counts as a job, and what its bound is."""

    def test_it_reads_a_bound(self, tmp_path: Path) -> None:
        """The ordinary case: one job, one bound, and the line carrying it."""
        jobs = parse_workflow_jobs(_workflow(tmp_path, "a.yml", BOUNDED))
        assert [(j.job_id, j.bound) for j in jobs] == [("check", JobBound(minutes=15, line_no=15))]

    def test_a_trailing_comment_does_not_hide_a_job(self, tmp_path: Path) -> None:
        """SILENT SKIP, the one direction this rule cannot afford.

        An earlier parser required a colon at end of line, so
        ``  check:  # windows only`` was not a job and its missing bound
        was never reported. A false alarm is arguable; a job that
        disappears from an enforcement rule is not.
        """
        body = 'name: T\n\n"on":\n  push:\n\njobs:\n  check:  # windows only\n    runs-on: x\n'
        assert [j.job_id for j in parse_workflow_jobs(_workflow(tmp_path, "a.yml", body))] == [
            "check"
        ]

    def test_a_trailing_comment_does_not_hide_a_bound(self, tmp_path: Path) -> None:
        """The same skip on the bound would invent a missing-bound finding."""
        body = (
            'name: T\n\n"on":\n  push:\n\njobs:\n'
            "  check:\n    runs-on: x\n    timeout-minutes: 20  # measured\n"
        )
        jobs = parse_workflow_jobs(_workflow(tmp_path, "a.yml", body))
        assert jobs[0].bound == JobBound(minutes=20, line_no=9)

    def test_a_four_space_workflow_is_parsed(self, tmp_path: Path) -> None:
        """The job indent is LEARNED, not assumed to be two spaces.

        YAML does not require two, and a file indented four would
        otherwise contribute no jobs at all -- passing silently.
        """
        body = (
            'name: T\n\n"on":\n  push:\n\njobs:\n'
            "    build:\n        runs-on: x\n        timeout-minutes: 15\n"
            "    deploy:\n        runs-on: x\n"
        )
        jobs = parse_workflow_jobs(_workflow(tmp_path, "a.yml", body))
        assert [(j.job_id, j.bound is None) for j in jobs] == [("build", False), ("deploy", True)]

    def test_a_bare_key_inside_a_job_is_not_a_job(self, tmp_path: Path) -> None:
        """``steps:`` is a bare key too; only the job indent distinguishes it."""
        jobs = parse_workflow_jobs(_workflow(tmp_path, "a.yml", BOUNDED))
        assert [j.job_id for j in jobs] == ["check"]

    def test_a_job_without_a_bound_carries_none(self, tmp_path: Path) -> None:
        """Absent must be its own value, not zero: zero is a bound."""
        jobs = parse_workflow_jobs(_workflow(tmp_path, "a.yml", UNBOUNDED))
        assert jobs[0].bound is None

    def test_keys_under_on_and_permissions_are_not_jobs(self, tmp_path: Path) -> None:
        """THE PARSE BUG THIS PREVENTS, and it fails toward false alarms.

        ``push:`` under ``on:`` is also a two-space key. Counting it as a
        job would report an unbounded job nobody can bound, and a rule that
        cannot be satisfied gets switched off rather than obeyed.
        """
        jobs = parse_workflow_jobs(_workflow(tmp_path, "a.yml", BOUNDED))
        assert [j.job_id for j in jobs] == ["check"]

    def test_it_points_at_the_job_not_the_file(self, tmp_path: Path) -> None:
        """A violation a reader has to search for is one they skim past."""
        jobs = parse_workflow_jobs(_workflow(tmp_path, "a.yml", BOUNDED))
        assert jobs[0].line_no == 13

    def test_every_job_in_a_multi_job_workflow_is_found(self, tmp_path: Path) -> None:
        """A bound on the first job must not be credited to the second."""
        body = (
            'name: Two\n\n"on":\n  push:\n\njobs:\n'
            "  build:\n    runs-on: ubuntu-latest\n    timeout-minutes: 15\n"
            "  deploy:\n    runs-on: ubuntu-latest\n"
        )
        jobs = parse_workflow_jobs(_workflow(tmp_path, "a.yml", body))
        assert [(j.job_id, j.bound is None) for j in jobs] == [("build", False), ("deploy", True)]

    def test_a_section_after_jobs_does_not_contribute_jobs(self, tmp_path: Path) -> None:
        """A column-zero key closes the map; what follows is not a job."""
        body = (
            'name: T\n\n"on":\n  push:\n\njobs:\n'
            "  check:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n"
            "concurrency:\n  group: g\n"
        )
        jobs = parse_workflow_jobs(_workflow(tmp_path, "a.yml", body))
        assert [j.job_id for j in jobs] == ["check"]

    def test_a_timeout_before_any_job_is_not_attached(self, tmp_path: Path) -> None:
        """With no job open there is nothing to attach a bound to.

        Guards the indexing: the parser must not read the last element of
        an empty list.
        """
        body = 'name: T\n\n"on":\n  push:\n\njobs:\n    timeout-minutes: 5\n'
        assert parse_workflow_jobs(_workflow(tmp_path, "a.yml", body)) == []


class TestDiscovery:
    """Which files the rule considers."""

    def test_it_finds_yml_and_yaml(self, tmp_path: Path) -> None:
        """Both spellings are workflows; missing one would be a blind spot."""
        _workflow(tmp_path, "a.yml", BOUNDED)
        _workflow(tmp_path, "b.yaml", BOUNDED)
        assert [p.name for p in workflow_files(tmp_path)] == ["a.yml", "b.yaml"]

    def test_it_ignores_other_files(self, tmp_path: Path) -> None:
        """A README beside the workflows is not a workflow."""
        _workflow(tmp_path, "a.yml", BOUNDED)
        (tmp_path / ".github" / "workflows" / "README.md").write_text("x", encoding="utf-8")
        assert [p.name for p in workflow_files(tmp_path)] == ["a.yml"]

    def test_a_tree_with_no_workflows_yields_none(self, tmp_path: Path) -> None:
        """Most of the forty-one packages are checked against such a tree.

        Returning nothing is a true statement -- no workflows, no unbounded
        jobs -- and the rule must not manufacture a finding about a
        directory that is not there. That THIS repository does have
        workflows is asserted separately, below.
        """
        assert workflow_files(tmp_path) == []


class TestTheRule:
    """What the guard reports."""

    def test_a_bounded_workflow_is_clean(self, tmp_path: Path) -> None:
        """The healthy case must not raise or report."""
        _workflow(tmp_path, "a.yml", BOUNDED)
        assert WorkflowTimeoutRule(_config(tmp_path)).run([]) == []

    def test_an_unbounded_job_is_reported(self, tmp_path: Path) -> None:
        """The defect itself."""
        _workflow(tmp_path, "a.yml", UNBOUNDED)
        found = WorkflowTimeoutRule(_config(tmp_path)).run([])
        assert len(found) == 1
        assert found[0].kind == "missing-timeout-minutes"
        assert "check" in found[0].line

    def test_a_bound_above_the_ceiling_is_reported(self, tmp_path: Path) -> None:
        """A 300-minute bound is the default wearing a number."""
        _workflow(tmp_path, "a.yml", BOUNDED.replace("timeout-minutes: 15", "timeout-minutes: 300"))
        found = WorkflowTimeoutRule(_config(tmp_path)).run([])
        assert [v.kind for v in found] == ["implausible-timeout-minutes"]

    def test_the_ceiling_itself_is_allowed(self, tmp_path: Path) -> None:
        """The packages matrix sits exactly on it, so off-by-one matters."""
        body = BOUNDED.replace("timeout-minutes: 15", f"timeout-minutes: {MAX_BOUND_MINUTES}")
        _workflow(tmp_path, "a.yml", body)
        assert WorkflowTimeoutRule(_config(tmp_path)).run([]) == []

    def test_a_zero_bound_is_reported(self, tmp_path: Path) -> None:
        """Zero parses as a number and bounds nothing."""
        _workflow(tmp_path, "a.yml", BOUNDED.replace("timeout-minutes: 15", "timeout-minutes: 0"))
        found = WorkflowTimeoutRule(_config(tmp_path)).run([])
        assert [v.kind for v in found] == ["implausible-timeout-minutes"]

    def test_it_reports_every_offender_not_only_the_first(self, tmp_path: Path) -> None:
        """One fixed job must not hide the next."""
        _workflow(tmp_path, "a.yml", UNBOUNDED)
        _workflow(tmp_path, "b.yml", UNBOUNDED)
        assert len(WorkflowTimeoutRule(_config(tmp_path)).run([])) == 2

    def test_the_package_files_are_not_its_subject(self, tmp_path: Path) -> None:
        """The rule reads workflows, so the file list cannot change it."""
        _workflow(tmp_path, "a.yml", UNBOUNDED)
        rule = WorkflowTimeoutRule(_config(tmp_path))
        assert rule.run([]) == rule.run([tmp_path / "nonexistent.py"])


class TestThisRepository:
    """Run the rule against the real workflows, not only against fixtures.

    WITHOUT THIS THE RULE COULD PASS ON NOTHING. Every case above builds
    its own tree, so all of them would stay green if the parser stopped
    recognising real workflows entirely -- which is how a guard outlives
    its subject. These two assert the rule still has one, and that this
    repository satisfies it.
    """

    def test_the_root_matches_the_finder(self) -> None:
        """The constant and the real locator must name the same directory.

        If this file ever moves, the parents[3] hop silently starts naming
        some other tree and every case below would then assert about it
        instead of about this repository.
        """
        assert find_monorepo_root(Path(__file__).parent) == REPO_ROOT

    def test_every_real_workflow_yields_at_least_one_job(self) -> None:
        """PER FILE, because an aggregate floor hides a partial blind spot.

        An earlier version asserted only ``len(jobs) > 5`` across all
        files while the repo declares nine, so up to three could stop
        being recognised -- by a comment, an indent, any parse
        assumption -- and the floor would still hold. The subject would
        not be empty, it would be SHORT, and short is the harder one to
        notice. Naming the file that yields nothing is the difference.
        """
        empty = [p.name for p in workflow_files(REPO_ROOT) if not parse_workflow_jobs(p)]
        assert empty == []

    def test_it_finds_this_repositorys_jobs(self) -> None:
        """Non-vacuity: the parse must still recognise real workflows."""
        jobs = [job for path in workflow_files(REPO_ROOT) for job in parse_workflow_jobs(path)]
        assert len(jobs) > 5

    def test_every_job_in_this_repository_is_bounded(self) -> None:
        """The property the rule exists to keep, asserted on the real files."""
        found = WorkflowTimeoutRule(_config(REPO_ROOT)).run([])
        assert [f"{v.file.name}:{v.line_no} {v.line}" for v in found] == []

    def test_removing_a_real_bound_is_caught(self, tmp_path: Path) -> None:
        """WATCH IT FAIL, on real workflows rather than on my own fixtures.

        Every other failing case here is built from a string in this file,
        so all of them share whatever I assumed a workflow looks like. This
        copies the repository's ACTUAL workflows, deletes one
        ``timeout-minutes`` line, and requires the rule to name that job --
        so the fixtures and the subject cannot agree with each other while
        both being wrong.
        """
        real = workflow_files(REPO_ROOT)
        target = next(p for p in real if any(j.bound is not None for j in parse_workflow_jobs(p)))
        victim_id, victim_bound = next(
            (j.job_id, j.bound) for j in parse_workflow_jobs(target) if j.bound is not None
        )

        # Delete the bound's OWN line, by number. Matching on the text
        # `timeout-minutes: <n>` instead would strip every job sharing
        # that value -- grandma-pages.yml has two jobs both bounded at
        # 15 -- and the single-violation assertion below would then see
        # two, failing for a reason that has nothing to do with the rule.
        kept = [
            line
            for index, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1)
            if index != victim_bound.line_no
        ]
        _workflow(tmp_path, target.name, "\n".join(kept) + "\n")

        found = WorkflowTimeoutRule(_config(tmp_path)).run([])
        assert [v.kind for v in found] == ["missing-timeout-minutes"]
        assert victim_id in found[0].line

    def test_the_watch_it_fail_case_covers_every_bounded_real_workflow(self) -> None:
        """ONE file stripped proves one file; the rule's subject is all of them.

        The audit of this rule noted the watch-it-fail case exercises
        only the first bounded workflow in sorted order. Rather than
        widen that case and slow it, this asserts the weaker thing it
        actually needs: every real workflow carrying a bound can have
        that bound located by line, which is what the strip depends on.
        """
        located = {
            path.name: [j.bound.line_no for j in parse_workflow_jobs(path) if j.bound is not None]
            for path in workflow_files(REPO_ROOT)
        }
        assert all(lines == sorted(set(lines)) for lines in located.values())
        assert sum(len(lines) for lines in located.values()) > 5
