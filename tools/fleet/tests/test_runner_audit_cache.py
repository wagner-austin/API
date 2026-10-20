"""The cache rows: which directories a runner host is held to, and how.

The rows run under the audit's own functions with ``du`` a stand-in that
answers as each case says (tests/_audit_rows.py), so what is asserted is the
verdict each row reaches.
"""

from __future__ import annotations

import pathlib

import pytest

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import runner_audit_cache
from tests._audit_rows import run_rows
from tests._runner_fixtures import a_host, an_install


def _host() -> HostRunnerSpec:
    """A host with two installs of one repository.

    Returns:
        The spec.
    """
    return a_host(
        [an_install("wagner-austin/MCPs"), an_install("wagner-austin/MCPs", "lavender-2", slot=2)]
    )


class TestCacheRows:
    def test_the_cache_comes_first_then_each_work_tree_in_roster_order(self) -> None:
        assert runner_audit_cache.cache_rows(_host()) == [
            runner_audit_cache.CacheRow(
                check_id="cache:/home/gharunner/.cache:ceiling-60gb",
                path="/home/gharunner/.cache",
                ceiling_gb=60,
                reason=runner_audit_cache.CACHE_REASON,
            ),
            runner_audit_cache.CacheRow(
                check_id="cache:/home/gharunner/actions-runner-mcps-1/_work:ceiling-15gb",
                path="/home/gharunner/actions-runner-mcps-1/_work",
                ceiling_gb=15,
                reason=runner_audit_cache.WORK_REASON,
            ),
            runner_audit_cache.CacheRow(
                check_id="cache:/home/gharunner/actions-runner-mcps-2/_work:ceiling-15gb",
                path="/home/gharunner/actions-runner-mcps-2/_work",
                ceiling_gb=15,
                reason=runner_audit_cache.WORK_REASON,
            ),
        ]


class TestTheRowsRunForReal:
    def test_each_directory_within_its_own_ceiling_passes(self, tmp_path: pathlib.Path) -> None:
        rows = run_rows(
            tmp_path,
            runner_audit_cache.render_cache_check_lines(_host()),
            {"du": 'case "$3" in */.cache) echo "60G	$3" ;; *) echo "15G	$3" ;; esac'},
        )
        assert rows == [
            "CHECK cache:/home/gharunner/.cache:ceiling-60gb OK",
            "CHECK cache:/home/gharunner/actions-runner-mcps-1/_work:ceiling-15gb OK",
            "CHECK cache:/home/gharunner/actions-runner-mcps-2/_work:ceiling-15gb OK",
        ]

    def test_a_tree_past_its_ceiling_drifts_naming_what_it_holds(
        self, tmp_path: pathlib.Path
    ) -> None:
        [_, first, _] = run_rows(
            tmp_path,
            runner_audit_cache.render_cache_check_lines(_host()),
            {"du": 'case "$3" in *mcps-1*) echo "16G	$3" ;; *) echo "1G	$3" ;; esac'},
        )
        assert first == (
            "CHECK cache:/home/gharunner/actions-runner-mcps-1/_work:ceiling-15gb DRIFT "
            "/home/gharunner/actions-runner-mcps-1/_work holds 16 GB against a ceiling of 15 GB; "
            "du said: 16G	/home/gharunner/actions-runner-mcps-1/_work (exit 0)"
        )

    @pytest.mark.parametrize(
        ("du", "said"),
        [
            ('echo "du: cannot access $3: No such file or directory" >&2; exit 1', "exit 1"),
            ('echo "du: fts_read failed: Input/output error"', "exit 0"),
        ],
    )
    def test_an_answer_that_is_not_a_size_drifts_and_never_passes(
        self, tmp_path: pathlib.Path, du: str, said: str
    ) -> None:
        [cache, *_] = run_rows(
            tmp_path, runner_audit_cache.render_cache_check_lines(_host()), {"du": du}
        )
        assert cache.startswith(
            "CHECK cache:/home/gharunner/.cache:ceiling-60gb DRIFT /home/gharunner/.cache holds "
            "an unreadable figure GB against a ceiling of 60 GB; du said: "
        )
        assert cache.endswith(f"({said})")

    @pytest.mark.parametrize("bad", ["/home/it's", "/home/$x", "/home/a`b"])
    def test_a_path_that_cannot_be_embedded_is_refused(self, bad: str) -> None:
        install = an_install()
        install["workdir"] = bad
        with pytest.raises(ValueError, match="cannot be embedded"):
            runner_audit_cache.render_cache_check_lines(a_host([install]))
