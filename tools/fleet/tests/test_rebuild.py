"""The rebuild lane's pure half: argv composition, refusals, result prose.

The tick-level behaviour -- a claimed ``build-bases`` job reported started,
executed, and closed -- lives in ``test_agent.py`` beside the other claim
outcomes; this module holds what needs no queue: the exact make invocation
(the label rides as a make command-line variable, measured before trusted),
the refusals that keep an unparseable label out of the fleet journal, and
the closing detail's tail.
"""

from __future__ import annotations

import pathlib

from fleet.core import _test_hooks, rebuild


class TestRebuildArgv:
    def test_composes_the_exact_make_invocation(self) -> None:
        root = pathlib.Path("C:/Users/Test/PROJECTS/MCPs")

        argv = rebuild.rebuild_argv(root, "opus-fleet-0911")

        assert argv == (
            "make",
            "-C",
            str(root),
            "build-bases",
            "BOARD_AGENT_LABEL=opus-fleet-0911",
        )


class TestRefusals:
    def test_a_missing_checkout_is_refused_by_code(self, tmp_path: pathlib.Path) -> None:
        absent = tmp_path / "no-such-checkout"

        refusal = rebuild.refusal_for(absent, "opus-fleet-0911")

        if refusal is None:
            raise AssertionError("expected a refusal for an absent checkout")
        assert refusal.startswith(f"{rebuild.ROOT_MISSING_CODE}:")
        assert str(absent) in refusal

    def test_a_label_outside_the_board_grammar_is_refused_by_code(
        self, tmp_path: pathlib.Path
    ) -> None:
        # The queue only length-checks the label, and this string is 17
        # characters of valid length that would land inside an argv element
        # and then inside the journal's agent field.
        refusal = rebuild.refusal_for(tmp_path, 'x" && echo pwned')

        if refusal is None:
            raise AssertionError("expected a refusal for a lawless label")
        assert refusal.startswith(f"{rebuild.LABEL_INVALID_CODE}:")

    def test_an_uppercase_label_is_refused_the_same_way(self, tmp_path: pathlib.Path) -> None:
        refusal = rebuild.refusal_for(tmp_path, "Opus-Fleet-0911")

        if refusal is None:
            raise AssertionError("expected a refusal for an uppercase label")
        assert refusal.startswith(f"{rebuild.LABEL_INVALID_CODE}:")

    def test_a_real_checkout_and_a_kebab_label_pass(self, tmp_path: pathlib.Path) -> None:
        assert rebuild.refusal_for(tmp_path, "opus-fleet-0911") is None


class TestDescribeResult:
    def test_carries_the_exit_code_and_the_combined_output(self) -> None:
        detail = rebuild.describe_result(
            _test_hooks.CommandResult(
                returncode=2, stdout="Base images rebuilt.\n", stderr="warning: x\n"
            )
        )

        assert detail == "make build-bases exited 2: Base images rebuilt.\nwarning: x"

    def test_keeps_the_tail_of_a_long_output_where_make_puts_the_failure(
        self,
    ) -> None:
        head = "layer export line\n" * 400
        detail = rebuild.describe_result(
            _test_hooks.CommandResult(
                returncode=1, stdout=head, stderr="make: *** [Makefile:73: build-bases] Error 1"
            )
        )

        assert len(detail) <= rebuild.DETAIL_TAIL_CHARS + len("make build-bases exited 1: ")
        assert detail.endswith("make: *** [Makefile:73: build-bases] Error 1")
