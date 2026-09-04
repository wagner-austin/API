"""CLI: score already-generated completions against the repo's checkers.

Generation is deliberately NOT this tool's job. The instrument scores files
that already exist on disk, so the same code scores a base model, a trained
adapter, or a human writing by hand, and a sweep can be re-scored after a
guard rule changes without regenerating anything.

Usage:
    code-style-eval --holdout H.jsonl --generated-dir D --interpreter P \\
        --arm candidate --out outcomes.jsonl --check-cwd PACKAGE_DIR
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core.config import config_test_hooks
from platform_core.continuation_task import (
    EvalPrompt,
    build_prompts,
    generated_path,
    item_root,
)
from platform_core.json_utils import dump_json_str

from code_style_eval.cli import _test_hooks
from code_style_eval.contracts.outcomes import ItemOutcome, encode_item_outcome
from code_style_eval.core.checks import checker_environment, score_item
from code_style_eval.core.provenance import verify_scoring_environment

_HOLDOUT_FLAG = "--holdout"
_GENERATED_FLAG = "--generated-dir"
_INTERPRETER_FLAG = "--interpreter"
_ARM_FLAG = "--arm"
_OUT_FLAG = "--out"
_PROMPT_LINES_FLAG = "--prompt-lines"
_CWD_FLAG = "--check-cwd"

_FLAGS: tuple[str, ...] = (
    _HOLDOUT_FLAG,
    _GENERATED_FLAG,
    _INTERPRETER_FLAG,
    _ARM_FLAG,
    _OUT_FLAG,
    _PROMPT_LINES_FLAG,
    _CWD_FLAG,
)

_DEFAULT_PROMPT_LINES = 20


class Arguments:
    """Parsed command-line arguments.

    Attributes:
        holdout: Holdout JSONL emitted by code-corpus.
        generated_dir: Directory holding one generated file per item.
        interpreter: Interpreter that has ruff, mypy and the guards.
        arm: Name recorded on every outcome, identifying the model.
        out: Where to write the outcomes, one JSON object per line.
        prompt_lines: How many lines of each file the model was shown.
        check_cwd: Package directory the checkers are invoked from, which is
            what makes ``python -m scripts.guard`` resolve. It is not the
            tree being checked: each item's own root is.
    """

    __slots__ = (
        "arm",
        "check_cwd",
        "generated_dir",
        "holdout",
        "interpreter",
        "out",
        "prompt_lines",
    )

    def __init__(
        self,
        *,
        holdout: pathlib.Path,
        generated_dir: pathlib.Path,
        interpreter: str,
        arm: str,
        out: pathlib.Path,
        prompt_lines: int,
        check_cwd: pathlib.Path,
    ) -> None:
        """Store the parsed arguments.

        Args:
            holdout: Holdout JSONL path.
            generated_dir: Directory of generated files.
            interpreter: Interpreter path.
            arm: Arm name.
            out: Output path.
            prompt_lines: Prompt length in lines.
            check_cwd: Package directory the checkers are invoked from.
        """
        self.holdout = holdout
        self.generated_dir = generated_dir
        self.interpreter = interpreter
        self.arm = arm
        self.out = out
        self.prompt_lines = prompt_lines
        self.check_cwd = check_cwd


def _take_value(tokens: Sequence[str], index: int, flag: str) -> str:
    """Read the value following a flag.

    Args:
        tokens: All argument tokens.
        index: Index of the flag itself.
        flag: The flag name, for the error message.

    Returns:
        The value.

    Raises:
        ValueError: If the flag is the last token.
    """
    if index + 1 >= len(tokens):
        raise ValueError(f"{flag} requires a value")
    return tokens[index + 1]


def parse_arguments(tokens: Sequence[str]) -> Arguments:
    """Parse the command line.

    Args:
        tokens: Arguments excluding the program name.

    Returns:
        The parsed arguments.

    Raises:
        ValueError: If a flag is unknown, a required flag is missing, or
            ``--prompt-lines`` is not a positive integer.
    """
    values: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in _FLAGS:
            raise ValueError(f"unknown argument '{token}'; known flags: {_FLAGS}")
        values[token] = _take_value(tokens, index, token)
        index += 2

    for required in (
        _HOLDOUT_FLAG,
        _GENERATED_FLAG,
        _INTERPRETER_FLAG,
        _ARM_FLAG,
        _OUT_FLAG,
        _CWD_FLAG,
    ):
        if required not in values:
            raise ValueError(f"{required} is required")

    raw_lines = values.get(_PROMPT_LINES_FLAG, str(_DEFAULT_PROMPT_LINES))
    if not raw_lines.isdigit() or int(raw_lines) <= 0:
        raise ValueError(f"{_PROMPT_LINES_FLAG} must be a positive integer, got '{raw_lines}'")

    return Arguments(
        holdout=pathlib.Path(values[_HOLDOUT_FLAG]),
        generated_dir=pathlib.Path(values[_GENERATED_FLAG]),
        interpreter=values[_INTERPRETER_FLAG],
        arm=values[_ARM_FLAG],
        out=pathlib.Path(values[_OUT_FLAG]),
        prompt_lines=int(raw_lines),
        check_cwd=pathlib.Path(values[_CWD_FLAG]),
    )


def score_arm(arguments: Arguments, prompts: Sequence[EvalPrompt]) -> list[ItemOutcome]:
    """Score every prompt whose generation exists.

    An item with no generated file is SKIPPED rather than recorded as a
    failure. A missing generation means the sweep did not produce one, which
    is a fact about the run and not about the model's style; recording it as
    a failure would let a crashed generation masquerade as a style result.

    Args:
        arguments: Parsed arguments.
        prompts: Prompts built from the holdout.

    Returns:
        One outcome per scored item, in prompt order.
    """
    # Built ONCE. Rebuilding per item would glob the repository once per
    # item for an answer that cannot change during a sweep.
    #
    # The parent environment comes from platform_core's hook rather than
    # from os.environ, which the env guard bans outside that module. The
    # hook exists for this exact case: subprocess REPLACES the environment
    # when given one, so a caller adding a single variable has to start
    # from the parent's.
    env = checker_environment(arguments.check_cwd, config_test_hooks.get_environment())
    outcomes: list[ItemOutcome] = []
    for prompt in prompts:
        target = generated_path(arguments.generated_dir, prompt["item_id"])
        if not target.is_file():
            continue
        outcomes.append(
            score_item(
                item_id=prompt["item_id"],
                arm=arguments.arm,
                interpreter=arguments.interpreter,
                target=target,
                root=item_root(arguments.generated_dir, prompt["item_id"]),
                cwd=arguments.check_cwd,
                env=env,
            )
        )
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    """Score an arm's generations and write the outcomes.

    Args:
        argv: Arguments excluding the program name. Defaults to the process
            arguments.

    Returns:
        Exit code 0 when the outcomes were written.

    Raises:
        importlib.metadata.PackageNotFoundError: Propagated from
            :func:`verify_scoring_environment` when the scoring environment is
            not the one a comparison of these outcomes would claim. Checked
            BEFORE any work, because the failure it prevents is a run that
            succeeds and is wrong.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    arguments = parse_arguments(tokens)
    verify_scoring_environment(_test_hooks.record_distributions)

    records = arguments.holdout.read_text(encoding="utf-8").splitlines()
    prompts = build_prompts(records, arguments.prompt_lines)
    outcomes = score_arm(arguments, prompts)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(dump_json_str(encode_item_outcome(outcome)) + "\n" for outcome in outcomes)
    arguments.out.write_text(body, encoding="utf-8")

    passed = sum(1 for outcome in outcomes if outcome["all_passed"])
    _test_hooks.emit(
        f"arm {arguments.arm}: scored {len(outcomes)} of {len(prompts)} prompt(s), "
        f"{passed} passed every checker"
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point."""
    raise SystemExit(main())


__all__ = [
    "Arguments",
    "entrypoint",
    "main",
    "parse_arguments",
    "score_arm",
]


# Without this, `python -m code_style_eval.cli.evaluate` imports the module,
# defines these functions and exits 0 -- having scored nothing while
# reporting success. That is worse than a crash: the console script works, so
# the two invocation forms disagree, and the silent one looks exactly like a
# run that legitimately produced no output.
#
# It cost a real scoring run on 2026-09-04, against 226 generated files that
# had just taken an A30 thirty-three minutes to produce. The sibling packages
# `model_trainer` and `hpc3` each carry this guard and a test that enforces
# it; this package carried neither.
if __name__ == "__main__":
    entrypoint()
