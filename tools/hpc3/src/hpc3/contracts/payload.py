"""Whether a command can run where the batch script puts it.

A host run's command is a LINE IN THE BATCH SCRIPT, so bash runs it and every
shell construct works. An imaged run reads the same in the document and is
different in a way nothing else shows: the script renders

    apptainer exec ... "<image>" \\
        env PATH="<env>/bin:$PATH" \\
        <command>

and ``<command>`` is interpolated raw into that continuation. An unquoted
``&&`` therefore does NOT reach the container at all -- bash splits the line
first, so ``apptainer exec ... env ... cd /pub/x`` runs as one command, ``env``
tries to exec a program named ``cd``, and everything after the operator runs
on the HOST, outside the image.

WHAT THAT COST, on 2026-08-28. Seven training jobs went out with a
``cd ... && ... && python -m char_lstm.train ...`` payload, copied from the
host-run project beside it. All seven died in under three seconds with
``/usr/bin/env: 'cd': No such file or directory``, exit 127, before a GPU was
touched. Preflight passed, and correctly: ``sbatch --test-only`` answers
whether the SCHEDULER would admit the job, never whether the command can
execute.

QUOTING IS THE WHOLE DIFFICULTY, and the first version of this check got it
wrong. ``python -c 'import torch, json; print(...)'`` contains a semicolon and
is fine -- bash's quoting hands it to python as one argument -- and six
committed probe documents that had really run were refused by a naive
substring scan. So the command is lexed, and only a token that IS an operator
counts. ``$SLURM_JOB_ID`` is likewise left alone: the outer shell expands it
before apptainer sees it, which is useful and works.
"""

from __future__ import annotations

import shlex

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.image import ImageReference

SHELL_OPERATORS = frozenset({"&&", "||", ";", "|", ">", ">>", "<", "&"})
"""Tokens that split a command line, which only a shell acts on.

Deliberately NOT including ``VAR=value`` prefixes: the batch script runs the
payload through ``env``, which takes assignments before the command, so
``floor``'s ``HF_HOME=... TRANSFORMERS_OFFLINE=1 modeltrainer-score-baseline``
needs no shell at all.
"""

SHELL_PREFIXES = ("bash -c ", "sh -c ")
"""Asking for a shell explicitly, which is the supported spelling."""


def check_imaged_command_can_run(image: ImageReference | None, command: str) -> None:
    """Refuse an imaged payload that only a shell could execute.

    Args:
        image: Image the payload runs inside, or None for a host run.
        command: The payload, as the document states it.

    Raises:
        AppError: With ``IMAGED_COMMAND_NEEDS_A_SHELL`` when an imaged command
            carries an unquoted shell operator without asking for a shell, or
            when its quoting is unbalanced and no lexer can say what it means.
    """
    if image is None or command.startswith(SHELL_PREFIXES):
        return
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise AppError(
            Hpc3ErrorCode.IMAGED_COMMAND_NEEDS_A_SHELL,
            f"This command's quoting does not close: {error}. It is rendered "
            "into a batch script line, so a quote left open swallows what "
            "follows it.",
        ) from error
    operators = [token for token in tokens if token in SHELL_OPERATORS]
    if operators == []:
        return
    raise AppError(
        Hpc3ErrorCode.IMAGED_COMMAND_NEEDS_A_SHELL,
        f"This command runs inside an image and uses {operators[0]!r}, which "
        "only a shell performs. An imaged payload is interpolated into the "
        "batch script's `apptainer exec` line, so the operator splits that "
        "line: the first half fails trying to exec its first word, and "
        f'everything after it runs OUTSIDE the image. Wrap it: bash -c "'
        f'{command}". A host run needs no wrapper.',
    )


__all__ = ["SHELL_OPERATORS", "SHELL_PREFIXES", "check_imaged_command_can_run"]
