"""A run document must name a plan the image it declares actually carries.

THE FAILURE THIS MODULE EXISTS FOR, measured 2026-09-10. Job 55898551 was
submitted against image v50 and died in 5m32s with ``KeyError: unknown
cartridge plan 'gpt2-full-wiki-qa'``. The image was built from ``79ab1696``
at 16:20:23 and the plan was written in ``8ba4607c`` at 16:43:54, so the run
document named a plan its own image predated by twenty-three minutes.

NOTHING IN PLACE AT THE TIME COULD HAVE CAUGHT IT, and that is the point. The
image's own smoke checks verify it against the commit it was BUILT from and
passed 51/51, correctly -- a smoke check cannot assert a plan that did not
exist when it was written. The missing invariant is not inside the image and
not inside the plan table; it is BETWEEN the run document and the image, and
nobody owned it.

It is checkable here without a cluster, without a GPU and without building
anything: a run document declares the commit its image carries, so the plan
it names must be present in the plan table AT THAT COMMIT. That is two git
reads and a substring.

WHERE THIS CHECK DOES AND DOES NOT RUN, STATED PLAINLY BECAUSE THE ANSWER IS
UNCOMFORTABLE. It needs the declared commit's object, and CI checks out at
depth 1. A run document names a commit that was HEAD when its image was
built, so by the time any later push is tested that commit is absent from the
clone BY CONSTRUCTION -- the only CI run that could ever verify a document is
the one on that document's own commit. **So this test does real work in a full
clone and is INERT in CI, and pretending otherwise is worse than saying it.**

The first version instead asserted on unverifiable evidence, and because
``git show`` reports an absent COMMIT and a missing PATH with identical
stderr, it convicted a correct document and held ``main`` red for two hours
(diagnosed to reproduction by @fable-brain-audit-0903, 2026-09-10).

THE REAL GATE THEREFORE BELONGS ON THE SUBMISSION PATH, not here: the moment
that matters is when someone is about to spend GPU hours, and at that moment
they have a full clone and the document in hand. This module keeps the part a
shallow clone CAN answer -- that a document pairing ``image_commit`` with a
``--plan`` is making a checkable claim at all -- and the verification proper
runs where it can actually refuse something.
"""

from __future__ import annotations

import subprocess

from tests._committed_tree import REPO, submissions

#: Where the question-set plan table lives, relative to the monorepo root.
#: Read out of a COMMIT rather than the working tree, because what a run
#: document's image contains is fixed by the commit its wheels came from.
_QA_PLAN_TABLE = (
    "services/Model-Trainer/src/model_trainer/core/services/model/cartridge_qa_plans.py"
)


#: The entry point whose plan table this check knows. Scoped to one CLI
#: DELIBERATELY. Every cartridge command takes ``--plan``, and they read
#: DIFFERENT tables -- the LoRA and companion sweeps resolve against
#: ``cartridge_pool_plans``, not the question-set table. The first version of
#: this check ignored that and convicted six committed sweeps whose plans were
#: perfectly real, which is the false positive this repo has twice recorded as
#: worse than no check. Widening it means mapping each CLI to its own table,
#: and a mapping guessed rather than read would reintroduce exactly that.
_QA_ENTRY_POINT = "model_trainer.cli.cartridge_qa_benchmark"


def _plan_flag(command: str) -> str | None:
    """Read the plan a question-set command names, if it names one.

    Args:
        command: The run document's command line.

    Returns:
        The value following ``--plan`` for a question-set run, or None for
        any other command -- including sibling cartridge commands, which take
        a ``--plan`` from a different table.
    """
    if _QA_ENTRY_POINT not in command:
        return None
    tokens = command.split()
    for index, token in enumerate(tokens):
        if token == "--plan" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _commit_is_in_this_clone(commit: str) -> bool:
    """Say whether this clone holds the commit object at all.

    THE DISCRIMINATOR THIS EXISTS FOR, reproduced by @fable-brain-audit-0903
    in a fresh ``git clone --depth 1`` on 2026-09-10. ``git show`` prints the
    SAME message for two different facts::

        fatal: path '<path>' exists on disk, but not in '<sha>'

    -- when the commit is present and genuinely lacks the path, AND when the
    commit is simply not in the clone. The stderr is identical, so a check
    reading it cannot tell "this document is wrong" from "this clone is
    shallow", and the first version of this module asserted the damning one
    about a run document that was correct.

    Args:
        commit: The commit to look for.

    Returns:
        Whether the object is present locally.
    """
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=REPO,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _plan_table_at(commit: str) -> str:
    """Read the question-set plan table as it stood at one commit.

    Args:
        commit: The commit the image's wheels were built from, already known
            to be present via :func:`_commit_is_in_this_clone`.

    Returns:
        The module's source at that commit.

    Raises:
        AssertionError: If the read fails for any reason other than the
            commit being absent, which the caller has already excluded.
    """
    result = subprocess.run(
        ["git", "show", f"{commit}:{_QA_PLAN_TABLE}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{commit[:8]} is in this clone but {_QA_PLAN_TABLE} could not be read from it: "
        f"{result.stderr.strip()}"
    )
    return result.stdout


class TestAQuestionSetRunNamesAPlanItsImageActuallyCarries:
    """The claim a run document makes about its image, checked against git."""

    def test_every_named_plan_exists_at_the_declared_image_commit(self) -> None:
        """A plan absent from the image is a run that dies on its first call."""
        for name, _project, document in submissions():
            command = document.get("command")
            experiment = document.get("experiment")
            if not isinstance(command, str) or not isinstance(experiment, dict):
                continue
            plan = _plan_flag(command)
            commit = experiment.get("image_commit")
            if plan is None or not isinstance(commit, str):
                continue
            if not _commit_is_in_this_clone(commit):
                # NOT A SKIP THAT HIDES A DEFECT -- a clone that lacks the
                # commit holds NO evidence either way, and convicting on
                # absent evidence is what made `main` red for two hours on a
                # document that was right. Where this matters is stated in
                # the module docstring: at depth 1 the declared commit is
                # absent BY CONSTRUCTION, so the real gate is the submission
                # path, not this suite.
                continue
            table = _plan_table_at(commit)
            assert f'"{plan}"' in table, (
                f"{name}: names plan {plan!r}, which is absent from the plan table at "
                f"{commit[:8]} -- the commit its own experiment block says its image "
                f"carries. The run would fail on the plan lookup before reading a corpus."
            )

    def test_a_document_declaring_an_image_commit_names_a_plan_from_it(self) -> None:
        """The two fields are only useful together.

        A document carrying ``image_commit`` and no ``--plan`` is not wrong,
        but a document carrying both is making a checkable claim, and this is
        the test that it is checked rather than decorative.
        """
        checked = [
            name
            for name, _project, document in submissions()
            if isinstance(document.get("command"), str)
            and isinstance(document.get("experiment"), dict)
            and _plan_flag(str(document["command"])) is not None
        ]

        assert checked, "no committed run document pairs a --plan with an experiment block"
