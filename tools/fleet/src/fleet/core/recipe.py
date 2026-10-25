"""What a node runs once a project's tree is staged.

Split out of :mod:`fleet.core.dispatch`, which owns the ORDER of a run, when
that file reached the 600-line ceiling (MCPs board task 649fcc7a): a recipe
is what the declaration asks the build to do, and :func:`recipe_for` is the
one place each declaration becomes it.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from fleet.contracts.project import ProjectConfig
from fleet.contracts.source import InstallStep
from fleet.contracts.tags import NodeTag
from fleet.core import names


class Recipe(TypedDict):
    """What the node runs once the tree is staged.

    Attributes:
        path: The project's directory inside the staged tree, ``""`` for the
            root; the recipe's working directory.
        install: The steps run at the tree's root before the recipe, each an
            argv in the source grammar with the phase it is timed as; empty
            when the recipe installs its own dependencies.
        isolated_docker: True when the project declares the ``docker`` tag,
            so its build runs as the node's execdocker user against that
            user's rootless daemon, never as the runner, whose own daemon on
            diphtheria is the production stack's
            (:mod:`fleet.core.linux_isolated_build`).
        elevated: True when the project declares the ``elevated`` tag, so its
            build launches at RunLevel Highest with the ssh account's full
            administrator token (:mod:`fleet.contracts.elevation`); only a
            node's elevated runner ever claims such a job.
        loopback_only: True when the project declares ``loopback_only``, so
            its ``make check`` runs in a network namespace holding only lo
            once its install steps have run in the host's
            (:mod:`fleet.contracts.check_unit`).
    """

    path: str
    install: tuple[InstallStep, ...]
    isolated_docker: bool
    elevated: bool
    loopback_only: bool


def recipe_for(plan: ProjectConfig, *, path: str, install: tuple[InstallStep, ...]) -> Recipe:
    """The recipe for one project, isolated when it declares the docker tag,
    elevated when it declares the elevated tag, and in a loopback-only
    network when it declares ``loopback_only``.

    The one place each declaration becomes what the build does, so the
    queue's lane and the working-tree lane cannot disagree about which builds
    run as execdocker, as an administrator or with only lo.

    Args:
        plan: The project's declaration, whose ``required_tags`` and
            ``loopback_only`` decide.
        path: The project's directory inside the staged tree.
        install: The steps run at the tree's root before the recipe.

    Returns:
        The recipe.
    """
    return Recipe(
        path=path,
        install=install,
        isolated_docker=NodeTag.DOCKER in plan["required_tags"],
        elevated=NodeTag.ELEVATED in plan["required_tags"],
        loopback_only=plan["loopback_only"],
    )


def working_tree_recipe(project: str, plan: ProjectConfig) -> Recipe:
    """The recipe a working-tree dispatch runs: ``make check`` in the
    project's own directory, whose key IS its repo-relative path, with no
    install steps because every poetry project here installs inside its
    recipe.

    Args:
        project: Repo-relative project path.
        plan: The project's declaration.

    Returns:
        The recipe.
    """
    return recipe_for(plan, path=project, install=())


def loopback_unit(built: Recipe, run_id: str) -> str | None:
    """The unit a loopback-only recipe's ``make check`` is bound to.

    Args:
        built: The recipe.
        run_id: The dispatch.

    Returns:
        The check's own unit, ``fleet-<run id>`` (:func:`fleet.core.names.
        task_name`, the name its launch starts it under), for a recipe whose
        project declares ``loopback_only``; None for every other.
    """
    return names.task_name(run_id) if built["loopback_only"] else None


__all__ = ["Recipe", "loopback_unit", "recipe_for", "working_tree_recipe"]
