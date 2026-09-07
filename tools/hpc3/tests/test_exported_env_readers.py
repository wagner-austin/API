"""Whether the variables this wrapper exports are declared, and read.

Two properties, and they fail for different reasons.

DECLARATION IS PINNED IN BOTH DIRECTIONS. ``EXPORTED_VARIABLES`` is the
launcher-to-payload interface. A rendered script exporting a name absent from
it, or a name in it that no rendered script exports, is a failure here rather
than a discovery months later. The comparison is against RENDERED OUTPUT, not
source, so an export emitted by a helper -- conditionally, or from another
module -- cannot slip past by being written somewhere other than the list.

READERS ARE ASSERTED, NOT ASSUMED. An exported variable nothing reads is
inert: present in the process, asserted by a render test, and doing nothing.
That is a latent bug in a launcher and an active one when a guard credits the
declaration -- ``PREEMPTIBLE_RUN_UNPROTECTED`` admits a long preemptible run
that declares ``checkpoint_steps``, on the reasoning that such a run survives
eviction, and the number reaches the payload as ``HPC3_CHECKPOINT_STEPS``.

The scan and the predicates live in
:mod:`platform_core.exported_env_audit` for the reason
:mod:`platform_core.entrypoint_audit` gives: two copies of a rule is how one
gets fixed and the other does not. This module supplies the roots and asserts.
"""

from __future__ import annotations

import pathlib

from platform_core.exported_env_audit import (
    ExportedVariable,
    audit_variables,
    describe_inert,
    inert_variables,
    originated_names,
)

from hpc3.contracts.job import JobSpec
from hpc3.core.sbatch import EXPORTED_VARIABLES, render_sbatch
from tests._sbatch_support import IMAGE, LOG_DIR, spec

#: The monorepo root, from this file's own location.
MONOREPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: The trees holding first-party packages a job's payload can come from.
PACKAGE_TREES: tuple[str, ...] = ("libs", "services", "clients", "tools")

#: An environment inside an image cannot sit under a bind-mounted host root,
#: so the image postures declare the image's own prefix.
IMAGE_ENV_PATH = "/opt/env"

#: This package, which is the LAUNCHER and never a payload. Excluded from the
#: search by role, not by convenience: ``EXPORTED_VARIABLES`` names every
#: variable as a string literal, so searching the launcher would let the
#: declaration satisfy itself and the audit would pass by construction --
#: reporting that a variable is read when the only thing naming it is the code
#: that exports it.
LAUNCHER_PACKAGE = "hpc3"


def payload_roots() -> tuple[pathlib.Path, ...]:
    """Every first-party package source directory in the monorepo.

    Scoped to ``src`` because that is this repository's layout for code --
    every package declares ``packages = [{include = ..., from = "src"}]`` --
    and because the alternative, walking whole package trees, reads generated
    run output as though it were a payload. It also is not readable: a run
    directory under ``tools/code-style-eval/runs`` refuses to open, and a
    reader that had to survive that would be swallowing an error to decide a
    variable is unread, which is the opposite of what this asserts.

    Returns:
        The ``src`` directories that exist, sorted.
    """
    return tuple(
        sorted(
            package / "src"
            for tree in PACKAGE_TREES
            for package in (MONOREPO_ROOT / tree).iterdir()
            if (package / "src").is_dir() and package.name != LAUNCHER_PACKAGE
        )
    )


def _render(job: JobSpec) -> str:
    """Render a job with no charge account, which is the free posture.

    Args:
        job: The spec to render.

    Returns:
        The batch script text.
    """
    return render_sbatch(job, log_dir=LOG_DIR, charge_account="")


def _every_rendered_name() -> frozenset[str]:
    """Every variable name any rendering of any job originates.

    Renders the postures that differ in what they export -- host against
    image, deterministic against not -- because a conditional export is
    exactly the kind that drifts from a declaration unnoticed.

    Returns:
        The union of originated names across those renderings.
    """
    postures = (
        spec(),
        spec(deterministic=True),
        spec(image=IMAGE, env_path=IMAGE_ENV_PATH),
        spec(image=IMAGE, env_path=IMAGE_ENV_PATH, deterministic=True),
    )
    names: set[str] = set()
    for job in postures:
        names.update(originated_names(_render(job)))
    return frozenset(names)


class TestDeclarationMatchesRender:
    def test_every_rendered_export_is_declared(self) -> None:
        declared = {variable["name"] for variable in EXPORTED_VARIABLES}
        undeclared = sorted(_every_rendered_name() - declared)
        assert undeclared == [], (
            f"rendered scripts export {undeclared}, which EXPORTED_VARIABLES does not "
            "declare; add them there with the behaviour the payload is expected to "
            "implement"
        )

    def test_every_declared_export_is_rendered(self) -> None:
        declared = {variable["name"] for variable in EXPORTED_VARIABLES}
        unrendered = sorted(declared - _every_rendered_name())
        assert unrendered == [], (
            f"EXPORTED_VARIABLES declares {unrendered}, which no rendering emits; "
            "a declared interface nothing sends is the same defect as an export "
            "nothing reads"
        )


class TestExportsAreRead:
    def test_no_exported_variable_is_inert(self) -> None:
        audits = audit_variables(EXPORTED_VARIABLES, payload_roots())
        assert inert_variables(audits) == (), describe_inert(audits)


class TestPurposesAreStated:
    def test_every_declared_variable_states_a_purpose(self) -> None:
        missing = sorted(
            variable["name"] for variable in EXPORTED_VARIABLES if not variable["purpose"].strip()
        )
        assert missing == [], (
            f"{missing} declare no purpose; an inert export is reported with the "
            "behaviour that is missing, which an empty purpose cannot say"
        )

    def test_declared_variables_are_unique(self) -> None:
        names = [variable["name"] for variable in EXPORTED_VARIABLES]
        assert sorted(names) == sorted(set(names))

    def test_every_declaration_carries_exactly_the_records_fields(self) -> None:
        """A declaration missing a field would report an inert variable
        without the behaviour that is absent, which is the half that makes
        the failure actionable."""
        expected = set(ExportedVariable.__annotations__)
        for variable in EXPORTED_VARIABLES:
            assert set(variable) == expected, variable["name"]

    def test_names_are_the_shell_form_the_script_emits(self) -> None:
        """A declared name that is not a legal shell identifier could never
        match a rendered export, so the bidirectional check would fail on it
        for a reason that reads as drift rather than as a typo."""
        for variable in EXPORTED_VARIABLES:
            name = variable["name"]
            assert name.isupper()
            assert name.replace("_", "").isalnum()
