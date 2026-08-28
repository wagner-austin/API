"""Shared platform primitives, imported from the module that defines them.

THIS FILE IS DELIBERATELY EMPTY OF RE-EXPORTS, AND THAT IS A CHANGE. It used
to import fifty-nine names from eleven modules -- the data-bank client, the
OAuth flow, the FastAPI exception handlers, the security dependency -- so that
``from platform_core import AppError`` worked. The cost was that importing
ANYTHING from this package ran all of it.

WHAT THAT COST, MEASURED. ``import hpc3.contracts.workspace`` -- a contract in
a command-line tool that talks to Slurm over SSH, touching no network and
serving no requests -- took 368 ms and loaded 466 modules, among them the
whole of FastAPI, Starlette, httpx and both major versions of Pydantic. Every
``hpc3-watch`` invocation paid it.

The deeper cost is the one this was changed for. ``platform_core.run_record``
and ``platform_core.comparability`` are the shape every research run is meant
to emit, and research code that wants them -- a training repo, a benchmark
harness in another repository -- had to install a web framework to record a
number. That is a good enough reason for such code to keep hand-rolling its
own provenance format instead, which is exactly what happened: see
``docs/RESEARCH.md`` for the surfaces that did.

WHAT THIS DOES NOT BREAK. Submodule imports -- ``from platform_core import
cli_args``, ``json_utils``, ``job_keys`` -- are unaffected; Python resolves
those against the package directory, not against this file. Thirty-two of the
thirty-four bare-package imports in this monorepo were of that kind. The two
that were not named ``DataBankClient``, and now say
``from platform_core.data_bank_client import DataBankClient``, which is what
every other consumer of every other module here already said.

So: import from the module. ``from platform_core.errors import AppError``,
``from platform_core.run_record import RunRecord``. It is one line longer and
it says where the thing lives.
"""

from __future__ import annotations

__all__: list[str] = []
