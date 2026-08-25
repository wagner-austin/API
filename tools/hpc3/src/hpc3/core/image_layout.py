"""Where a build's inputs live inside the image.

One owner for these paths. The definition copies files to them, the build
script writes files that land at them, and the self-check is executed from
one of them -- three renderers that must agree on the same strings. Defining
them separately in each is how a rename silently breaks one caller, so they
are defined once and imported.

These are container-side paths only. They are never host paths and never
bind-mounted: ``/opt`` is chosen precisely because the cluster does not mount
over it, unlike ``/pub`` and the other roots the image contract refuses.
"""

from __future__ import annotations

SPEC_DIR = "/opt/spec"
"""Directory holding the requirements file, self-check and commit stamp."""

WHEEL_DIR = "/opt/wheels"
"""Directory holding the first-party wheels during the build."""

REQUIREMENTS_NAME = "requirements.txt"
"""Rendered pip requirements file, in the build directory and in SPEC_DIR."""

SELFCHECK_NAME = "selfcheck.py"
"""Rendered verification script, run by ``%post`` after installation."""

COMMIT_NAME = "GIT_COMMIT"
"""Commit stamp, copied into the environment for the trainer to read."""

DEFINITION_NAME = "image.def"
"""Rendered Apptainer definition, named so the build script can find it."""

__all__ = [
    "COMMIT_NAME",
    "DEFINITION_NAME",
    "REQUIREMENTS_NAME",
    "SELFCHECK_NAME",
    "SPEC_DIR",
    "WHEEL_DIR",
]
