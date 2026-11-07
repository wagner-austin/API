"""The mutation step an API package's ``make test`` runs after its suite (MCPs task e52e5148).

The API half of MCPs' Python step (``mcp-shared-py/src/mcp_shared_py/mutation``,
MCPs board task fec5851c): the same tool (cosmic-ray), the same scope (the
src lines a commit adds against ``origin/main``), the same refusal codes
(:class:`platform_core.error_codes_tooling.MutationErrorCode`) and the same
lines a reader sees. :mod:`.changes` and :mod:`.sites` are MCPs' own files,
lifted byte for byte and pinned in ``tools/maketools/lift-lock.json``; the
modules that run commands and read JSON are written here, on this
repository's one JSON reader and one error type, which MCPs' cannot import.
"""
