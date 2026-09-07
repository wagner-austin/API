"""Whether a variable a launcher exports into a job is read by anything.

THE DEFECT THIS FINDS. A launcher that exports ``FOO=1`` into a job's
environment has done something visible, testable and inert: the variable is
in the process, a test asserting the export line passes, and no payload ever
looks at it. The declaration and the behaviour are in different repositories
and nothing compares them, so the gap is invisible from either side.

That is worse than an unused constant when a guard reads the declaration.
:mod:`hpc3.contracts.job` admits a long preemptible run when it declares a
positive ``checkpoint_steps``, on the reasoning that such a run survives
eviction. The number reaches the job as ``HPC3_CHECKPOINT_STEPS`` and no
payload in this monorepo reads it, so the clause is satisfied by a value
whose only effect is satisfying the clause.

MEASURED COST ON THIS MONOREPO, 2026-09-04. Three of the six variables the
hpc3 sbatch wrapper exports had no reader anywhere: ``HPC3_PROJECT``,
``HPC3_CHECKPOINT_STEPS`` and ``HPC3_RESTART_COUNT``. The last two compound:
a payload that cannot tell it is a resumed run could not act on a checkpoint
cadence even if it read one. ``mi`` and ``turkic-lstm`` both declared a
cadence -- 500 and 27,344 steps -- against payloads that checkpoint on a
schedule of their own and have never seen either number.

WHY A READER IS A STRING LITERAL IN CODE. A variable named in a comment, a
docstring or a Markdown page is documentation, not a reader, and counting it
would let the very prose that describes the intended wiring stand in for the
wiring. The parse here is over string constants only, for the same reason
:mod:`platform_core.entrypoint_audit` parses instead of searching: a mention
is not a use.

WHAT THIS CANNOT SEE, stated because the limit is load-bearing. It reads the
roots it is given. A payload living outside them -- ``turkic-lstm`` runs from
a separate repository -- is not searched, so an empty reader set means "no
first-party reader in the roots supplied", never "no reader exists". Callers
name their payload roots positively rather than excusing files, so that
widening the search is an argument change and not an exemption.
"""

from __future__ import annotations

import ast
import pathlib

from typing_extensions import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

#: Directory names holding code this repository did not write. A vendored
#: dependency that happens to name a variable is not a payload honouring it,
#: and a virtualenv contains a copy of every first-party package anyway --
#: counting those would let a package satisfy the audit by being installed.
VENDORED_DIRECTORIES: frozenset[str] = frozenset(
    {".venv", "site-packages", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules"}
)


class ExportedVariable(TypedDict):
    """One environment variable a launcher exports into a job's process.

    Attributes:
        name: The variable name as exported, e.g. ``HPC3_RESTART_COUNT``.
        purpose: What the launcher intends the payload to do with it. Carried
            so an inert variable is reported with the behaviour that is
            missing rather than only with its name.
    """

    name: str
    purpose: str


class ReaderSite(TypedDict):
    """One place in a payload tree that names an exported variable in code.

    Attributes:
        path: Path to the module, as given by the search root.
        line: 1-based line of the string literal naming the variable.
    """

    path: str
    line: int


class VariableAudit(TypedDict):
    """One exported variable together with every site that reads it.

    Attributes:
        variable: The exported variable.
        readers: Every site naming it, ordered by path then line. Empty means
            no first-party reader in the roots searched.
    """

    variable: ExportedVariable
    readers: tuple[ReaderSite, ...]


def encode_exported_variable(variable: ExportedVariable) -> JSONObject:
    """Encode an exported variable for storage or reporting.

    Args:
        variable: The variable to encode.

    Returns:
        Its JSON object form.
    """
    return {"name": variable["name"], "purpose": variable["purpose"]}


def decode_exported_variable(value: JSONValue) -> ExportedVariable:
    """Decode an exported variable.

    Args:
        value: The JSON value to decode.

    Returns:
        The decoded variable.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing
            or of the wrong type.
    """
    obj = narrow_json_to_dict(value)
    return ExportedVariable(name=require_str(obj, "name"), purpose=require_str(obj, "purpose"))


def encode_reader_site(site: ReaderSite) -> JSONObject:
    """Encode a reader site for storage or reporting.

    Args:
        site: The site to encode.

    Returns:
        Its JSON object form.
    """
    return {"path": site["path"], "line": site["line"]}


def decode_reader_site(value: JSONValue) -> ReaderSite:
    """Decode a reader site.

    Args:
        value: The JSON value to decode.

    Returns:
        The decoded site.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing
            or of the wrong type.
    """
    obj = narrow_json_to_dict(value)
    return ReaderSite(path=require_str(obj, "path"), line=require_int(obj, "line"))


def encode_variable_audit(audit: VariableAudit) -> JSONObject:
    """Encode one variable's audit for storage or reporting.

    Args:
        audit: The audit to encode.

    Returns:
        Its JSON object form.
    """
    return {
        "variable": encode_exported_variable(audit["variable"]),
        "readers": [encode_reader_site(site) for site in audit["readers"]],
    }


def decode_variable_audit(value: JSONValue) -> VariableAudit:
    """Decode one variable's audit.

    Args:
        value: The JSON value to decode.

    Returns:
        The decoded audit.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing
            or of the wrong type.
    """
    obj = narrow_json_to_dict(value)
    readers = require_list(obj, "readers")
    return VariableAudit(
        variable=decode_exported_variable(obj["variable"]),
        readers=tuple(decode_reader_site(entry) for entry in readers),
    )


def is_vendored(path: pathlib.Path) -> bool:
    """Whether a path lies inside code this repository did not write.

    Args:
        path: The path to test.

    Returns:
        True when any component names a vendored directory.
    """
    return any(part in VENDORED_DIRECTORIES for part in path.parts)


def payload_modules(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Every first-party Python module under a payload root.

    Args:
        root: The directory to search.

    Returns:
        The module paths, sorted, excluding vendored trees.
    """
    return tuple(sorted(p for p in root.rglob("*.py") if not is_vendored(p)))


def string_literals(source: str) -> tuple[tuple[str, int], ...]:
    """Every string constant in a module, with the line it sits on.

    Comments and docstring PROSE are excluded by construction: a comment is
    not in the tree at all, and a docstring is one literal whose whole text
    must equal the variable name to count -- which no docstring does.

    Args:
        source: The module source.

    Returns:
        Pairs of literal value and 1-based line number, in traversal order.

    Raises:
        SyntaxError: If the source does not parse.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append((node.value, node.lineno))
    return tuple(found)


#: The shell keyword a launcher script uses to put a variable in the payload's
#: environment.
EXPORT_KEYWORD = "export "


def shell_exported_names(script: str) -> tuple[str, ...]:
    """Every variable name a rendered launcher script exports.

    Reads the launcher's OWN generated output, not source, which is what
    makes the pairing with a declared list total: a line emitted by any
    helper is still an ``export`` line in the finished script, so a
    conditional or delegated export cannot slip past a declaration check by
    being written somewhere else.

    Args:
        script: The rendered script text.

    Returns:
        The exported names, sorted and deduplicated.
    """
    return tuple(sorted({name for name, _ in shell_export_assignments(script)}))


def shell_export_assignments(script: str) -> tuple[tuple[str, str], ...]:
    """Every ``export`` in a rendered launcher script, as name and raw value.

    Args:
        script: The rendered script text.

    Returns:
        Name and value pairs in script order, values with surrounding double
        quotes stripped. Lines exporting a bare name carry an empty value.
    """
    assignments: list[tuple[str, str]] = []
    for raw in script.splitlines():
        line = raw.strip()
        if not line.startswith(EXPORT_KEYWORD):
            continue
        name, separator, value = line[len(EXPORT_KEYWORD) :].partition("=")
        if separator:
            assignments.append((name.strip(), value.strip().strip('"')))
    return tuple(assignments)


def modifies_existing(name: str, value: str) -> bool:
    """Whether an export extends the variable it names rather than creating it.

    ``export PATH="/opt/env/bin:$PATH"`` prepends to a variable the operating
    system already set; ``export HPC3_PROJECT="mi"`` originates one. Only the
    second is a message from launcher to payload, so only the second can be
    inert. The test is self-reference, which is structural -- a name is not
    excused by appearing on a list somebody maintains, it falls out of scope
    by the shape of its own assignment.

    Args:
        name: The exported name.
        value: Its raw value.

    Returns:
        True when the value expands the same variable it assigns.
    """
    return f"${name}" in value or f"${{{name}" in value


def originated_names(script: str) -> tuple[str, ...]:
    """Every variable a launcher script CREATES for its payload to read.

    Args:
        script: The rendered script text.

    Returns:
        The originated names, sorted and deduplicated, excluding exports that
        merely extend a pre-existing variable.
    """
    return tuple(
        sorted(
            {
                name
                for name, value in shell_export_assignments(script)
                if not modifies_existing(name, value)
            }
        )
    )


def site_order(site: ReaderSite) -> tuple[str, int]:
    """Sort key placing reader sites in path then line order.

    A named function rather than a lambda: a lambda's parameter is untyped,
    and an untyped parameter in a strict package is an ``Any`` expression.

    Args:
        site: The site to order.

    Returns:
        Its path and line, in that precedence.
    """
    return (site["path"], site["line"])


def reader_sites(name: str, roots: tuple[pathlib.Path, ...]) -> tuple[ReaderSite, ...]:
    """Every site under the given roots that names a variable in code.

    Args:
        name: The exported variable name.
        roots: Payload trees to search.

    Returns:
        The sites, ordered by path then line.

    Raises:
        SyntaxError: If a module under a root does not parse.
    """
    sites: list[ReaderSite] = []
    for root in roots:
        for module in payload_modules(root):
            source = module.read_text(encoding="utf-8")
            if name not in source:
                continue
            sites.extend(
                ReaderSite(path=module.as_posix(), line=line)
                for literal, line in string_literals(source)
                if literal == name
            )
    return tuple(sorted(sites, key=site_order))


def audit_variables(
    variables: tuple[ExportedVariable, ...], roots: tuple[pathlib.Path, ...]
) -> tuple[VariableAudit, ...]:
    """Audit every exported variable against the payload trees that may read it.

    Args:
        variables: The variables a launcher exports.
        roots: Payload trees to search.

    Returns:
        One audit per variable, in the order given.

    Raises:
        SyntaxError: If a module under a root does not parse.
    """
    return tuple(
        VariableAudit(variable=variable, readers=reader_sites(variable["name"], roots))
        for variable in variables
    )


def inert_variables(audits: tuple[VariableAudit, ...]) -> tuple[str, ...]:
    """The exported variables that no searched payload reads.

    Args:
        audits: The audits to filter.

    Returns:
        The names with no reader, in the order audited.
    """
    return tuple(audit["variable"]["name"] for audit in audits if not audit["readers"])


def describe_inert(audits: tuple[VariableAudit, ...]) -> str:
    """Render the inert variables as a failure message naming what is missing.

    Args:
        audits: The audits to describe.

    Returns:
        One line per inert variable, giving its name and its stated purpose,
        so the reader sees the behaviour that is absent rather than only the
        name that is unused. Empty string when every variable has a reader.
    """
    return "\n".join(
        f"{audit['variable']['name']}: exported, read by nothing. "
        f"Intended purpose: {audit['variable']['purpose']}"
        for audit in audits
        if not audit["readers"]
    )


__all__ = [
    "EXPORT_KEYWORD",
    "VENDORED_DIRECTORIES",
    "ExportedVariable",
    "ReaderSite",
    "VariableAudit",
    "audit_variables",
    "decode_exported_variable",
    "decode_reader_site",
    "decode_variable_audit",
    "describe_inert",
    "encode_exported_variable",
    "encode_reader_site",
    "encode_variable_audit",
    "inert_variables",
    "is_vendored",
    "modifies_existing",
    "originated_names",
    "payload_modules",
    "reader_sites",
    "shell_export_assignments",
    "shell_exported_names",
    "site_order",
    "string_literals",
]
