"""Pin one version of the shared test and lint toolchain across every package.

WHY. Forty-two packages in this monorepo declare the same six or seven dev
tools, and on 2026-09-04 they had resolved to FIVE different mypy versions
(1.17.0 through 1.20.2), EIGHT different ruff versions (0.14.4 through
0.14.14) and four different pytest versions. Nothing chose that spread; it is
what happens when forty-two lock files are refreshed on forty-two different
days.

The spread is not harmless. A guard or a type error that fires on one
package's mypy and not on another's makes "the gate passed" mean different
things in different directories, and the whole repository is built on that
sentence meaning one thing.

WHY NOTHING HAD REACHED mypy 2.x. Every package declared ``mypy = "^1.13.0"``.
A caret on a 1.x release pins the MAJOR, so `poetry update` could never cross
into 2.x however often it ran -- the constraint, not the resolver, was holding
the whole repository a major version behind. ``ruff = "^0.14.4"`` has the same
shape one level down: for a 0.x release a caret pins the MINOR, so ruff was
locked inside 0.14 while 0.16 shipped.

This script only rewrites the constraints it is given, only inside the dev
dependency group, and only where the tool is already declared. It does not add
a tool to a package that never asked for one, and it does not touch runtime
dependencies.
"""

from __future__ import annotations

import pathlib
import re
import sys

#: The shared toolchain, and the single version every package should ask for.
#:
#: Read from PyPI on 2026-09-04 rather than assumed. Carets are kept rather
#: than exact pins because the lock file is what actually fixes the version --
#: the constraint's job is to say which major line a package is on, which is
#: precisely the thing that was wrong before.
STANDARD = {
    "mypy": "^2.3.1",
    "ruff": "^0.16.6",
    "pytest": "^9.1.1",
    "pytest-cov": "^7.1.0",
    "pytest-xdist": "^3.8.0",
    "pytest-asyncio": "^1.4.0",
    "pytest-timeout": "^2.4.0",
}

DEV_GROUP = re.compile(r"(\[tool\.poetry\.group\.dev\.dependencies\])(.*?)(?=\n\[|\Z)", re.S)


def restandardise(text: str) -> tuple[str, list[str]]:
    """Rewrite a pyproject's dev-group constraints to the standard set.

    Args:
        text: The pyproject.toml contents.

    Returns:
        The rewritten contents and the tools whose constraint changed. A tool
        already at the standard constraint is not reported, so the caller can
        tell a real edit from a no-op.
    """
    match = DEV_GROUP.search(text)
    if match is None:
        return text, []

    body = match.group(2)
    changed: list[str] = []

    def replace(line_match: re.Match[str]) -> str:
        name, current = line_match.group(1), line_match.group(2).strip()
        wanted = STANDARD.get(name)
        # A table-valued constraint (extras, markers) is left alone: rewriting
        # it to a bare string would silently drop whatever it carries.
        if wanted is None or current.startswith("{") or current == f'"{wanted}"':
            return line_match.group(0)
        changed.append(f"{name} {current} -> \"{wanted}\"")
        return f'{name} = "{wanted}"'

    new_body = re.sub(r"^([A-Za-z0-9_.-]+)\s*=\s*(.+)$", replace, body, flags=re.M)
    return text[: match.start(2)] + new_body + text[match.end(2) :], changed


def main(argv: list[str]) -> int:
    """Rewrite every package's dev-group constraints.

    Args:
        argv: Package directories to rewrite, repo-relative. Rewrites every
            package carrying a pyproject.toml when none are given.

    Returns:
        Exit code 0.
    """
    repo = pathlib.Path(__file__).resolve().parent.parent
    if argv:
        targets = [repo / name for name in argv]
    else:
        targets = [
            manifest.parent
            for root in ("clients", "services", "libs", "tools")
            for manifest in sorted((repo / root).glob("*/pyproject.toml"))
        ]

    touched = 0
    for package in targets:
        manifest = package / "pyproject.toml"
        if not manifest.is_file():
            continue
        original = manifest.read_text(encoding="utf-8")
        rewritten, changed = restandardise(original)
        if changed:
            manifest.write_text(rewritten, encoding="utf-8")
            touched += 1
            sys.stdout.write(f"{package.relative_to(repo).as_posix()}\n")
            for entry in changed:
                sys.stdout.write(f"    {entry}\n")
    sys.stdout.write(f"\n{touched} package(s) rewritten\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
