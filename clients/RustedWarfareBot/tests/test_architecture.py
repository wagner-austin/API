"""Structural invariants, checked mechanically rather than by review.

The audit that produced this refactor found no competing planners. It found
layers that never ran: functions written, tested, documented with measured
evidence, and called by nothing; and two loops where the seam between them was
the defect. Neither shows up in a passing test suite, because each piece was
individually correct.

These are the checks that would have caught them. They read the source rather
than importing it, because what they assert is about *shape* -- which module
owns the socket, which names are reachable -- and shape is not observable from
inside a running program.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src" / "rw_bot"
_SCRIPTS = _PROJECT_ROOT / "scripts"
_TESTS = _PROJECT_ROOT / "tests"

#: The one module allowed to read from the agent.
#:
#: Named rather than derived: the point of the check is that this is a decision
#: somebody made, so changing it should mean editing this line and explaining
#: why in the same commit.
_LOOP_MODULE = "campaign.py"

#: The loop's sending arm: the only other module allowed to touch the
#: channel. The pair IS the loop; a third sender is the regression this
#: guard exists to catch ([[policy-loop]]).
_SENDING_MODULE = "dispatching.py"


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _module_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_exactly_one_module_reads_from_the_agent() -> None:
    """One loop, and the source can prove it.

    The bot used to have two: a build loop that ran the opening plan to
    completion and a fight loop that took over afterwards. While building there
    was no army and no economy; once fighting there was no build policy at all
    ([[policy-loop]]). Nothing in a passing suite objected, because both loops
    worked.
    """
    readers = [
        path.name
        for path in _python_files(_SRC)
        if "next_sample(" in _module_source(path) and "def next_sample" not in _module_source(path)
    ]
    assert readers == [_LOOP_MODULE]


def test_every_policy_module_but_the_loop_is_pure() -> None:
    """Decisions are values in and values out; only the loop touches the channel.

    A pure decision can be argued with in a test without a game running, which
    is what makes the playing logic reviewable at all.
    """
    impure = [
        path.name
        for path in _python_files(_SRC / "policy")
        if "AgentChannel" in _module_source(path)
    ]
    assert sorted(impure) == sorted([_LOOP_MODULE, _SENDING_MODULE])


def test_no_module_writes_orders_except_the_loop() -> None:
    """Dispatch has one owner, so an order cannot be sent from a decision.

    The owner is the loop's sending arm: the campaign reads and arbitrates,
    every order leaves through :mod:`rw_bot.policy.dispatching`, and a pure
    policy module can do neither.
    """
    senders = [
        path.name
        for path in _python_files(_SRC)
        if any(
            verb in _module_source(path)
            for verb in ("channel.send_build", "channel.send_produce", "channel.send_attack")
        )
    ]
    assert senders == [_SENDING_MODULE]


def _exported_functions(path: Path) -> list[str]:
    """Return the module-level functions a module lists in ``__all__``.

    Functions only. A TypedDict or a constant is reachable as a return type or
    through a docstring reference without any other module naming it, so
    demanding a mention would report the whole policy layer's vocabulary as
    dead. A function that nobody calls has no such excuse.

    Args:
        path: The module to read.

    Returns:
        The exported function names, empty when there are none.
    """
    tree = ast.parse(_module_source(path), filename=str(path))
    defined = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    exported: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "__all__" not in targets or not isinstance(node.value, ast.List):
            continue
        # Narrowed one element at a time: a literal's value is untyped, and this
        # package does not permit widening it back with a cast.
        exported = []
        for element in node.value.elts:
            if not isinstance(element, ast.Constant):
                continue
            literal = element.value
            if isinstance(literal, str):
                exported.append(literal)
    return [name for name in exported if name in defined]


def test_no_exported_policy_name_is_unreachable() -> None:
    """A decision function nobody calls is a layer that does not run.

    ``production_bound`` was written, tested, and documented with the measured
    evidence for it -- a run that banked 7,013 credits behind a single factory --
    and then called by nothing at all. It was even left out of its own
    ``__all__``. The suite was green throughout ([[policy-production]]).

    Reachability is judged over the whole tree including tests: a name only its
    own unit test mentions is exactly the case this exists to catch, so the
    defining module is excluded from the search and its test is not.
    """
    corpus = {
        path: _module_source(path)
        for root in (_SRC, _SCRIPTS, _TESTS)
        for path in _python_files(root)
    }
    unreachable: list[str] = []
    for path in _python_files(_SRC / "policy") + _python_files(_SRC / "mechanics"):
        for name in _exported_functions(path):
            mentions = sum(text.count(name) for other, text in corpus.items() if other != path)
            if mentions == 0:
                unreachable.append(f"{path.name}:{name}")
    assert unreachable == []


def _module_name(path: Path) -> str:
    """Return a file's dotted module name as an import would spell it.

    Args:
        path: A python file under ``src`` or ``scripts``.

    Returns:
        The dotted name; a package's ``__init__`` answers to the package.
    """
    base = _SRC.parent if _SRC.parent in path.parents else _PROJECT_ROOT
    parts = list(path.relative_to(base).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_modules(path: Path) -> set[str]:
    """Return every dotted name a module's import statements can reach.

    A ``from a.b import c`` line contributes both ``a.b`` and ``a.b.c``,
    because ``c`` may be a submodule or a symbol and only the caller's
    module table can tell which.

    Args:
        path: The module to read.

    Returns:
        The imported names, unfiltered.
    """
    tree = ast.parse(_module_source(path), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


#: Modules pyproject exposes as console scripts, which makes them entry
#: points exactly as ``scripts/play.py`` is. Named rather than parsed out of
#: pyproject, for the same reason :data:`_LOOP_MODULE` is: an entry point is
#: a decision somebody made, so adding one should mean editing this line in
#: the same commit. A drift is loud either way -- a script added here but
#: not to pyproject is dead and this check says so; one added to pyproject
#: but not here fails the same way until the decision is recorded.
_CONSOLE_SCRIPT_MODULES = ("rw_bot.harness.boot_log_cli", "rw_bot.harness.fleet_http")


def test_every_production_module_is_wired_to_an_entry_point() -> None:
    """A module production never imports is a layer that cannot run.

    The audit that produced this file found functions written, tested and
    documented, and called by nothing -- and the reachability check above
    still has the hole that hid them: it counts a mention anywhere,
    including tests, so a module wired to nothing but its own suite reads
    as alive. This one walks the import graph from the ``scripts`` entry
    points -- the surface ``make play``, ``make sweep`` and the analysis
    tools actually execute -- and refuses any ``rw_bot`` module the walk
    never reaches. Shipping a new layer therefore *requires* wiring it:
    the build fails on a module that exists only for its tests.

    Importing ``a.b.c`` imports ``a`` and ``a.b`` on the way, so a reached
    module marks its ancestor packages reached too.
    """
    modules = {
        _module_name(path): path for root in (_SRC, _SCRIPTS) for path in _python_files(root)
    }
    reached = {name for name in modules if name.split(".")[0] == "scripts"}
    reached.update(name for name in _CONSOLE_SCRIPT_MODULES if name in modules)
    frontier = list(reached)
    while frontier:
        for imported in _imported_modules(modules[frontier.pop()]):
            if imported not in modules or imported in reached:
                continue
            reached.add(imported)
            frontier.append(imported)
            parts = imported.split(".")
            reached.update(
                ancestor
                for ancestor in (".".join(parts[:length]) for length in range(1, len(parts)))
                if ancestor in modules
            )
    unwired = sorted(
        path.relative_to(_PROJECT_ROOT).as_posix()
        for name, path in modules.items()
        if name not in reached and name.split(".")[0] == "rw_bot"
    )
    assert unwired == []


def test_every_decoder_has_an_encoder() -> None:
    """A fact that can be read and not written cannot be re-emitted as a fixture.

    The round trip is what lets a captured dump become a test corpus, and what
    keeps a hand-written fixture from drifting away from the shape the agent
    actually writes ([[wire-contract-ndjson]]).
    """
    missing: list[str] = []
    for path in _python_files(_SRC / "mechanics") + _python_files(_SRC / "wire"):
        text = _module_source(path)
        # Record decoders only. The stat catalogue parses the engine's
        # ``-printunits`` log, which is prose with a shape rather than a record
        # format, and re-emitting it would be writing a fixture generator for a
        # file the engine alone produces.
        if "parse_object" not in text and "records_of_kind" not in text:
            continue
        if "def decode_" in text and "def encode_" not in text:
            missing.append(path.name)
    assert missing == []


def test_no_suppression_appears_anywhere() -> None:
    """Strictness is the contract, and a suppression is a silent exemption."""
    # Assembled rather than written out, so this file does not report itself.
    banned = ("type: " + "ignore", "no" + "qa", "pragma: " + "no cover")
    offenders = [
        f"{path.name}:{token}"
        for root in (_SRC, _SCRIPTS, _TESTS)
        for path in _python_files(root)
        for token in banned
        if token in _module_source(path)
    ]
    assert offenders == []


def test_the_wire_declares_its_child_counts_in_one_place() -> None:
    """Two readers need the total and must not disagree about it.

    The channel used to add up the declared counts itself, so a new record kind
    left it reading a sample one record short -- and the decoder then rejected
    that sample as truncated, which is a true complaint about the wrong thing.
    """
    channel = _module_source(_SRC / "control" / "channel.py")
    assert "declared_children" in channel
    for field in ("visible", "pools", "options", "players"):
        assert f'"{field}"' not in channel


#: The most lines a module may hold before it is doing more than one job.
#:
#: Not a style preference. Every split this ceiling has forced was a module that
#: had quietly grown a second concern: ``build_order`` was deciding what to make
#: *and* where it would stand, ``economy`` was claiming income *and* covering
#: what it claimed, ``state`` was declaring the wire's vocabulary *and* parsing
#: it. Each of those pairs is asked by different callers for different reasons,
#: and each split let one side be read without the other.
#:
#: Set at the top of the 400-600 band the project works to, so a module has
#: room to carry the evidence for its own decisions -- which is most of what the
#: docstrings here are -- before the ceiling starts arguing with them.
_MAX_MODULE_LINES = 600


_AGENT_SRC = _PROJECT_ROOT / "agent" / "src" / "rwbot" / "agent"


def test_no_module_has_grown_a_second_job() -> None:
    """A module past the ceiling is doing more than one thing.

    Checked mechanically because the failure is gradual: nobody writes a
    900-line module, they add forty lines to an already-large one and every
    individual step looks reasonable. The number is arbitrary; noticing is not.

    Everywhere code is written, and deliberately not only ``src``. For a week
    this ran over ``src`` and ``scripts`` alone while five test modules and
    three agent classes stood over the line, the worst at 1,170 -- the cap a
    file nobody checks is no cap. Those eight were named in a shrink-only
    backlog for exactly one commit and then split, so the list is gone: there
    is no longer any file this rule is allowed to be lenient about, and adding
    one back would be a decision somebody has to argue for in review rather
    than a line quietly appended to a set.
    """
    watched = [path for root in (_SRC, _SCRIPTS, _TESTS) for path in _python_files(root)] + sorted(
        _AGENT_SRC.glob("*.java")
    )
    oversized = [
        f"{path.relative_to(_PROJECT_ROOT).as_posix()}:{len(_module_source(path).splitlines())}"
        for path in watched
        if len(_module_source(path).splitlines()) > _MAX_MODULE_LINES
    ]
    assert oversized == []
