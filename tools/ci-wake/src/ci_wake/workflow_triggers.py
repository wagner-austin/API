"""Which pushed refs start a workflow, read from the workflows themselves.

THE FAILURE (MCPs board task 24547f50). On 2026-10-11 a session pushed the
tag ``archive/dx-mod`` to MCPs, whose two workflows run on pushes to
``main`` and nothing else. The hook enrolled it, and fifteen minutes later
this bridge told the session NO RUN EVER APPEARED, offering three possible
causes for a push that could never have had a run. The answer was in the
repository the whole time: a run exists for a push only if some workflow's
push trigger admits the ref it updates, and the workflows are files in the
pushed commit. So the enrolment reads them, and a push no trigger admits is
not enrolled, with the workflows it read named.

NOT "NEVER ENROL A TAG". API's ``packages.yml`` declares a bare ``push:``,
which GitHub runs for every branch AND every tag, so a tag rule would be
right in one repository and wrong in the other. Only the workflow decides.

WHAT GITHUB DOES, as its workflow syntax reference states it for
``on.push.<branches|branches-ignore|tags|tags-ignore>``:

- a push trigger with no branch or tag filter runs for every branch and
  every tag;
- one with branch filters only runs for no tag, and one with tag filters
  only runs for no branch;
- a positive filter admits a ref matching any of its patterns, an ignore
  filter a ref matching none, and naming both for one kind is an invalid
  workflow;
- a push event exists for branches and tags only.

WHAT IS NOT JUDGED: ``paths`` and ``paths-ignore`` on a branch push.
GitHub decides those against the remote's previous tip, or for a new
branch against a base this hook cannot know, so a branch push some trigger
admits is enrolled, and the bridge's NO RUN EVER APPEARED still names path
filters as one cause. GitHub does not evaluate path filters for tags, so a
tag is judged in full.

LINE-WISE, NOT A YAML PARSER, for the reason ``monorepo_guards``'
``workflow_timeout_rules`` gives: the shape CI already constrains needs a
few patterns, and a new dependency here would have to reach the virtualenv
every ``pre-push`` in two repositories runs before the code importing it
did, or every push would fail. The reader is STRICT instead of lenient: a
shape it does not read is refused as ``WORKFLOW_TRIGGER_UNREAD`` naming the
file and line, because a reader that skipped what it could not parse would
answer "no workflow runs" for exactly the workflow it never read. Its suite
reads every workflow in this repository and MCPs's two ``on:`` blocks, so a
workflow written in a new shape fails here before it fails a push.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final, NamedTuple

from platform_core.error_codes_tooling import CiWakeErrorCode
from platform_core.errors import AppError

#: ``on:`` at column zero, bare or quoted (``"on":`` keeps YAML 1.1 readers
#: from taking the key as the boolean true; API's workflows use both).
_ON_HEADER: Final = re.compile(r"""^(?:on|"on"|'on'):(.*)$""")

#: A mapping key below it, its indent captured rather than assumed.
_KEY: Final = re.compile(r"^( +)([A-Za-z_][A-Za-z0-9_-]*):(.*)$")

#: A block sequence item.
_ITEM: Final = re.compile(r"^( +)- (.*)$")

#: An unquoted pattern or event name. A leading ``*`` would be a YAML alias,
#: so a pattern opening with one must be quoted, as GitHub's own examples are.
_BARE: Final = re.compile(r"^[A-Za-z0-9_./-][A-Za-z0-9_./*-]*$")

#: The filters GitHub defines for ``on.push``, by the field each fills.
_FILTERS: Final[tuple[str, ...]] = (
    "branches",
    "branches-ignore",
    "tags",
    "tags-ignore",
    "paths",
    "paths-ignore",
)

#: GitHub's filter operators beyond ``*`` and ``**``: ``?`` and ``+``
#: quantify the character before them and ``[`` opens a class. None is used
#: by any workflow read here, so each is refused by name rather than guessed.
_UNREAD_OPERATORS: Final[tuple[str, ...]] = ("?", "+", "[", "]")

#: A pattern's pieces: ``**``, ``*``, or a run of literal characters.
_PATTERN_TOKEN: Final = re.compile(r"\*\*|\*|[^*]+")

BRANCH_PREFIX: Final = "refs/heads/"
TAG_PREFIX: Final = "refs/tags/"


class PushTrigger(NamedTuple):
    """One workflow's ``on.push`` ref filters; None for a filter it omits.

    Attributes:
        branches: Patterns a branch must match one of.
        branches_ignore: Patterns a branch must match none of.
        tags: Patterns a tag must match one of.
        tags_ignore: Patterns a tag must match none of.
    """

    branches: tuple[str, ...] | None
    branches_ignore: tuple[str, ...] | None
    tags: tuple[str, ...] | None
    tags_ignore: tuple[str, ...] | None


#: A bare ``push`` event: every branch and every tag.
EVERY_REF: Final = PushTrigger(None, None, None, None)


class Workflow(NamedTuple):
    """One workflow file, as far as a push is concerned.

    Attributes:
        name: Its file name under ``.github/workflows``.
        push: Its push trigger, or None when it declares none.
    """

    name: str
    push: PushTrigger | None


def _unread(name: str, line: int, what: str) -> AppError[CiWakeErrorCode]:
    """Build the refusal for a shape the reader does not read.

    Args:
        name: The workflow's file name.
        line: The 1-based line.
        what: What was found there, in words.

    Returns:
        The error to raise.
    """
    return AppError(
        code=CiWakeErrorCode.WORKFLOW_TRIGGER_UNREAD,
        message=(
            f"{name} line {line}: {what}; whether this push starts a workflow is read from "
            "every workflow's on: block, and one it cannot read cannot be judged"
        ),
    )


def _strip_comment(text: str) -> str:
    """Remove a trailing ``#`` comment, outside quotes.

    Args:
        text: A line's text after its key.

    Returns:
        The text before the comment, whitespace stripped.
    """
    quote = ""
    for index, char in enumerate(text):
        if quote != "":
            quote = "" if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char == "#" and (index == 0 or text[index - 1] in " \t"):
            return text[:index].strip()
    return text.strip()


def _scalar(text: str, name: str, line: int) -> str:
    """Read one event name or pattern.

    Args:
        text: The value, comment already stripped.
        name: The workflow's file name.
        line: The 1-based line.

    Returns:
        The scalar, quotes removed.

    Raises:
        AppError: ``WORKFLOW_TRIGGER_UNREAD`` for anything but a quoted
            string without its own quote inside, or a plain word.
    """
    for quote in "\"'":
        if len(text) >= 2 and text[0] == quote and text[-1] == quote:
            inner = text[1:-1]
            if quote not in inner:
                return inner
    if _BARE.match(text) is None:
        raise _unread(name, line, f"{text!r} is not a quoted string or a plain word")
    return text


def _flow_list(text: str, name: str, line: int) -> tuple[str, ...]:
    """Read ``[a, b]``.

    Args:
        text: The value, comment already stripped, starting with ``[``.
        name: The workflow's file name.
        line: The 1-based line.

    Returns:
        Its items.

    Raises:
        AppError: ``WORKFLOW_TRIGGER_UNREAD`` when it does not close on the
            same line or an item is not a scalar.
    """
    if not text.endswith("]"):
        raise _unread(name, line, f"{text!r} is a flow sequence that does not close on its line")
    inner = text[1:-1].strip()
    if inner == "":
        return ()
    return tuple(_scalar(item.strip(), name, line) for item in inner.split(","))


def _indent(text: str) -> int:
    """Count a line's leading spaces.

    Args:
        text: The line.

    Returns:
        Its indent.
    """
    return len(text) - len(text.lstrip(" "))


def _ref_pattern(key: str, pattern: str, name: str, line: int) -> str:
    """Refuse a ref filter pattern using an operator this reader does not match.

    Path filter patterns are kept as written: they are never matched here
    (see this module's docstring), so an operator in one decides nothing.

    Args:
        key: The filter the pattern belongs to.
        pattern: The pattern.
        name: The workflow's file name.
        line: The 1-based line.

    Returns:
        The pattern unchanged.

    Raises:
        AppError: ``WORKFLOW_TRIGGER_UNREAD`` for a branch or tag pattern
            that negates or uses ``?``, ``+``, ``[`` or ``]``.
    """
    if key.startswith("paths"):
        return pattern
    if pattern.startswith("!"):
        raise _unread(name, line, f"pattern {pattern!r} negates, which is order-dependent")
    for operator in _UNREAD_OPERATORS:
        if operator in pattern:
            raise _unread(name, line, f"pattern {pattern!r} uses the operator {operator!r}")
    return pattern


def _push_trigger(body: Sequence[tuple[int, str]], name: str, push_line: int) -> PushTrigger:
    """Read the lines below ``push:``.

    Args:
        body: Its lines, deeper than the event's indent, comments removed.
        name: The workflow's file name.
        push_line: The 1-based line of ``push:``.

    Returns:
        The trigger.

    Raises:
        AppError: ``WORKFLOW_TRIGGER_UNREAD`` for a key GitHub does not
            define, a repeated key, a value that is neither a flow nor a
            block sequence, both filters for one kind, or an unread pattern.
    """
    filters: dict[str, tuple[str, ...]] = {}
    indent = _indent(body[0][1])
    # The filter whose block sequence the next ``- item`` lines extend, or
    # the empty string after a flow sequence, which takes no items below it.
    open_key = ""
    for number, text in body:
        item = _ITEM.match(text)
        if item is not None and open_key != "" and _indent(text) > indent:
            pattern = _scalar(_strip_comment(item.group(2)), name, number)
            filters[open_key] = (*filters[open_key], _ref_pattern(open_key, pattern, name, number))
            continue
        found = _KEY.match(text)
        if found is None or _indent(text) != indent:
            raise _unread(name, number, f"{text.strip()!r} is not a filter of push")
        key = found.group(2)
        if key not in _FILTERS:
            raise _unread(name, number, f"push has no filter named {key!r}")
        if key in filters:
            raise _unread(name, number, f"push names {key!r} twice")
        value = _strip_comment(found.group(3))
        if value.startswith("["):
            patterns = _flow_list(value, name, number)
            filters[key] = tuple(_ref_pattern(key, p, name, number) for p in patterns)
            open_key = ""
        elif value == "":
            filters[key] = ()
            open_key = key
        else:
            raise _unread(name, number, f"{key} is {value!r}, not a sequence")
    for kind in ("branches", "tags"):
        if kind in filters and f"{kind}-ignore" in filters:
            raise _unread(name, push_line, f"push names both {kind} and {kind}-ignore")
    return PushTrigger(
        branches=filters.get("branches"),
        branches_ignore=filters.get("branches-ignore"),
        tags=filters.get("tags"),
        tags_ignore=filters.get("tags-ignore"),
    )


def _event_block(body: Sequence[tuple[int, str]], name: str) -> PushTrigger | None:
    """Read the event mapping below ``on:``.

    Args:
        body: Its lines, comments and blank lines removed.
        name: The workflow's file name.

    Returns:
        The push trigger, or None when no event is ``push``.

    Raises:
        AppError: ``WORKFLOW_TRIGGER_UNREAD`` for a line that is not an
            event at the block's indent, a repeated ``push``, or ``push``
            given an inline value.
    """
    indent = _indent(body[0][1])
    events: list[tuple[int, str, str, list[tuple[int, str]]]] = []
    for number, text in body:
        if _indent(text) > indent and events:
            events[-1][3].append((number, text))
            continue
        found = _KEY.match(text)
        if found is None or _indent(text) != indent:
            raise _unread(name, number, f"{text.strip()!r} is not an event of on:")
        events.append((number, found.group(2), _strip_comment(found.group(3)), []))
    pushes = [event for event in events if event[1] == "push"]
    if len(pushes) > 1:
        raise _unread(name, pushes[1][0], "on: names push twice")
    if not pushes:
        return None
    number, _, value, below = pushes[0]
    if value != "":
        raise _unread(name, number, f"push is {value!r}, not a mapping of filters or empty")
    # ``push:`` with nothing below it is API's packages.yml: every ref.
    return _push_trigger(below, name, number) if below else EVERY_REF


def read_workflow(name: str, text: str) -> Workflow:
    """Read one workflow's push trigger.

    Args:
        name: Its file name, for the refusal.
        text: Its content.

    Returns:
        The workflow.

    Raises:
        AppError: ``WORKFLOW_TRIGGER_UNREAD`` when it has no ``on:`` at
            column zero, two of them, or one in a shape this does not read.
    """
    lines = text.splitlines()
    headers = [
        (index, found)
        for index, line in enumerate(lines)
        if (found := _ON_HEADER.match(line)) is not None
    ]
    if not headers:
        raise _unread(name, 1, "no on: at column zero")
    if len(headers) > 1:
        raise _unread(name, headers[1][0] + 1, "a second on: at column zero")
    start, header = headers[0]
    value = _strip_comment(header.group(1))
    if value.startswith("{"):
        raise _unread(name, start + 1, "on: is a flow mapping")
    if value != "":
        events = (
            _flow_list(value, name, start + 1)
            if value.startswith("[")
            else (_scalar(value, name, start + 1),)
        )
        return Workflow(name, EVERY_REF if "push" in events else None)
    body: list[tuple[int, str]] = []
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() == "" or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            break
        body.append((index + 1, line.rstrip()))
    if not body:
        raise _unread(name, start + 1, "on: is empty")
    return Workflow(name, _event_block(body, name))


def _matches(pattern: str, name: str) -> bool:
    """Match a ref name against a GitHub filter pattern.

    Args:
        pattern: A pattern of literals, ``*`` (any run without ``/``) and
            ``**`` (any run), the operators :func:`_ref_pattern` admits.
        name: The branch or tag name.

    Returns:
        Whether the whole name matches.
    """
    tokens = [found.group(0) for found in _PATTERN_TOKEN.finditer(pattern)]
    regex = "".join(
        ".*" if token == "**" else "[^/]*" if token == "*" else re.escape(token) for token in tokens
    )
    return re.fullmatch(regex, name) is not None


def admits(trigger: PushTrigger, ref: str) -> bool:
    """Whether a push trigger runs for a push updating this ref.

    Args:
        trigger: The workflow's push trigger.
        ref: The remote ref the push updates, ``refs/...``.

    Returns:
        True when GitHub would start the workflow for it, path filters
        aside (see this module's docstring).
    """
    if ref.startswith(BRANCH_PREFIX):
        name = ref[len(BRANCH_PREFIX) :]
        positive, ignore = trigger.branches, trigger.branches_ignore
    elif ref.startswith(TAG_PREFIX):
        name = ref[len(TAG_PREFIX) :]
        positive, ignore = trigger.tags, trigger.tags_ignore
    else:
        return False
    if trigger == EVERY_REF:
        return True
    if positive is not None:
        return any(_matches(pattern, name) for pattern in positive)
    if ignore is not None:
        return not any(_matches(pattern, name) for pattern in ignore)
    return False


def describe(workflow: Workflow) -> str:
    """Say what a workflow's push trigger admits, for the hook's line.

    Args:
        workflow: The workflow.

    Returns:
        ``check.yml: push on branches main``, ``packages.yml: push on every
        branch and tag``, or ``runner-reaper-proof.yml: no push trigger``.
    """
    if workflow.push is None:
        return f"{workflow.name}: no push trigger"
    if workflow.push == EVERY_REF:
        return f"{workflow.name}: push on every branch and tag"
    labels = ("branches", "branches-ignore", "tags", "tags-ignore")
    filters = [
        f"{label} {', '.join(patterns) if patterns else '(none)'}"
        for label, patterns in zip(labels, workflow.push, strict=True)
        if patterns is not None
    ]
    return f"{workflow.name}: push on {'; '.join(filters)}"


__all__ = [
    "BRANCH_PREFIX",
    "EVERY_REF",
    "TAG_PREFIX",
    "PushTrigger",
    "Workflow",
    "admits",
    "describe",
    "read_workflow",
]
