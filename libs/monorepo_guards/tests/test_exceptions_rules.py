from __future__ import annotations

from pathlib import Path

from monorepo_guards.exceptions_rules import ExceptionsRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_exceptions_rule_flags_silent_and_broad(tmp_path: Path) -> None:
    code = (
        "try:\n"
        "    1/0\n"
        "except Exception:\n"
        "    pass\n"
        "\n"
        "try:\n"
        "    1/0\n"
        "except Exception:\n"
        "    logger.error('x')\n"
        "\n"
        "try:\n"
        "    1/0\n"
        "except Exception:\n"
        "    raise\n"
        "\n"
        "try:\n"
        "    1/0\n"
        "except ValueError:\n"
        "    a = 1\n"
    )
    path = tmp_path / "exc_mod.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "silent-except-body" in kinds
    assert "broad-except-requires-log-and-raise" in kinds
    assert "except-discards-the-error" in kinds


def test_exceptions_rule_typed_with_log_or_raise_is_ok(tmp_path: Path) -> None:
    code = (
        "try:\n"
        "    1/0\n"
        "except ValueError:\n"
        "    logger.error('x')\n"
        "\n"
        "try:\n"
        "    1/0\n"
        "except KeyError:\n"
        "    raise\n"
    )
    path = tmp_path / "typed_ok.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([path])
    assert violations == []


def test_exceptions_rule_no_body_detected(tmp_path: Path) -> None:
    # An except header at EOF with no body should be flagged as silent body
    code = "try:\n    1/0\nexcept Exception:\n"
    path = tmp_path / "no_body.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "silent-except-body" in kinds


def test_exceptions_rule_broad_with_log_and_raise_and_skip_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.py"
    _write(empty, "")

    code = (
        "try:\n    1/0\nexcept Exception:\n    logger.error('x')\n    raise RuntimeError('fail')\n"
    )
    path = tmp_path / "broad_ok.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([empty, path])
    assert violations == []


def test_exceptions_rule_body_start_skips_blank_lines(tmp_path: Path) -> None:
    code = "try:\n    1/0\nexcept Exception:\n\n    logger.error('x')\n    raise\n"
    path = tmp_path / "blank_body.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([path])
    # Broad except with both log and raise after a blank line should be allowed
    assert violations == []


def test_exceptions_rule_accepts_write_line_as_surfacing(tmp_path: Path) -> None:
    code = (
        "try:\n"
        "    1/0\n"
        "except ValueError as error:\n"
        "    _test_hooks.write_line(f'refused: {error}')\n"
        "    result = None\n"
    )
    path = tmp_path / "write_line_ok.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([path])
    # write_line is the stdlib-only clients' output channel; calling it
    # in a narrow except body surfaces the failure like a log call.
    assert violations == []


def test_exceptions_rule_accepts_emit_error_as_surfacing(tmp_path: Path) -> None:
    """A CLI's top-level translator writes the refusal to stderr and exits.

    That is the loudest surfacing available to a command-line tool -- the
    operator reads stderr, not the log -- so it satisfies the rule for the
    same reason write_line does.
    """
    code = (
        "try:\n"
        "    return main(None)\n"
        "except ValueError as usage:\n"
        "    _test_hooks.emit_error(f'usage: {usage}')\n"
        "    return EXIT_REFUSED\n"
    )
    path = tmp_path / "emit_error_ok.py"
    _write(path, code)

    rule = ExceptionsRule()
    assert rule.run([path]) == []


def test_exceptions_rule_still_flags_a_body_that_surfaces_nothing(tmp_path: Path) -> None:
    """The rule has to stay able to fail, or the two acceptances above are
    indistinguishable from a rule that accepts everything.

    The body here neither transfers control nor names ``usage``, so the caught
    error reaches nobody and execution carries on as though it never happened.
    That is the one shape this rule exists to catch.
    """
    code = "try:\n    return main(None)\nexcept ValueError as usage:\n    fallback = EXIT_REFUSED\n"
    path = tmp_path / "silent.py"
    _write(path, code)

    rule = ExceptionsRule()
    assert [v.kind for v in rule.run([path])] == ["except-discards-the-error"]


def test_exceptions_rule_stops_at_a_dedented_ordinary_statement(tmp_path: Path) -> None:
    """A handler's body ends at the first dedent, not only at except/finally/else.

    The scanner used to break only on a dedented ``except``/``finally``/``else``,
    so any other dedented line let it keep reading. A ``raise`` belonging to the
    code AFTER the try block was then credited to the handler, and a silent
    handler passed. The failure mode was a false negative, so nothing went red.
    """
    code = (
        "def handler() -> None:\n"
        "    try:\n"
        "        risky()\n"
        "    except ValueError:\n"
        "        fallback = 1\n"
        "    raise RuntimeError('this belongs to handler, not to the except body')\n"
    )
    path = tmp_path / "dedent_mod.py"
    _write(path, code)

    violations = ExceptionsRule().run([path])
    assert [v.kind for v in violations] == ["except-discards-the-error"]


def test_exceptions_rule_stops_at_a_following_def(tmp_path: Path) -> None:
    """The next function's body must not be read as part of the handler."""
    code = (
        "def first() -> None:\n"
        "    try:\n"
        "        risky()\n"
        "    except OSError:\n"
        "        fallback = 1\n"
        "\n"
        "def second() -> None:\n"
        "    logger.error('unrelated')\n"
        "    raise RuntimeError('unrelated')\n"
    )
    path = tmp_path / "next_def_mod.py"
    _write(path, code)

    violations = ExceptionsRule().run([path])
    assert [v.kind for v in violations] == ["except-discards-the-error"]


def test_exceptions_rule_accepts_a_typed_handler_that_transfers_control(
    tmp_path: Path,
) -> None:
    """A typed handler that returns has decided the outcome, not swallowed it.

    ``except ValueError: return stripped`` is how a typed conversion fallback
    is written. Requiring a log or a raise there would force every such
    handler to either shout about an expected condition or stop being a
    fallback. The sibling TypeScript rule draws the same line: an arm that
    rethrows, returns or exits needs nothing further.
    """
    code = (
        "def parse(text: str) -> object:\n"
        "    try:\n"
        "        return float(text)\n"
        "    except ValueError:\n"
        "        return text\n"
    )
    path = tmp_path / "transfer_mod.py"
    _write(path, code)

    assert ExceptionsRule().run([path]) == []


def test_exceptions_rule_accepts_a_handler_that_names_the_error(tmp_path: Path) -> None:
    """Falling through is fine when the body carries the error somewhere."""
    code = (
        "def collect(items: list[str]) -> list[str]:\n"
        "    refusals: list[str] = []\n"
        "    for item in items:\n"
        "        try:\n"
        "            check(item)\n"
        "        except CheckError as error:\n"
        "            refusals.append(f'{item}: {error}')\n"
        "    return refusals\n"
    )
    path = tmp_path / "alias_mod.py"
    _write(path, code)

    assert ExceptionsRule().run([path]) == []


def test_exceptions_rule_alias_match_is_word_bounded(tmp_path: Path) -> None:
    """An alias must be referenced, not merely appear inside another name.

    ``err`` occurring inside ``error_count`` is not a reference to the caught
    exception, and accepting it would let any handler mentioning a similarly
    spelled local pass.
    """
    code = (
        "def run() -> None:\n"
        "    try:\n"
        "        risky()\n"
        "    except OSError as err:\n"
        "        error_count = 1\n"
    )
    path = tmp_path / "wordbound_mod.py"
    _write(path, code)

    assert [v.kind for v in ExceptionsRule().run([path])] == ["except-discards-the-error"]
