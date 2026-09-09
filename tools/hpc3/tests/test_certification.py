"""Tests for refusing a run whose inputs no certification record vouches for.

The check exists because :mod:`hpc3.core.inputs` closed only the loud half.
An ABSENT payload killed job 55806418 in fourteen seconds; a payload that is
PRESENT and wrong trains to completion and is comparable to nothing. So these
hold it to the distinction that makes it worth having: it refuses a file whose
digest no neighbouring record names, and it stays silent about a file that is
merely missing, because that is a different question with a different answer.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core import _test_hooks as core_hooks
from hpc3.core._test_hooks import CommandResult
from hpc3.core.certification import (
    certification_probe,
    parse_probe,
    require_inputs_certified,
    uncertified,
)

_PAYLOAD = "/pub/w/payloads/train.json"
_SPEC = "/pub/w/specs/gen.json"

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _probe_output(*blocks: tuple[str, str, Sequence[str]]) -> str:
    """Render what the probe prints for a set of paths.

    Args:
        blocks: One ``(path, digest, record_digests)`` per path. An empty
            digest renders a file the cluster could not hash.

    Returns:
        The probe's stdout.
    """
    lines: list[str] = []
    for path, digest, records in blocks:
        lines.append(f"FILE {path}")
        if digest:
            lines.append(f"{digest}  {path}")
        lines.append(f"RECORDS {path}")
        lines.extend(f"{record}  something.jsonl" for record in records)
    return "\n".join(lines) + "\n"


class TestTheProbe:
    """One command, and it must not fail on the ordinary answers."""

    def test_it_asks_about_every_path(self) -> None:
        command = certification_probe((_PAYLOAD, _SPEC))

        assert _PAYLOAD in command
        assert _SPEC in command

    def test_a_missing_file_and_an_empty_glob_are_answers_rather_than_failures(self) -> None:
        """Both halves end `; true`.

        Without it a missing file makes sha256sum exit non-zero, or an empty
        `*-digests.txt` glob makes cat exit non-zero, and run_remote raises
        REMOTE_COMMAND_FAILED -- reporting an infrastructure fault for the
        two ordinary cases this check exists to detect.
        """
        command = certification_probe((_PAYLOAD,))

        assert command.count("; true") == 2

    def test_it_reads_records_from_the_file_s_own_directory(self) -> None:
        """Beside the data, which is where hpc3-stage writes them."""
        command = certification_probe((_PAYLOAD,))

        assert "dirname" in command
        assert "-digests.txt" in command


class TestReadingTheAnswer:
    def test_a_certified_file_is_matched_to_its_record(self) -> None:
        parsed = parse_probe(_probe_output((_PAYLOAD, _DIGEST_A, [_DIGEST_A])), (_PAYLOAD,))

        assert parsed[_PAYLOAD] == (_DIGEST_A, frozenset({_DIGEST_A}))

    def test_a_record_naming_other_digests_does_not_vouch_for_this_file(self) -> None:
        parsed = parse_probe(_probe_output((_PAYLOAD, _DIGEST_A, [_DIGEST_B])), (_PAYLOAD,))

        assert parsed[_PAYLOAD] == (_DIGEST_A, frozenset({_DIGEST_B}))

    def test_an_absent_file_reads_as_no_digest(self) -> None:
        parsed = parse_probe(_probe_output((_PAYLOAD, "", [])), (_PAYLOAD,))

        assert parsed[_PAYLOAD] == ("", frozenset())

    def test_a_path_the_probe_said_nothing_about_is_still_keyed(self) -> None:
        """The result is keyed by what was ASKED, not by what came back.

        A path missing from the output would otherwise raise a KeyError in
        the caller, turning a quiet cluster into a crash rather than a
        refusal.
        """
        parsed = parse_probe("", (_PAYLOAD, _SPEC))

        assert parsed == {_PAYLOAD: ("", frozenset()), _SPEC: ("", frozenset())}

    def test_output_before_any_marker_is_ignored(self) -> None:
        """ssh banners and motd lines arrive ahead of the first marker."""
        noise = "Welcome to the cluster\nLast login: never\n"
        parsed = parse_probe(noise + _probe_output((_PAYLOAD, _DIGEST_A, [_DIGEST_A])), (_PAYLOAD,))

        assert parsed[_PAYLOAD] == (_DIGEST_A, frozenset({_DIGEST_A}))

    def test_a_line_that_is_not_a_digest_line_does_not_become_one(self) -> None:
        """sha256sum's own error text sits in the file section."""
        output = f"FILE {_PAYLOAD}\nsha256sum: {_PAYLOAD}: No such file\nRECORDS {_PAYLOAD}\n"
        parsed = parse_probe(output, (_PAYLOAD,))

        assert parsed[_PAYLOAD] == ("", frozenset())

    def test_several_paths_are_kept_apart(self) -> None:
        parsed = parse_probe(
            _probe_output(
                (_PAYLOAD, _DIGEST_A, [_DIGEST_A]),
                (_SPEC, _DIGEST_B, []),
            ),
            (_PAYLOAD, _SPEC),
        )

        assert parsed[_PAYLOAD] == (_DIGEST_A, frozenset({_DIGEST_A}))
        assert parsed[_SPEC] == (_DIGEST_B, frozenset())


class TestWhichAreUnvouchedFor:
    def test_a_file_no_record_names_is_reported(self) -> None:
        probed: dict[str, tuple[str, frozenset[str]]] = {_PAYLOAD: (_DIGEST_A, frozenset())}

        assert uncertified((_PAYLOAD,), probed) == (_PAYLOAD,)

    def test_a_file_its_record_names_is_not(self) -> None:
        probed: dict[str, tuple[str, frozenset[str]]] = {
            _PAYLOAD: (_DIGEST_A, frozenset({_DIGEST_A}))
        }

        assert uncertified((_PAYLOAD,), probed) == ()

    def test_an_absent_file_is_not_reported_here(self) -> None:
        """Absence is require_inputs_present's question.

        Reporting it twice would tell a reader to certify a file they already
        know is missing, and bury the ones that are present and unvouched --
        which are the only ones this check can say anything new about.
        """
        probed: dict[str, tuple[str, frozenset[str]]] = {_PAYLOAD: ("", frozenset())}

        assert uncertified((_PAYLOAD,), probed) == ()

    def test_order_follows_the_declaration(self) -> None:
        probed: dict[str, tuple[str, frozenset[str]]] = {
            _PAYLOAD: (_DIGEST_A, frozenset()),
            _SPEC: (_DIGEST_B, frozenset()),
        }

        assert uncertified((_PAYLOAD, _SPEC), probed) == (_PAYLOAD, _SPEC)


class TestTheRefusal:
    def test_an_uncertified_input_is_refused_and_named_with_its_digest(self) -> None:
        def _fake_run(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
            _ = stdin_bytes, argv
            return CommandResult(
                returncode=0,
                stdout=_probe_output((_PAYLOAD, _DIGEST_A, [_DIGEST_B])),
                stderr="",
            )

        original = core_hooks.run
        core_hooks.run = _fake_run
        try:
            with pytest.raises(AppError) as caught:
                require_inputs_certified("hpc3", (_PAYLOAD,))
        finally:
            core_hooks.run = original

        error: AppError[Hpc3ErrorCode] = caught.value
        message = str(error)
        assert error.code == Hpc3ErrorCode.RUN_INPUT_UNCERTIFIED
        assert _PAYLOAD in message
        # The digest is in the message because the next thing a reader does is
        # compare it to the repo copy, and making them run sha256sum again is
        # making them re-derive what this command already knows.
        assert _DIGEST_A in message
        assert "hpc3-stage" in message

    def test_a_certified_input_passes(self) -> None:
        def _fake_run(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
            _ = stdin_bytes, argv
            return CommandResult(
                returncode=0,
                stdout=_probe_output((_PAYLOAD, _DIGEST_A, [_DIGEST_A])),
                stderr="",
            )

        original = core_hooks.run
        core_hooks.run = _fake_run
        try:
            require_inputs_certified("hpc3", (_PAYLOAD,))
        finally:
            core_hooks.run = original

    def test_no_paths_reaches_no_network(self) -> None:
        """A run declaring nothing must not pay for an ssh round trip."""

        def _explode(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
            _ = stdin_bytes
            raise AssertionError(f"the cluster was contacted for nothing: {list(argv)}")

        original = core_hooks.run
        core_hooks.run = _explode
        try:
            require_inputs_certified("hpc3", ())
        finally:
            core_hooks.run = original

    def test_every_unvouched_path_is_named_not_only_the_first(self) -> None:
        """Otherwise a reader stages one file and runs into the next."""

        def _fake_run(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
            _ = stdin_bytes, argv
            return CommandResult(
                returncode=0,
                stdout=_probe_output((_PAYLOAD, _DIGEST_A, []), (_SPEC, _DIGEST_B, [])),
                stderr="",
            )

        original = core_hooks.run
        core_hooks.run = _fake_run
        try:
            with pytest.raises(AppError) as caught:
                require_inputs_certified("hpc3", (_PAYLOAD, _SPEC))
        finally:
            core_hooks.run = original

        message = str(caught.value)
        assert _PAYLOAD in message
        assert _SPEC in message

    def test_one_round_trip_for_the_whole_set(self) -> None:
        """This sits in front of every submission; a per-path probe would
        make the common case -- everything certified -- the slowest."""
        calls: list[list[str]] = []

        def _fake_run(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
            _ = stdin_bytes
            calls.append(list(argv))
            return CommandResult(
                returncode=0,
                stdout=_probe_output(
                    (_PAYLOAD, _DIGEST_A, [_DIGEST_A]), (_SPEC, _DIGEST_B, [_DIGEST_B])
                ),
                stderr="",
            )

        original = core_hooks.run
        core_hooks.run = _fake_run
        try:
            require_inputs_certified("hpc3", (_PAYLOAD, _SPEC))
        finally:
            core_hooks.run = original

        assert len(calls) == 1


__all__ = [
    "TestReadingTheAnswer",
    "TestTheProbe",
    "TestTheRefusal",
    "TestWhichAreUnvouchedFor",
]
