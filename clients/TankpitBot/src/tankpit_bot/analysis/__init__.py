"""Typed archive analysis: one owner for the capture-scan pipeline.

The forty one-shot miners that once lived under ``analysis_scripts/``
(retired 2026-08-17 — ungated, 8,543 strict-mypy errors from the bar;
recoverable from git history, findings recorded in the wiki) each
re-implemented this same pipeline privately — 30 re-wrote the session
load, 26 the XOR bring-up, and 10 forked the frame walk that
:mod:`tankpit_bot.protocol.framing` has owned since the protocol
layer was written. Eight of those ten forks were textually distinct
versions of identical arithmetic.

This package is the single owner of that pipeline. It composes modules
that already exist rather than restating them, and it is inside ``src``
so it carries the same strict typing, guard rules and coverage floor as
every other package here.
"""

from __future__ import annotations

from tankpit_bot.analysis.scan import (
    decode_session_frames,
    load_capture_session,
    scan_archive,
    scan_session,
)
from tankpit_bot.analysis.types import (
    SESSION_SKIP_REASONS,
    DecodedFrameDict,
    ScannedSessionDict,
    SessionSkipReason,
    SkippedSessionDict,
    decode_decoded_frame,
    decode_skipped_session,
    encode_decoded_frame,
    encode_skipped_session,
    require_hex_bytes,
    require_session_skip_reason,
)

__all__ = [
    "SESSION_SKIP_REASONS",
    "DecodedFrameDict",
    "ScannedSessionDict",
    "SessionSkipReason",
    "SkippedSessionDict",
    "decode_decoded_frame",
    "decode_session_frames",
    "decode_skipped_session",
    "encode_decoded_frame",
    "encode_skipped_session",
    "load_capture_session",
    "require_hex_bytes",
    "require_session_skip_reason",
    "scan_archive",
    "scan_session",
]
