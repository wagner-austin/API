"""Shared entrypoint helpers for live action-lab probe sessions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypedDict, TypeVar

from platform_core.json_utils import JSONObject, dump_json_str
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.action_lab.capture import save_capture_session
from tankpit_bot.types import CapturedMessage

log = get_logger(__name__)

ProbeT_co = TypeVar("ProbeT_co", bound="ProbeArtifactsProtocol", covariant=True)
ProbeT = TypeVar("ProbeT", bound="ProbeArtifactsProtocol")
SessionT = TypeVar("SessionT")
StandardSessionT = TypeVar("StandardSessionT", bound="StandardProbeSessionDict")


class ProbeArtifactsProtocol(Protocol):
    """Minimal probe surface required to persist probe artifacts."""

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return captured probe messages."""

    @property
    def magic(self) -> str | None:
        """Return the active capture magic, when available."""


class ProbeCaptureMetadataDict(TypedDict):
    """Capture-session metadata extracted from one probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str


class StandardProbeSessionDict(TypedDict):
    """Shared session fields used by the standard probe entrypoint helpers."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    capture_session_path: str


class ProbeFactoryProtocol(Protocol[ProbeT_co]):
    """Typed callable used to build one live probe instance."""

    def __call__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
    ) -> ProbeT_co:
        """Create one probe instance."""


def run_and_save_probe_session(
    *,
    probe_factory: ProbeFactoryProtocol[ProbeT],
    run_session: Callable[[ProbeT], SessionT],
    extract_capture_metadata: Callable[[SessionT], ProbeCaptureMetadataDict],
    encode_with_capture_path: Callable[[SessionT, str], JSONObject],
    summary_formatter: Callable[[SessionT], str],
    target_url: str,
    output_path: str,
    headless: bool,
    prefer_account: bool,
) -> SessionT:
    """Run one live probe session and persist its structured artifacts.

    Args:
        probe_factory: Typed probe constructor.
        run_session: Session execution callback for the constructed probe.
        extract_capture_metadata: Capture metadata extractor for the session.
        encode_with_capture_path: Encoder that attaches the saved capture path.
        summary_formatter: Human-readable summary formatter.
        target_url: Browser target URL.
        output_path: JSON output path for the structured session artifact.
        headless: Whether the browser should run headless.
        prefer_account: Whether to prefer account login.

    Returns:
        Completed and persisted session payload.
    """
    probe = probe_factory(target_url, headless=headless, prefer_account=prefer_account)
    session = run_session(probe)
    capture_metadata = extract_capture_metadata(session)
    capture_session_path = save_capture_session(
        session_id=capture_metadata["session_id"],
        start_timestamp_ms=capture_metadata["start_timestamp_ms"],
        end_timestamp_ms=capture_metadata["end_timestamp_ms"],
        base_url=capture_metadata["base_url"],
        messages=probe.messages,
        magic=probe.magic,
        output_path=output_path,
    )
    encoded = encode_with_capture_path(session, capture_session_path)
    json_str = dump_json_str(encoded, compact=False, indent=2)
    _test_hooks.write_text(Path(output_path), json_str)
    log.info(summary_formatter(session))
    return session


def extract_standard_capture_metadata(
    session: StandardProbeSessionDict,
) -> ProbeCaptureMetadataDict:
    """Extract standard capture metadata from one persisted probe session.

    Args:
        session: Session payload containing the standard metadata keys.

    Returns:
        Standard capture metadata dict used for raw capture persistence.
    """
    return ProbeCaptureMetadataDict(
        session_id=session["session_id"],
        start_timestamp_ms=session["start_timestamp_ms"],
        end_timestamp_ms=session["end_timestamp_ms"],
        base_url=session["base_url"],
    )


def encode_standard_probe_session(
    session: StandardSessionT,
    capture_session_path: str,
    *,
    encoder: Callable[[StandardSessionT], JSONObject],
) -> JSONObject:
    """Attach the capture path and encode one standard probe session.

    Args:
        session: Session payload to encode.
        capture_session_path: Persisted raw capture-session artifact path.
        encoder: Typed encoder for the session payload.

    Returns:
        Encoded JSON object for the session.
    """
    session["capture_session_path"] = capture_session_path
    return encoder(session)


def run_and_save_standard_probe_session(
    *,
    probe_factory: ProbeFactoryProtocol[ProbeT],
    run_session: Callable[[ProbeT], StandardSessionT],
    encoder: Callable[[StandardSessionT], JSONObject],
    summary_formatter: Callable[[StandardSessionT], str],
    target_url: str,
    output_path: str,
    headless: bool,
    prefer_account: bool,
) -> StandardSessionT:
    """Run and persist one standard probe session with shared metadata handling.

    Args:
        probe_factory: Typed probe constructor.
        run_session: Session execution callback for the constructed probe.
        encoder: Typed encoder for the session payload.
        summary_formatter: Human-readable summary formatter.
        target_url: Browser target URL.
        output_path: JSON output path for the structured session artifact.
        headless: Whether the browser should run headless.
        prefer_account: Whether to prefer account login.

    Returns:
        Completed and persisted session payload.
    """
    return run_and_save_probe_session(
        probe_factory=probe_factory,
        run_session=run_session,
        extract_capture_metadata=extract_standard_capture_metadata,
        encode_with_capture_path=(
            lambda session, capture_session_path: encode_standard_probe_session(
                session,
                capture_session_path,
                encoder=encoder,
            )
        ),
        summary_formatter=summary_formatter,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "ProbeArtifactsProtocol",
    "ProbeCaptureMetadataDict",
    "ProbeFactoryProtocol",
    "StandardProbeSessionDict",
    "encode_standard_probe_session",
    "extract_standard_capture_metadata",
    "run_and_save_probe_session",
    "run_and_save_standard_probe_session",
]
