"""Typed bindings to pyrage's real streaming API; no local cryptographic format."""

from __future__ import annotations

from typing import BinaryIO, Protocol


class AgeWriter(Protocol):
    """The write/flush file-like surface required by pyrage."""

    def write(self, data: bytes) -> int:
        """Write bytes or raise before exceeding the output limit."""

    def flush(self) -> None:
        """Flush the underlying file."""


class AgeRecipient(Protocol):
    """Public X25519 recipient as exposed by pyrage."""

    def __str__(self) -> str:
        """Return the public age1 encoding."""


class AgeIdentity(Protocol):
    """Secret in-memory identity; callers must never serialize it into evidence."""

    def __str__(self) -> str:
        """Return secret material only for explicit evaluator key custody."""

    def to_public(self) -> AgeRecipient:
        """Return this identity's public recipient."""


class IdentityFactory(Protocol):
    """Exact X25519 constructor surface."""

    def generate(self) -> AgeIdentity:
        """Generate a new identity with the library's secure randomness."""

    def from_str(self, value: str) -> AgeIdentity:
        """Parse a secret identity; malformed identities raise."""


class RecipientFactory(Protocol):
    """Public recipient parser."""

    def from_str(self, value: str) -> AgeRecipient:
        """Parse and validate the public age recipient."""


class X25519Module(Protocol):
    """X25519 types exported by the extension module."""

    Identity: IdentityFactory
    Recipient: RecipientFactory


class AgeBackend(Protocol):
    """File-like streaming APIs from pyrage 1.4.0."""

    def encrypt_io(
        self,
        reader: BinaryIO,
        writer: BinaryIO,
        recipients: list[AgeRecipient],
        armored: bool = False,
    ) -> None:
        """Stream age ciphertext and finalize its authentication."""

    def decrypt_io(
        self, reader: BinaryIO, writer: AgeWriter, identities: list[AgeIdentity]
    ) -> None:
        """Stream plaintext; success requires authentication through the final chunk."""


backend: AgeBackend = __import__("pyrage")
x25519: X25519Module = __import__("pyrage").x25519


__all__ = [
    "AgeBackend",
    "AgeIdentity",
    "AgeRecipient",
    "AgeWriter",
    "IdentityFactory",
    "RecipientFactory",
    "X25519Module",
    "backend",
    "x25519",
]
