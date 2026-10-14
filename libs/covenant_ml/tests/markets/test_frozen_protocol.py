"""Published experiment protocol bytes survive checkout line-ending conversion."""

from hashlib import sha256
from pathlib import Path


def test_temporal_transfer_protocol_retains_recorded_source_digest() -> None:
    protocol = Path(__file__).parents[2] / "docs" / "MAKER_TEMPORAL_TRANSFER_PROTOCOL.md"
    assert sha256(protocol.read_bytes()).hexdigest() == (
        "763fdf28497d32a4917838eb89b3dfec462dd3c4a396629a4114d2ee31f1247e"
    )
