"""Tests for ContainerTracker class."""

from __future__ import annotations

import base64

from tankpit_bot.capture import ContainerTracker
from tests.conftest import FakeFileSystem
from tests.sniffer.trackers.conftest import build_test_xor_table, make_payload


def _make_xor_payload(decoded_data: bytes, xor_table: bytes) -> str:
    """Create XOR-encoded base64 payload for testing."""
    encoded = bytes(decoded_data[i] ^ xor_table[i] for i in range(len(decoded_data)))
    body = bytes([0x2E]) + encoded
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()


class TestContainerTracker:
    """Tests for ContainerTracker class."""

    def test_init(self) -> None:
        """Test ContainerTracker initialization."""
        tracker = ContainerTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._containers == {}

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        tracker = ContainerTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_containers_property(self) -> None:
        """Test containers property returns copy of container dict."""
        tracker = ContainerTracker()
        tracker._containers = {1: 100, 2: 200}
        result = tracker.containers
        assert result == {1: 100, 2: 200}
        # Verify it's a copy
        result[3] = 300
        assert 3 not in tracker._containers

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        tracker = ContainerTracker()
        payload = make_payload(b"\x2e\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = ContainerTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None

    def test_process_message_returns_none_for_short_data(self) -> None:
        """Test process_message returns None for short data."""
        tracker = ContainerTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = base64.b64encode(b"\x02\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None


class TestContainerTrackerProcessMessage:
    """Tests for ContainerTracker.process_message with XOR decoding."""

    def test_process_message_returns_container_info(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes container events correctly."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = ContainerTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Container format decoded: 0x43 container_id_lo container_id_hi fuel_lo fuel_hi
        decoded_data = bytes([0x43, 0x64, 0x00, 0xE8, 0x03])  # id=100, fuel=1000
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "CONTAINER" in result
        assert "100" in result
        assert "1000" in result

    def test_container_depleted(self, fake_fs: FakeFileSystem) -> None:
        """Test container tracker handles depleted containers."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = ContainerTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # First: container has fuel
        payload1 = _make_xor_payload(bytes([0x43, 0x64, 0x00, 0xE8, 0x03]), xor_table)
        tracker.process_message(payload1)

        # Second: container depleted
        payload2 = _make_xor_payload(bytes([0x43, 0x64, 0x00, 0x00, 0x00]), xor_table)
        result = tracker.process_message(payload2)

        assert result, "Expected non-None result from process_message"
        assert "DEPLETED" in result

    def test_containers_property_returns_state(self, fake_fs: FakeFileSystem) -> None:
        """Test containers property returns current state."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = ContainerTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        payload = _make_xor_payload(bytes([0x43, 0x64, 0x00, 0xE8, 0x03]), xor_table)
        tracker.process_message(payload)

        containers = tracker.containers
        assert containers[100] == 1000


class TestContainerTrackerEdgeCases:
    """Tests for ContainerTracker edge cases and uncovered branches."""

    def test_set_magic_returns_early_when_no_static_key(self) -> None:
        """Test set_magic does nothing when static key missing."""
        from tankpit_bot import _test_hooks
        from tests.conftest import FakeFileSystem

        fs = FakeFileSystem()
        _test_hooks.path_exists = fs.path_exists
        _test_hooks.read_text = fs.read_text

        tracker = ContainerTracker()
        tracker.set_magic("testmagic")
        assert tracker._xor_table is None

    def test_process_message_wrong_body_length(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for body != 6 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = ContainerTracker()
        tracker.set_magic("testmagic123")

        # Body length is 5 (should be 6)
        payload = make_payload(b"\x2e\x01\x02\x03\x04")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_wrong_prefix(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for body not starting with 0x2E."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = ContainerTracker()
        tracker.set_magic("testmagic123")

        # Body starts with 0x30 instead of 0x2E
        payload = make_payload(b"\x30\x01\x02\x03\x04\x05")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_wrong_decoded_type(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None when decoded[0] != 0x43."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = ContainerTracker()
        tracker.set_magic("testmagic123")

        xor_table = build_test_xor_table(static_key, "testmagic123")

        # Use 0x99 instead of 0x43
        decoded_data = bytes([0x99, 0x64, 0x00, 0xE8, 0x03])
        payload = _make_xor_payload(decoded_data, xor_table)
        result = tracker.process_message(payload)
        assert result is None

    def test_container_update_with_diff(self, fake_fs: FakeFileSystem) -> None:
        """Test container tracker shows diff when fuel changes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = ContainerTracker()
        tracker.set_magic("testmagic123")

        xor_table = build_test_xor_table(static_key, "testmagic123")

        # First: container 100 has 1000 fuel
        payload1 = _make_xor_payload(bytes([0x43, 0x64, 0x00, 0xE8, 0x03]), xor_table)
        tracker.process_message(payload1)

        # Second: container 100 now has 500 fuel (change)
        payload2 = _make_xor_payload(bytes([0x43, 0x64, 0x00, 0xF4, 0x01]), xor_table)
        result = tracker.process_message(payload2)

        if result is None:
            raise AssertionError("Expected non-None result from process_message")
        assert "CONTAINER:100" in result
        assert "500" in result
        assert "-500" in result  # diff shown

    def test_new_container_empty(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message shows EMPTY for new container with fuel=0."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = ContainerTracker()
        tracker.set_magic("testmagic123")

        xor_table = build_test_xor_table(static_key, "testmagic123")

        # New container (never seen before) with fuel=0
        decoded_data = bytes([0x43, 0xC8, 0x00, 0x00, 0x00])  # id=200, fuel=0
        payload = _make_xor_payload(decoded_data, xor_table)
        result = tracker.process_message(payload)

        if result is None:
            raise AssertionError("Expected non-None result from process_message")
        assert "CONTAINER:200" in result
        assert "EMPTY" in result
