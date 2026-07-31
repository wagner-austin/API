"""The planner's side of the agent channel, driven without a socket.

The connection is behind a Protocol, so these exercise the real channel code
against a scripted peer. What is being tested is sample framing — the channel
must not hand a caller a roster it can only partly see — and that orders leave
in the exact form the agent parses.
"""

from __future__ import annotations

from types import TracebackType

import pytest

from rw_bot.control import _test_hooks
from rw_bot.control.channel import (
    AgentChannel,
    ChannelError,
    open_channel,
)
from rw_bot.wire.codec import WireError
from rw_bot.wire.command import (
    ability_order,
    attack_move_order,
    attack_order,
    build_order,
    move_order,
    produce_order,
)

_FRAME_3 = (
    '{"kind":"frame","frame":854,"clock_ms":2907,"visible":3,"pools":0,"options":0,"players":0,'
    '"credits":4000,"defeated":false,"wiped":false,"players_left":6}'
)
_FRAME_1 = (
    '{"kind":"frame","frame":9,"clock_ms":30,"visible":1,"pools":0,"options":0,"players":0,'
    '"credits":4000,"defeated":false,"wiped":false,"players_left":6}'
)


def _entity(index: int, unit_id: int, type_name: str) -> str:
    return (
        f'{{"kind":"entity","frame":854,"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":1.0,"y":2.0,'
        f'"team":0,"mine":true,"hostile":false,"movement":"LAND","group":1,"flying":false,"submerged":false,"touching_water":false,"hp":100.0,"max_hp":100.0,"complete":true,"queued":0}}'
    )


class _ScriptedPeer:
    """Serves prepared lines and records what was sent back.

    Attributes:
        sent: Every line the channel wrote, in order.
        closed: Whether the channel released the connection.
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.sent: list[str] = []
        self.closed = False

    def send_line(self, line: str) -> None:
        """Record one written line.

        Args:
            line: Line content, without a newline.
        """
        self.sent.append(line)

    def read_line(self) -> str:
        """Serve the next prepared line, or end of stream.

        Returns:
            The next line, or an empty string once exhausted.
        """
        if not self._lines:
            return ""
        return self._lines.pop(0)

    def close(self) -> None:
        """Mark the connection released."""
        self.closed = True


class _StubbedConnect:
    """Binds the connect hook to a scripted peer for the duration of a block.

    Attributes:
        peer: The peer every connection returns.
        calls: One ``(host, port, timeout)`` triple per connection.
    """

    def __init__(self, peer: _ScriptedPeer) -> None:
        self.peer = peer
        self.calls: list[tuple[str, int, float]] = []
        self._original: _test_hooks.ConnectProto = _test_hooks.connect

    def __call__(self, host: str, port: int, timeout_s: float) -> _test_hooks.Connection:
        """Record one connection attempt.

        Args:
            host: Host asked for.
            port: Port asked for.
            timeout_s: Timeout asked for.

        Returns:
            The scripted peer.
        """
        self.calls.append((host, port, timeout_s))
        return self.peer

    def __enter__(self) -> _StubbedConnect:
        """Install this stub as the connect hook.

        Returns:
            This stub.
        """
        self._original = _test_hooks.connect
        _test_hooks.connect = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the original connect hook.

        Args:
            exc_type: Exception class raised in the block, if any.
            exc: Exception raised in the block, if any.
            traceback: Traceback of the raised exception, if any.
        """
        _test_hooks.connect = self._original


def test_a_whole_sample_is_returned_once_its_entities_have_arrived() -> None:
    peer = _ScriptedPeer(
        [
            _FRAME_3,
            _entity(0, 213, "commandCenter"),
            _entity(1, 214, "builder"),
            _entity(2, 217, "editorOrBuilder"),
        ]
    )
    sample = AgentChannel(peer).next_sample()
    assert sample["frame"] == 854
    assert [e["unit_id"] for e in sample["entities"]] == [213, 214, 217]


def test_a_partial_sample_is_never_returned() -> None:
    """The stream ends one entity short, so no roster is handed over at all."""
    peer = _ScriptedPeer([_FRAME_3, _entity(0, 213, "commandCenter")])
    with pytest.raises(ChannelError) as caught:
        AgentChannel(peer).next_sample()
    assert caught.value.code == "RW-CHAN-001"
    assert "1 line(s) were pending" not in caught.value.message


def test_an_immediately_closed_stream_is_reported() -> None:
    with pytest.raises(ChannelError) as caught:
        AgentChannel(_ScriptedPeer([])).next_sample()
    assert caught.value.code == "RW-CHAN-001"
    assert "0 line(s) were pending" in caught.value.message


def test_a_sample_declaring_no_entities_completes_on_its_frame_line() -> None:
    empty = (
        '{"kind":"frame","frame":1,"clock_ms":0,"visible":0,"pools":0,"options":0,"players":0,'
        '"credits":4000,"defeated":false,"wiped":false,"players_left":6}'
    )
    assert AgentChannel(_ScriptedPeer([empty])).next_sample()["entities"] == ()


def test_successive_samples_are_read_in_order() -> None:
    peer = _ScriptedPeer(
        [
            _FRAME_1,
            '{"kind":"entity","frame":9,"index":0,"id":1,"type":"builder",'
            '"class":"u","x":0.0,"y":0.0,"team":0,"mine":true,"hostile":false,"movement":"LAND","group":1,"flying":false,"submerged":false,"touching_water":false,'
            '"hp":1.0,"max_hp":1.0,"complete":true,"queued":0}',
            '{"kind":"frame","frame":10,"clock_ms":33,"visible":0,"pools":0,"options":0,"players":0,'
            '"credits":4000,"defeated":false,"wiped":false,"players_left":6}',
        ]
    )
    channel = AgentChannel(peer)
    assert channel.next_sample()["frame"] == 9
    assert channel.next_sample()["frame"] == 10


def test_an_entity_count_disagreeing_with_the_frame_still_fails_the_decoder() -> None:
    """Framing reads the count; the decoder re-checks it, so both must agree."""
    peer = _ScriptedPeer(
        [
            _FRAME_1,
            '{"kind":"entity","frame":99,"index":0,"id":1,"type":"b","class":"u",'
            '"x":0.0,"y":0.0,"team":0,"mine":true,"hostile":false,"movement":"LAND","group":1,"flying":false,"submerged":false,"touching_water":false,"hp":1.0,"max_hp":1.0,"complete":true,"queued":0}',
        ]
    )
    with pytest.raises(WireError) as caught:
        AgentChannel(peer).next_sample()
    assert caught.value.code == "RW-WIRE-004"


def test_a_move_order_leaves_in_the_agent_format() -> None:
    peer = _ScriptedPeer([])
    AgentChannel(peer).send_move(move_order(unit_id=214, x=4550.0, y=2610.0))
    assert peer.sent == ['{"kind":"move","unit_id":214,"x":4550.0,"y":2610.0}']


def test_a_build_order_leaves_in_the_agent_format() -> None:
    peer = _ScriptedPeer([])
    AgentChannel(peer).send_build(
        build_order(unit_id=214, type_name="landFactory", x=4450.0, y=2730.0)
    )
    assert peer.sent == [
        '{"kind":"build","unit_id":214,"x":4450.0,"y":2730.0,"type":"landFactory"}'
    ]


def test_a_produce_order_leaves_in_the_agent_format() -> None:
    peer = _ScriptedPeer([])
    AgentChannel(peer).send_produce(produce_order(unit_id=213, type_name="scout"))
    assert peer.sent == ['{"kind":"produce","unit_id":213,"type":"scout"}']


def test_an_attack_order_leaves_in_the_agent_format() -> None:
    peer = _ScriptedPeer([])
    AgentChannel(peer).send_attack(attack_order(unit_id=276, target_id=216))
    assert peer.sent == ['{"kind":"attack","unit_id":276,"target_id":216}']


def test_an_ability_order_leaves_in_the_agent_format() -> None:
    peer = _ScriptedPeer([])
    AgentChannel(peer).send_ability(ability_order(unit_id=213, key="c_2"))
    assert peer.sent == ['{"kind":"ability","unit_id":213,"key":"c_2"}']


def test_an_ack_leaves_in_the_agent_format() -> None:
    """Sent unconditionally, so the planner need not know the agent's mode."""
    peer = _ScriptedPeer([])
    AgentChannel(peer).send_ack()
    assert peer.sent == ['{"kind":"ack"}']


def test_closing_releases_the_connection() -> None:
    peer = _ScriptedPeer([])
    AgentChannel(peer).close()
    assert peer.closed is True


def test_open_channel_passes_the_host_port_and_timeout_through() -> None:
    peer = _ScriptedPeer([])
    with _StubbedConnect(peer) as stub:
        channel = open_channel(27200)
        channel.close()
    assert stub.calls == [("127.0.0.1", 27200, 30.0)]
    assert peer.closed is True


def test_open_channel_honours_an_explicit_host_and_timeout() -> None:
    peer = _ScriptedPeer([])
    with _StubbedConnect(peer) as stub:
        open_channel(9999, host="10.0.0.5", timeout_s=1.5)
    assert stub.calls == [("10.0.0.5", 9999, 1.5)]


def test_an_attack_move_order_leaves_in_the_agent_format() -> None:
    peer = _ScriptedPeer([])
    AgentChannel(peer).send_attack_move(attack_move_order(unit_id=7, x=990.0, y=2010.0))
    assert peer.sent == ['{"kind":"attack_move","unit_id":7,"x":990.0,"y":2010.0}']
