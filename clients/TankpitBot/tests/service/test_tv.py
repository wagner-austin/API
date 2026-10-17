"""Casting to the TV, run against a real renderer on the loopback interface.

Every test here sends real HTTP through the production seam to
:class:`tests.service._tv_server.TvServer`, which answers as the
operator's Samsung did on 2026-10-17, faults as a UPnP renderer does,
and stalls as a TV in standby does. Nothing in the request path is
stood in for.
"""

from __future__ import annotations

import socket
from xml.etree import ElementTree

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service import tv
from tankpit_bot.service._test_hooks.tv import (
    SoapReplyDict,
    decode_soap_reply,
    encode_soap_reply,
)
from tankpit_bot.service.tv import (
    AV_TRANSPORT,
    TvError,
    TvStateDict,
    TvTargetDict,
    decode_tv_state,
    decode_tv_target,
    encode_tv_state,
    encode_tv_target,
    media_metadata,
    resolve_tv_target,
)
from tankpit_bot.stream.hls import PLAYLIST_CONTENT_TYPE
from tests.conftest import FakeEnv
from tests.service._tv_server import TvServer, fault_envelope

MEDIA_URL = "https://tankpit.example/demo/mosaic/index.m3u8?a=1&b=2"


def _target(server: TvServer) -> TvTargetDict:
    """Aim at a loopback renderer.

    Args:
        server: The renderer.

    Returns:
        The target.
    """
    return TvTargetDict(control_url=server.control_url, media_url=MEDIA_URL)


def _closed_port() -> int:
    """Return a loopback port nothing listens on.

    Returns:
        A port that was free a moment ago and is closed now.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    return port


class TestRealSeam:
    """The production POST, against a real server, in every outcome."""

    def test_the_post_carries_the_soap_headers_and_returns_the_answer(self) -> None:
        """The renderer sees the action header and the XML; the caller gets the reply."""
        with TvServer() as server:
            server.answer_state("PLAYING")
            reply = service_hooks._real_post_soap(
                server.control_url, f'"{AV_TRANSPORT}#GetTransportInfo"', b"<x/>", 5.0
            )

        assert reply["status"] == 200
        assert "<CurrentTransportState>PLAYING</CurrentTransportState>" in reply["body"]
        assert server.requests[0]["path"] == "/upnp/control/AVTransport1"
        assert server.requests[0]["soap_action"] == f'"{AV_TRANSPORT}#GetTransportInfo"'
        assert server.requests[0]["content_type"] == 'text/xml; charset="utf-8"'
        assert server.requests[0]["body"] == "<x/>"

    def test_a_fault_is_returned_with_its_body_not_raised(self) -> None:
        """A 500 is an answer to read, not an exception."""
        with TvServer() as server:
            server.answers["Play"] = (500, fault_envelope(701, "Transition not available"))
            reply = service_hooks._real_post_soap(
                server.control_url, f'"{AV_TRANSPORT}#Play"', b"<x/>", 5.0
            )

        assert reply["status"] == 500
        assert "<errorCode>701</errorCode>" in reply["body"]

    def test_a_tv_that_never_answers_times_out(self) -> None:
        """A renderer holding the connection open fails at the timeout."""
        with TvServer() as server:
            server.stall = True
            with pytest.raises(TimeoutError):
                service_hooks._real_post_soap(
                    server.control_url, f'"{AV_TRANSPORT}#Play"', b"<x/>", 0.3
                )

    def test_a_tv_that_is_off_refuses_the_connection(self) -> None:
        """Nothing listening is an OSError from the seam itself."""
        url = f"http://127.0.0.1:{_closed_port()}/upnp/control/AVTransport1"
        with pytest.raises(ConnectionRefusedError):
            service_hooks._real_post_soap(url, f'"{AV_TRANSPORT}#Play"', b"<x/>", 2.0)

    @pytest.mark.parametrize("url", ["https://127.0.0.1/x", "http:///x", "tv.local/x"])
    def test_a_url_that_is_not_plain_http_to_a_host_is_refused(self, url: str) -> None:
        """Renderers serve plain HTTP; anything else is a configuration error."""
        with pytest.raises(ValueError, match="http://host"):
            service_hooks._real_post_soap(url, '"x"', b"", 1.0)

    def test_a_url_without_a_path_posts_to_the_root(self) -> None:
        """A bare host is posted to ``/``."""
        with TvServer() as server:
            root = server.control_url.split("/upnp", 1)[0]
            reply = service_hooks._real_post_soap(root, f'"{AV_TRANSPORT}#Stop"', b"<x/>", 5.0)

        assert reply["status"] == 200
        assert server.requests[0]["path"] == "/"


class TestCast:
    """Load, play, and read back, as a phone's cast button does."""

    def test_cast_loads_the_stream_plays_it_and_reports_the_state(self) -> None:
        """Three actions in order; the URL travels escaped, with its media type."""
        with TvServer() as server:
            server.answer_state("PLAYING")
            state = tv.cast(_target(server))

        assert state == {"transport_state": "PLAYING", "media_url": MEDIA_URL}
        actions = [
            request["soap_action"].rsplit("#", 1)[-1].strip('"') for request in server.requests
        ]
        assert actions == ["SetAVTransportURI", "Play", "GetTransportInfo"]
        load = server.requests[0]["body"]
        assert (
            "<CurrentURI>https://tankpit.example/demo/mosaic/index.m3u8?a=1&amp;b=2</CurrentURI>"
            in load
        )
        assert f'&lt;res protocolInfo="http-get:*:{PLAYLIST_CONTENT_TYPE}:*"&gt;' in load
        assert "<Speed>1</Speed>" in server.requests[1]["body"]

    def test_stop_stops_and_reports_the_state(self) -> None:
        """Stop, then read back."""
        with TvServer() as server:
            server.answer_state("STOPPED")
            state = tv.stop(_target(server))

        assert state["transport_state"] == "STOPPED"
        actions = [
            request["soap_action"].rsplit("#", 1)[-1].strip('"') for request in server.requests
        ]
        assert actions == ["Stop", "GetTransportInfo"]

    def test_a_refused_action_names_the_upnp_error(self) -> None:
        """A fault becomes TV_REFUSED carrying the renderer's code and words."""
        with TvServer() as server:
            server.answers["SetAVTransportURI"] = (500, fault_envelope(716, "Resource not found"))
            with pytest.raises(TvError, match="UPnP error 716: Resource not found") as caught:
                tv.cast(_target(server))

        assert caught.value.code == "TV_REFUSED"
        assert len(server.requests) == 1

    def test_a_refusal_without_a_fault_body_still_says_what_failed(self) -> None:
        """An empty error answer is TV_REFUSED with no code to report."""
        with TvServer() as server:
            server.answers["Stop"] = (503, "<busy/>")
            with pytest.raises(TvError, match="UPnP error none: no description") as caught:
                tv.stop(_target(server))

        assert caught.value.code == "TV_REFUSED"

    def test_an_answer_that_is_not_xml_propagates_as_a_parse_error(self) -> None:
        """A web page where a fault belongs is not softened here; the route names it."""
        with TvServer() as server:
            server.answers["Play"] = (404, "<html><body>Not Found")
            with pytest.raises(ElementTree.ParseError):
                tv.cast(_target(server))

    def test_a_state_answer_without_a_state_is_unreadable(self) -> None:
        """GetTransportInfo must say what the TV is doing."""
        with (
            TvServer() as server,
            pytest.raises(TvError, match="without a CurrentTransportState") as caught,
        ):
            tv.cast(_target(server))

        assert caught.value.code == "TV_UNREADABLE"


class TestTarget:
    """The TV's address and the stream's URL, from the environment only."""

    def test_both_variables_make_the_target(self, fake_env: FakeEnv) -> None:
        """The two variables, as given."""
        fake_env.set("TANKPIT_TV_CONTROL_URL", "http://10.0.0.5:9197/upnp/control/AVTransport1")
        fake_env.set("TANKPIT_MOSAIC_URL", MEDIA_URL)

        assert resolve_tv_target() == {
            "control_url": "http://10.0.0.5:9197/upnp/control/AVTransport1",
            "media_url": MEDIA_URL,
        }

    @pytest.mark.parametrize("missing", ["TANKPIT_TV_CONTROL_URL", "TANKPIT_MOSAIC_URL"])
    def test_either_variable_missing_is_no_tv(self, fake_env: FakeEnv, missing: str) -> None:
        """No guessing: an unset address is TV_NOT_CONFIGURED."""
        for name in ("TANKPIT_TV_CONTROL_URL", "TANKPIT_MOSAIC_URL"):
            if name != missing:
                fake_env.set(name, "http://set")
        with pytest.raises(TvError, match="must both be set") as caught:
            resolve_tv_target()

        assert caught.value.code == "TV_NOT_CONFIGURED"


class TestCodecs:
    """Every TV shape round-trips through JSON, and a bad one is refused."""

    def test_a_target_round_trips(self) -> None:
        """Encode then decode is the identity."""
        target = TvTargetDict(control_url="http://tv/x", media_url=MEDIA_URL)
        assert decode_tv_target(encode_tv_target(target)) == target

    def test_a_state_round_trips(self) -> None:
        """Encode then decode is the identity."""
        state = TvStateDict(transport_state="PLAYING", media_url=MEDIA_URL)
        assert encode_tv_state(state) == {"transport_state": "PLAYING", "media_url": MEDIA_URL}
        assert decode_tv_state(encode_tv_state(state)) == state

    def test_a_soap_reply_round_trips(self) -> None:
        """Encode then decode is the identity."""
        reply = SoapReplyDict(status=500, body="<fault/>")
        assert decode_soap_reply(encode_soap_reply(reply)) == reply

    def test_missing_fields_are_refused(self) -> None:
        """Each decoder requires every field."""
        with pytest.raises(JSONTypeError):
            decode_tv_target({"control_url": "http://tv/x"})
        with pytest.raises(JSONTypeError):
            decode_tv_state({"media_url": MEDIA_URL})
        with pytest.raises(JSONTypeError):
            decode_soap_reply({"status": "500", "body": ""})


def test_the_metadata_describes_one_escaped_video_item() -> None:
    """The item's resource is the URL, escaped, offered as an HLS playlist."""
    metadata = media_metadata(MEDIA_URL)

    assert "<upnp:class>object.item.videoItem</upnp:class>" in metadata
    assert "<dc:title>TankPit live</dc:title>" in metadata
    assert (
        f'<res protocolInfo="http-get:*:{PLAYLIST_CONTENT_TYPE}:*">'
        "https://tankpit.example/demo/mosaic/index.m3u8?a=1&amp;b=2</res>"
    ) in metadata
