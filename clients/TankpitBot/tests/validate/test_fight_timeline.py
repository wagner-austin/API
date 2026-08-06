"""Human episodes and fight rows — built on the real extraction.

Every fixture goes through ``extract_shadow_timeline`` over encoded
wire frames (the same decode path production runs); nothing is faked
below the timeline.
"""

from __future__ import annotations

from tankpit_bot.validate.fight_timeline import (
    classify_fuel_delta,
    extract_human_episodes,
    render_fight_rows,
)
from tankpit_bot.validate.shadow_timeline import ShadowTimelineDict, extract_shadow_timeline
from tests.validate.builders import (
    aimed_shot_message,
    deactivation_message,
    identity_message,
    make_session,
    movement_response_message,
    named_identity_message,
    short_sync_message,
    sync_message,
)

_SELF = 1301
_HUMAN = 2678
_BOT = 510
_T0 = 1_785_000_000_000


def _duel_timeline() -> ShadowTimelineDict:
    """Self trades with human ``nope`` from one tile; a bot watches.

    The human fires four shots over four ticks; self returns three
    from the SAME source tile (streak 3), then one from a new tile.
    The human kills self at the end.
    """
    messages = [
        identity_message(_T0, _SELF),
        named_identity_message(_T0 + 1, _HUMAN, "nope"),
        named_identity_message(_T0 + 2, _BOT, "orange-2"),
        sync_message(_T0 + 3, _SELF, 1, 1100),
        # Non-self and short-form syncs exercise the fuel loop's skip
        # branches: only the self tank's absolute readings render.
        sync_message(_T0 + 4, _HUMAN, 1, 900),
        short_sync_message(_T0 + 5, _SELF, 1),
        aimed_shot_message(_T0 + 2_000, _HUMAN, (200, 142), (201, 143), 0),
        aimed_shot_message(_T0 + 2_100, _SELF, (201, 143), (200, 142), 1),
        sync_message(_T0 + 2_200, _SELF, 1, 1010),
        aimed_shot_message(_T0 + 4_000, _HUMAN, (199, 141), (201, 143), 0),
        aimed_shot_message(_T0 + 4_100, _SELF, (201, 143), (199, 141), 1),
        aimed_shot_message(_T0 + 6_000, _HUMAN, (198, 140), (201, 143), 1),
        aimed_shot_message(_T0 + 6_100, _SELF, (201, 143), (198, 140), 3),
        movement_response_message(_T0 + 7_000, _SELF, 202, 144),
        aimed_shot_message(_T0 + 8_000, _HUMAN, (198, 140), (202, 144), 0),
        aimed_shot_message(_T0 + 8_100, _SELF, (202, 144), (198, 140), 3),
        deactivation_message(_T0 + 9_000, _SELF, _HUMAN),
    ]
    return extract_shadow_timeline(make_session(messages))


class TestHumanEpisodes:
    def test_duel_yields_one_episode_with_exact_counts(self) -> None:
        """The human's window, both shot counts, and the death land."""
        episodes = extract_human_episodes(_duel_timeline())
        assert len(episodes) == 1
        episode = episodes[0]
        assert episode["name"] == "nope"
        assert episode["tank_id"] == _HUMAN
        assert episode["shots_by_human"] == 4
        assert episode["our_shots_in_window"] == 3
        assert episode["deaths_to_human"] == 1
        assert episode["kills_of_human"] == 0
        assert episode["first_shot_ms"] == _T0 + 2_000
        assert episode["last_shot_ms"] == _T0 + 8_000

    def test_stationary_streak_counts_consecutive_same_tile_shots(self) -> None:
        """Three shots from (201,143) then one from (202,144): streak 3."""
        episodes = extract_human_episodes(_duel_timeline())
        assert episodes[0]["max_stationary_streak"] == 3

    def test_kill_only_human_yields_a_windowless_episode(self) -> None:
        """A human we killed without them ever firing still appears."""
        messages = [
            identity_message(_T0, _SELF),
            named_identity_message(_T0 + 1, _HUMAN, "Belton"),
            deactivation_message(_T0 + 5_000, _HUMAN, _SELF),
        ]
        episodes = extract_human_episodes(extract_shadow_timeline(make_session(messages)))
        assert len(episodes) == 1
        assert episodes[0]["kills_of_human"] == 1
        assert episodes[0]["shots_by_human"] == 0
        assert episodes[0]["max_stationary_streak"] == 0

    def test_own_tank_is_never_an_opponent(self) -> None:
        """A named self tank must not appear as a human episode."""
        messages = [
            named_identity_message(_T0, _SELF, "Artax"),
            named_identity_message(_T0 + 1, _HUMAN, "nope"),
            aimed_shot_message(_T0 + 2_000, _SELF, (10, 10), (11, 11), 0),
            aimed_shot_message(_T0 + 4_000, _HUMAN, (12, 12), (10, 10), 0),
        ]
        episodes = extract_human_episodes(extract_shadow_timeline(make_session(messages)))
        assert [episode["name"] for episode in episodes] == ["nope"]

    def test_bystander_human_and_bots_yield_no_episode(self) -> None:
        """A greeted human with no engagement and a firing bot: nothing."""
        messages = [
            identity_message(_T0, _SELF),
            named_identity_message(_T0 + 1, _HUMAN, "Belton"),
            named_identity_message(_T0 + 2, _BOT, "orange-2"),
            aimed_shot_message(_T0 + 2_000, _BOT, (10, 10), (11, 11), 0),
        ]
        episodes = extract_human_episodes(extract_shadow_timeline(make_session(messages)))
        assert episodes == []


class TestFuelCauses:
    def test_measured_cost_table(self) -> None:
        """Each measured delta maps to its cause bucket."""
        assert classify_fuel_delta(-10) == "own_shot"
        assert classify_fuel_delta(-45) == "incoming_single_or_homing"
        assert classify_fuel_delta(-90) == "incoming_dual"
        assert classify_fuel_delta(129) == "gain"
        assert classify_fuel_delta(-291) == "spend_or_multi"
        assert classify_fuel_delta(-3) == "walk_or_misc"


class TestFightRows:
    def test_rows_cover_every_kind_in_timestamp_order(self) -> None:
        """Shots, fuel causes, positions, and the kill all render."""
        rows = render_fight_rows(_duel_timeline(), _T0, _T0 + 9_000)
        kinds = [row["kind"] for row in rows]
        assert "shot" in kinds and "fuel" in kinds
        assert "position" in kinds and "kill" in kinds
        stamps = [row["timestamp_ms"] for row in rows]
        assert stamps == sorted(stamps)
        fuel_rows = [row for row in rows if row["kind"] == "fuel"]
        assert fuel_rows[0]["description"] == "fuel 1100 -> 1010 (incoming_dual)"
        kill_rows = [row for row in rows if row["kind"] == "kill"]
        assert kill_rows == [
            {
                "timestamp_ms": _T0 + 9_000,
                "kind": "kill",
                "actor": "self",
                "description": "DEACTIVATED by nope",
            }
        ]

    def test_window_excludes_outside_events_but_tracks_prior_fuel(self) -> None:
        """A window starting mid-session still attributes the first
        in-window fuel delta against the pre-window reading."""
        timeline = _duel_timeline()
        rows = render_fight_rows(timeline, _T0 + 2_150, _T0 + 7_500)
        assert all(_T0 + 2_150 <= row["timestamp_ms"] <= _T0 + 7_500 for row in rows)
        assert [row["kind"] for row in rows if row["kind"] == "kill"] == []

    def test_unnamed_tank_renders_by_id(self) -> None:
        """A shooter with no identity message labels as ``tank <id>``."""
        messages = [
            identity_message(_T0, _SELF),
            aimed_shot_message(_T0 + 2_000, 999, (10, 10), (11, 11), 0),
        ]
        timeline = extract_shadow_timeline(make_session(messages))
        rows = render_fight_rows(timeline, _T0, _T0 + 3_000)
        assert rows[0]["actor"] == "tank 999"
