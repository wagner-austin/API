---
title: Chat Messages
tags: [js-client, chat, protocol]
related:
  - "[[client-commands]]"
  - "[[client-constants]]"
source_paths:
  - "tpclient.js:243"
  - "tpclient.js:24"
fact_checked: "2026-07-29"
confidence: high
verified: 2026-07-29 (65 entries from JS E[] 2026-06-19; wire format + M echo + flood mute live-verified sniff-20260729-214411)
hubs: [js-client]
---

# Chat Messages

The game has exactly 65 predefined chat messages (E[0] through E[64]). There is no free-text chat — only these preset messages can be sent. Each message has targeting rules and voice recognition keywords.[^1]

## Message Properties

Each message is a `J(id, text, team_filter, has_position, is_visible, has_voice, voice_keywords)`:[^1]

- **team_filter (h)**: 0=same team only, 1=allies in zone, 2=zone-wide, 3=all players
- **has_position (l)**: true=includes sender's coordinates in the message
- **is_visible (m)**: true=appears in the chat message selector list
- **has_voice (i)**: true=can be triggered by voice recognition

## Complete Message Table

### Team Commands (filter=0, same team only)

| ID | Text | Position | Voice Keywords |
|----|------|----------|----------------|
| 0 | "Attack the red" | no | attack red, taxi red |
| 1 | "Attack the purple" | no | attack purple, taxi purple |
| 2 | "Attack the blue" | no | attack blue, taxi blue |
| 3 | "Attack the orange" | no | attack orange, taxi orange |
| 17 | "Follow me" | no | follow me |
| 19 | "Out of the way" | no | out way, out of the |
| 20 | "Stop shooting" | no | stop shooting |
| 21 | "Retreat" | no | retreat, get back, go back |
| 22 | "Let's team up" | no | lets team, whats team, team up |
| 23 | "Make base" | no | make base, make baby |
| 24 | "Move obstacle" | no | move obstacle, move optical |
| 25 | "Build bridge" | no | build bridge, built bridge |
| 26 | "Plant mines" | no | plant mines, plant mine, lay mines |
| 27 | "Blow up mines" | no | blow mines, blow mine, blow mind |
| 28 | "Use the radar" | no | use radar, us radar, news radar |
| 30 | "Hold on" | no | hold, old, bolt |
| 31 | "Getting fuel" | no | getting fuel, in fuel |
| 32 | "Getting equipment" | no | getting equipment, getting eq, in equipment, hitting equipment |
| 33 | "Sure!" | no | sure |
| 34 | "No way!" | no | no way |
| 43 | "Let's chill here for a while" | no | lets chill, chill here, for while |
| 51 | "Move obstacle off of ferry" | no | move obstacle, obstacle ferry, obstacle fairy |
| 52 | "Don't follow me!" | no | don't follow, leave me alone |
| 56 | "Put that here." | no | put that there, put that down |

### Ally/Zone Messages (filter=1, allies + zone)

| ID | Text | Position | Voice Keywords |
|----|------|----------|----------------|
| 4 | "HELP - Enemy!" | **yes** | help enemy, help me |
| 6 | "HELP - Fuel low!" | **yes** | help fuel, fuel low |
| 7 | "I'll help you" | no | help you |
| 8 | "Fuel detected here" | **yes** | fuel detected, fuel here |
| 9 | "Equipment detected here" | **yes** | equipment detected, equipment here, eq detected, eq here |
| 12 | "Base is here" | **yes** | base here, basis here, beast here, bass here |
| 13 | "Enemy base is here" | **yes** | enemy base, enemy bass |
| 14 | "Ferry located here" | **yes** | ferry located, ferry here |
| 15 | "Meet me" | **yes** | meet me, meet here |
| 53 | "I need equipment!" | **yes** | need equipment |
| 54 | "I need fuel!" | **yes** | need fuel |

### Zone-Wide Messages (filter=2, all in zone)

| ID | Text | Position | Voice Keywords |
|----|------|----------|----------------|
| 10 | "Thanks" | no | thanks, thank you |
| 11 | "No problem" | no | no problem |
| 18 | "Buzz off!" | no | buzz off, bug off, fuck off, go away |
| 35 | "Congrats" | no | congrats, congratulations |
| 36 | "You've got mad skills!" | no | got skills, mad skills |
| 37 | "My bad" | no | my bad, oops, sorry |
| 38 | "That was mine" | no | was mine |
| 60 | "Check the bulletin board." | no | check bulletin, check board, check forum |
| 61 | "Good job" | no | good job, good work |
| 62 | "You're Welcome" | no | you're welcome |

### All-Player Messages (filter=3, everyone)

| ID | Text | Position | Voice Keywords |
|----|------|----------|----------------|
| 5 | "Bring it!" | no | bring |
| 16 | "Come get me!" | no | come get, come me, get me |
| 29 | "Charge" | no | charge |
| 39 | "Long time no see" | no | long time |
| 40 | "Be right back" | no | be back, right back |
| 41 | "HELLO" | no | hello |
| 42 | "BYE" | no | bye |
| 44 | "That was whack!" | no | whack, wack, that black, that was what, that quack |
| 45 | "Whatever" | no | whatever |
| 46 | "Do your worst!" | no | your worst, bring it |
| 47 | "Don't Cry." | no | don't cry |
| 48 | "Is that the best you can do?" | no | the best you can do, that all you got |
| 49 | "My dog plays better than you!" | no | my dog better, you suck |
| 50 | "Nice try!" | no | nice try, nice one, nice take |
| 55 | "I gotta go." | no | gotta go, have to go |
| 57 | "Lame!" | no | lame |
| 58 | "Who's your daddy?" | no | who's your daddy, who your daddy |
| 59 | "I rule!" | no | i rule, irule |
| 63 | "I'm playing TankPit, mom." | no | tank pit mom, playing mom (+ many variants) |
| 64 | "I'm playing TankPit, dad." | no | tank pit dad, playing dad (+ many variants) |

### Hidden Messages (is_visible=false)

| ID | Text | Visible | Voice |
|----|------|---------|-------|
| 5 | "Bring it!" | **no** | yes |
| 29 | "Charge" | **no** | no |
| 63 | "I'm playing TankPit, mom." | **no** | yes |
| 64 | "I'm playing TankPit, dad." | **no** | yes |

These exist in the E[] table but don't appear in the chat selector list (`m=false`).[^1]

## Chat Display Order

The `li` array (line 243) defines the display order in the selector, NOT the ID order:[^1]

```
[0,1,2,3,12,40,27,25,18,42,60,16,35,46,47,52,13,9,14,17,8,31,32,61,41,4,6,30,55,53,54,59,7,48,57,43,22,39,23,15,24,51,37,49,50,11,34,19,26,56,21,20,33,10,38,44,28,45,58,36,62]
```

## Chat Controller (Bb class)

- **Cooldown**: 2400ms between sends
- **Queue depth**: Max 2 messages queued
- **Duplicate suppression**: Won't queue if same bytes as last message (`Fb(a.j[a.j.length-1], b)`)
- **Target validation**: Before sending, checks zone conditions:
  - filter=0 (team): requires teammate in 17×17 viewport
  - filter=2 (zone): requires ANY other tank in 17×17 viewport
  - filter=1,3: no target check

## Chat Wire Format

Sent via Hb command (code `m`, 4-6 bytes):[^1]
```
[6, 'm', message_id, x, y, use_position]
or
[4, 'm', message_id, use_position]
```

Position-bearing messages use the 6-byte format with sender's current coordinates.[^1]

## Wire-Verified (sniff-20260729-214411, 44 live sends)

Live capture of every selector message clicked at least once
(37 unique IDs reached the wire) settles the format:[^2]

- **Every observed send used the 6-byte form** `[6,'m',id,x,y,flag]`
  — including non-position messages like 41 HELLO. The 4-byte
  variant never appeared on the wire.
- **x,y = sender's current tile** for ordinary messages; for the
  auto-search messages the client substitutes the found target
  (id 8 "Fuel detected here" went out as `[8,104,212,0]` while the
  tank sat at 97,212 — the Db() nearest-fuel tile).
- **flag (byte 6) was 0 in all 44 sends.** The wiki's
  "use_position" reading is unconfirmed; no send ever set it.
- Sends are XOR-encoded exactly like every other binary command
  (`!` prefix + session-table XOR); the same table decodes them.
- **Server echo = the DOM display.** Each accepted chat comes back
  as inbound `M` (0x4D, [[decode-coverage]] row Qg):
  `M + tank_id(2 LE) + message_id + x + y`. The client's
  "Message sent:" log line follows the echo, not the local click.

### Server-side flood mute (NEW)

The first **8** messages (sent at the client's 2400 ms cooldown
pace) were echoed and displayed. **Every one of the remaining 36
sends was silently swallowed** — no `M` echo, no error frame —
including sends made 2+ minutes later, while other commands
(teleport, move, radar) kept working. Rapid-fire chat triggers a
server-side mute that lasted the rest of the session. Continued
sending may have kept re-arming it; the exact decay is unknown.
**Bot rule: chat must be rare and never retried on silence.**[^2]

### Client-side gate observed

Team-filter (h=0) messages clicked while solo printed
"No teammate in the zone" to the DOM and produced **no wire
frame** — the Bb target validation drops them before send.[^2]

## Bot Implementation (2026-07-29)

The bot speaks and hears chat as of 2026-07-29 (same session as the
wire crack):[^impl]

- **Outbound**: `protocol/chat.py` — the full 65-entry
  `CHAT_MESSAGES` table, `CHAT_HELLO = 41`, and
  `build_chat_command(message_id, x, y)` producing the plaintext
  frame `! 06 6D id x y 00`; the standard send path XORs it into the
  exact bytes the page client sends. Dispatch surface:
  `DispatchMixin.send_chat` (fire-and-forget, no HFSM transition,
  `chat_sent` diagnostic) behind the `"chat"` BotCommand — usable as
  a decision `secondary_command` so it rides the same tick as a
  primary action.
- **Inbound**: the 0x2E envelope router
  (`protocol/decoders/tank.py::_dispatch_protocol_misc`) now routes
  subtype 0x4D (>= 5 inner bytes) to `decode_chat_message`;
  the world-state dispatcher emits a `chat_received` diagnostic with
  the resolved preset text and, when `sender_id` is the self tank,
  latches `WorldService.last_chat_echo_message_id` — the delivery
  receipt against the flood mute. Chat x/y is sender-supplied and
  never mutates tank positions.
- **Behavior**: one HELLO per newly locked human target
  (`bot/ai/greeting.py`, contract row [[bot-behavior-contract]]
  §3.2), latched in `ai_state["greeted_target_id"]`; never retried
  on silence.
- **Sim**: the sim server decodes `m` sends and echoes the 0x4D
  broadcast to everyone including the sender (no mute modeled —
  bot policy keeps live sends far below the threshold).

[^2]: `runs/sniff/sniff-20260729-214411.capture_session.json` —
    44 decoded `m` sends, 8 inbound `M` echoes, tank_id 1301,
    solo practice-room session of 2026-07-29; decoded with the
    session magic + `xor_static_key.txt` table
    (`capture/xor.py::build_xor_table`).

## Position-Bearing Messages

These messages include the sender's world coordinates when sent:[^1]
- **4** "HELP - Enemy!" 
- **6** "HELP - Fuel low!"
- **8** "Fuel detected here"
- **9** "Equipment detected here"
- **12** "Base is here"
- **13** "Enemy base is here"
- **14** "Ferry located here"
- **15** "Meet me"
- **53** "I need equipment!"
- **54** "I need fuel!"

[^1]: JS truth: `tpclient.js` on disk — the `E[]` message table (line 24, frontmatter-pinned), the `li` display-order array (line 243, frontmatter-pinned), and the `Hb` send / `Bb` controller classes; all 65 entries traced 2026-06-19 (frontmatter `verified:` field) and re-checkable by grep against the file.

[^impl]: Implementation commits 2026-07-29: `a8d015ad` (protocol table + 0x4D tunnel route + outbound builder, wire-verified byte-identical against sniff-20260729-214411's 44 decoded sends), `58b8644a` (inbound dispatch + self-echo latch), `8e586ba6`/`e6eea852` (HELLO greeting + fire-and-forget executor route), `b4dbf0c8` (sim chat law); gate green at 5,339 tests / 100% coverage the same day.
