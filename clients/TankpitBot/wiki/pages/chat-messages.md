---
title: Chat Messages
tags: [js-client, chat, protocol]
related:
  - "[[client-commands]]"
  - "[[client-constants]]"
source_paths:
  - "tpclient.js:243"
  - "tpclient.js:24"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (all 65 entries traced from JS E[] object)
hubs: [js-client]
---

# Chat Messages

The game has exactly 65 predefined chat messages (E[0] through E[64]). There is no free-text chat — only these preset messages can be sent. Each message has targeting rules and voice recognition keywords.

## Message Properties

Each message is a `J(id, text, team_filter, has_position, is_visible, has_voice, voice_keywords)`:

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

These exist in the E[] table but don't appear in the chat selector list (`m=false`).

## Chat Display Order

The `li` array (line 243) defines the display order in the selector, NOT the ID order:

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

Sent via Hb command (code `m`, 4-6 bytes):
```
[6, 'm', message_id, x, y, use_position]
or
[4, 'm', message_id, use_position]
```

Position-bearing messages use the 6-byte format with sender's current coordinates.

## Position-Bearing Messages

These messages include the sender's world coordinates when sent:
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
