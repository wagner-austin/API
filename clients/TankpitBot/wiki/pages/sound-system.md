---
title: Sound System
tags: [js-client, audio]
related:
  - "[[js-source-map]]"
  - "[[client-constants]]"
source_paths:
  - "tpclient.js:135"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (all 18 audio buffers and their triggers traced from JS)
hubs: [js-client]
---

# Sound System

The game uses Web Audio API (AudioContext) with 18 sound effects, loaded from base64-encoded globals.[^1]

## Audio Architecture (Sc class, line 135)

```
Sc.j        = AudioContext instance
Sc.o        = GainNode (master volume)
Sc.source   = one-shot sound source (current)
Sc.m        = looping sound source (ambient)
Sc.i        = looping sound source 2 (drive/ferry)
Sc.h        = Zf instance (all 18 audio buffers)
Sc.l        = sound enabled flag
Sc.volume   = volume level (0-100)
```

Three playback channels:
1. **One-shot** (`source`): plays a sound once, stops previous one-shot
2. **Loop 1** (`m`): ambient loop (e.g., fuel deposit music)
3. **Loop 2** (`i`): movement loop (drive or ferry sound)[^1]

## Audio Buffers (Zf class)

All buffers loaded from global base64 variables via `AudioContext.decodeAudioData()`:[^1]

| Field | Global Variable | Trigger | Type |
|-------|----------------|---------|------|
| h.h | `audio_click` | UI click / move command | one-shot |
| h.j | `audio_wrong` | Error / invalid action | one-shot |
| h.P | `audio_drive` | Tank driving (looped while moving) | loop |
| h.s | `audio_ferry` | Ferry movement (looped, per-field variant) | loop |
| h.U | `audio_radar` | Radar sweep activation | one-shot |
| h.ba | `audio_shot` | Shoot command initiated | one-shot |
| h.i | `audio_explosion` | Explosion (hit, mine detonate) | one-shot |
| h.W | `audio_mine` | Mine placement | one-shot |
| h.u | `audio_grab` | Container pickup (fuel/equipment) | one-shot |
| h.l | `audio_equip` | Equipment gain | one-shot |
| h.m | `audio_depo` | Fuel deposit (looped while depositing) | loop |
| h.$ | `audio_hoist` | Obstacle hoist/carry pickup | one-shot |
| h.Y | `audio_build` | Obstacle build/drop | one-shot |
| h.o | `audio_error` | Command error (supervisor rejection) | one-shot |
| h.v | `audio_deactivate` | Tank deactivated (death) | one-shot |
| h.da | `audio_recharge` | Recharging/repairing | one-shot |
| h.aa | `audio_reactivate` | Tank reactivated (respawn) | one-shot |
| h.ca | `audio_splash` | Water splash (falling in water) | one-shot |

## Ferry Sound Variants

The ferry sound has per-field variants:[^1]

```javascript
// Line 136:
"undefined" === typeof audio_ferry[b] && (b = 0);
a.j.decodeAudioData(q(window.atob(audio_ferry[b])).buffer, ...);
```

The `audio_ferry` global is an array indexed by some field property. Falls back to index 0 if the variant doesn't exist.[^1]

## Sound Playback Functions

### N(a, b) — One-shot play (line 140)
Stops any current one-shot, creates new BufferSource, plays once:
```javascript
function N(a, b) {
  Cd(a);  // stop current
  if (null !== b && a.l && !a.u) {
    a.source = a.j.createBufferSource();
    a.source.connect(a.o);
    a.source.buffer = b;
    a.source.start(0);
  }
}
```

### Zd(a, b) — Loop play (line 139)
Stops current loop, creates new BufferSource with `loop=true`:
```javascript
function Zd(a, b) {
  kd(a);  // stop current loop
  if (null !== b && a.l && !a.u) {
    a.m = a.j.createBufferSource();
    a.m.loop = true;
    a.m.connect(a.o);
    a.m.buffer = b;
    a.m.start(0);
  }
  a.s = b;  // store for resume
}
```

### Ve(a, b) — Loop 2 play (line 139)
Same as Zd but for the second loop channel (drive/ferry).

## When Each Sound Plays

| Event | Sound | Channel |
|-------|-------|---------|
| Click on valid tile | h.h (click) | one-shot |
| Click on invalid tile | h.j (wrong) | one-shot |
| Shoot command sent | h.ba (shot) | one-shot |
| Projectile hits target | h.i (explosion) | one-shot |
| Mine detonates | h.i (explosion) | one-shot |
| Mine placed | h.W (mine) | one-shot |
| Radar activated | h.U (radar) | one-shot |
| Fuel/equipment pickup | h.u (grab) | one-shot |
| Equipment gained | h.l (equip) | one-shot |
| Fuel deposit started | h.m (depo) | loop 1 |
| Obstacle picked up | h.$ (hoist) | one-shot |
| Obstacle dropped/bridge built | h.Y (build) | one-shot |
| Supervisor error | h.o (error) | one-shot |
| Own tank deactivated | h.v (deactivate) | one-shot |
| Repairing after death | h.da (recharge) | one-shot |
| Reactivated | h.aa (reactivate) | one-shot |
| Water splash | h.ca (splash) | one-shot |
| Tank driving | h.P (drive) or h.s (ferry) | loop 2 |

## Volume Control

Volume is controlled via a GainNode:[^1]
```javascript
a.o.gain.value = a.volume / 100;
```

Range: 0 (mute) to 100 (full). Changed via Ha command (code `V`) sent to server for persistence.[^1]

## Sound Toggle

Sound on/off state is tracked by `this.l` flag. When toggled:
- Off: stops all three channels (one-shot, loop 1, loop 2)
- On: resumes stored loops (`s` for loop 1, `P` for loop 2)[^1]

Toggled via Ka command (code `C`).[^1]

## Unsupported Browsers

If `AudioContext` is undefined or `window.atob` is missing:[^1]
```javascript
this.u = true;  // "unsupported" flag
```

All playback functions check `!a.u` before playing — silently no-ops on unsupported browsers.[^1]

[^1]: JS truth: `tpclient.js` on disk (frontmatter-pinned `tpclient.js:135`) — `Sc` audio class (line 135), `Zf` buffers, play functions `N`/`Zd`/`Ve` (lines 139-140), ferry-variant fallback (line 136), all quoted verbatim in the fences above; all 18 buffers and triggers traced 2026-06-19 (frontmatter `verified:` field), re-checkable by grep.
