---
title: Fingerprint Algorithm
tags: [js-client, auth, fingerprint]
related:
  - "[[connection-protocol]]"
  - "[[xor-cipher]]"
source_paths:
  - tpclient.js lines 17-21 (sb class, MurmurHash3)
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (complete fingerprint + hash algorithm traced from JS)
hubs: [js-client]
---

# Fingerprint Algorithm

The game client generates a browser fingerprint during authentication using a MurmurHash3-based algorithm. This fingerprint is sent in the AUTH message.

## Data Collection (sb.get, line 18-21)

The fingerprint collects these browser properties in order:

1. `navigator.userAgent`
2. `navigator.language`
3. `screen.colorDepth`
4. `screen.height + "x" + screen.width` (as "HxW" string)
5. `new Date().getTimezoneOffset()`
6. `!!window.sessionStorage` (boolean, true/false)
7. `!!window.localStorage` (boolean, true/false)
8. `!!window.indexedDB` (boolean, true/false)
9. `typeof document.body.addBehavior` (for IE detection)
10. `typeof window.openDatabase` (for WebSQL detection)
11. `navigator.cpuClass` (IE/Edge only)
12. `navigator.platform`
13. `navigator.doNotTrack`
14. Plugin string (see below)
15. Canvas fingerprint (see below)

All values are joined with `"###"` separator into a single string.

## Plugin Enumeration (ub function, line 22)

```javascript
function ub(a) {
  return a.map(navigator.plugins, function(b) {
    var c = this.map(b, function(d) {
      return [d.type, d.suffixes].join("~");
    }, this).join(",");
    return [b.name, b.description, c].join("::");
  }, a).join(";");
}
```

Format: `name::description::type1~suffix1,type2~suffix2;name2::...`

## Canvas Fingerprint (lines 19-21)

Only generated if canvas 2D context is available:

```javascript
var c = document.createElement("canvas");
var d = c.getContext("2d");
d.textBaseline = "top";
d.font = "14px 'Arial'";
d.textBaseline = "alphabetic";
d.fillStyle = "#f60";
d.fillRect(125, 1, 62, 20);
d.fillStyle = "#069";
d.fillText("http://valve.github.io", 2, 15);
d.fillStyle = "rgba(102, 204, 0, 0.7)";
d.fillText("http://valve.github.io", 4, 17);
c = c.toDataURL();
```

The canvas data URL is appended to the fingerprint array.

## MurmurHash3 (lines 20-21)

The final fingerprint string is hashed using MurmurHash3 (32-bit):

```javascript
// Input: concatenated string "a"
var b = a.length & 3;
var c = a.length - b;
var e = 31;  // seed

for (d = 0; d < c; ) {
  var f = a.charCodeAt(d) & 255 |
          (a.charCodeAt(++d) & 255) << 8 |
          (a.charCodeAt(++d) & 255) << 16 |
          (a.charCodeAt(++d) & 255) << 24;
  ++d;
  f = 3432918353 * (f & 65535) + ((3432918353 * (f >>> 16) & 65535) << 16) & 4294967295;
  f = f << 15 | f >>> 17;
  f = 461845907 * (f & 65535) + ((461845907 * (f >>> 16) & 65535) << 16) & 4294967295;
  e ^= f;
  e = e << 13 | e >>> 19;
  e = 5 * (e & 65535) + ((5 * (e >>> 16) & 65535) << 16) & 4294967295;
  e = (e & 65535) + 27492 + (((e >>> 16) + 58964 & 65535) << 16);
}

// Tail handling
f = 0;
switch (b) {
  case 3: f ^= (a.charCodeAt(d + 2) & 255) << 16;
  case 2: f ^= (a.charCodeAt(d + 1) & 255) << 8;
  case 1: f ^= a.charCodeAt(d) & 255;
    f = 3432918353 * (f & 65535) + ((3432918353 * (f >>> 16) & 65535) << 16) & 4294967295;
    f = f << 15 | f >>> 17;
    e ^= 461845907 * (f & 65535) + ((461845907 * (f >>> 16) & 65535) << 16) & 4294967295;
}

// Finalization
e ^= a.length;
e ^= e >>> 16;
e = 2246822507 * (e & 65535) + ((2246822507 * (e >>> 16) & 65535) << 16) & 4294967295;
e ^= e >>> 13;
e = 3266489909 * (e & 65535) + ((3266489909 * (e >>> 16) & 65535) << 16) & 4294967295;
return "" + ((e ^ e >>> 16) >>> 0);  // unsigned 32-bit result as string
```

Constants:
- Seed: 31
- c1: 0xCC9E2D51 (3432918353)
- c2: 0x1B873593 (461845907)
- Finalization mix: 0x85EBCA6B (2246822507), 0xC2B2AE35 (3266489909)

These are standard MurmurHash3 constants.

## AUTH Message Format

The fingerprint is sent in the AUTH command (wa class, line 6):

```javascript
wa.prototype.toString = function() {
  return this.code + "AUTH !" + this.version + " " + this.i + "|" + this.j + " " + this.h;
};
// Result: "%AUTH !be {user_id}|{fingerprint} {magic}"
```

- `this.code` = `"%"` 
- `this.version` = `"be"` (hardcoded)
- `this.i` = `tankpit.user_id`
- `this.j` = fingerprint hash (MurmurHash3 output as decimal string)
- `this.h` = `tankpit.magic` (session key)

## Error Report Fingerprinting

If there's a pending error (Ng variable), it's also sent after AUTH:

```javascript
// Line 214:
Ng && Xa(oa, Da(new Ca(10, Ng)));
```

This sends any captured error as a Ca (error report) message with error code 10.

## Canvas Fingerprint Flag

The fingerprint class has a toggle:
```javascript
function sb() { this.h = true; }  // canvas fingerprinting enabled by default
```

`this.h` controls whether the canvas fingerprint step is included. Always true in the shipped code.
