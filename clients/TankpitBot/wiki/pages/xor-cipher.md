---
title: XOR Cipher
tags: [js-client, protocol, crypto]
related:
  - "[[js-source-map]]"
  - "[[client-commands]]"
source_paths:
  - tpclient.js lines 16-17 (qb table generation, za function)
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (cipher implementation traced through JS)
hubs: [js-client]
---

# XOR Cipher

The game uses a simple XOR cipher for binary messages. The key table is derived from a hardcoded string XORed with a per-session `tankpit.magic` value.

## Key Table Generation (line 16)

```javascript
var pb = tankpit.magic;  // per-session magic string from server HTML
var qb = new Uint8Array(1000);
for (var rb = 0; 1000 > rb; rb++)
  qb[rb] = "<hardcoded 1000-char string>".charCodeAt(rb) ^ pb.charCodeAt(rb % pb.length);
```

The hardcoded string is a fixed 1000-character sequence embedded in tpclient.js:
```
"Y1DcZyIAudY},fSP$:|[xH~r!U&^z?1V<sg%8*^cn*QAXf^:CE61[n*<+vGEhet'/R~e&AC8YH~Xqi!|}L.io{=jV..."
```

Each byte of `qb` = `hardcoded_char[i] XOR magic_char[i % magic.length]`.

The `magic` value changes per page load — it's injected server-side into the HTML as `tankpit.magic`.

## Encode/Decode Function (za, line 17)

```javascript
function za(a, b) {
  var c = qb;  // the 1000-byte key table
  "undefined" === typeof b && (b = a.length);
  var d = c.length;
  for (var e = 0; e < b; e++)
    a[e] ^= c[e % d];
  return a;
}
```

- `a` = byte array to encode/decode (modified in-place)
- `b` = optional length (defaults to a.length)
- XOR is symmetric: same function encodes and decodes
- Key table wraps at 1000 bytes

## Which Messages Use XOR

### Server→Client (inbound)

The binary message dispatch (line 217, case 46 in main switch):
```javascript
case 46:  // 0x2E = '.' — binary game message container
  if (Fa) {  // only when in-game
    var u = W.ta;
    for (var B = qb, T = l.length, pa = B.length, ja = 0; ja < T; ja++)
      l[ja] ^= B[ja % pa];  // XOR decode
    (l = Mf(l)) && ve(u, l);  // parse via V table
  }
```

All binary game messages arrive inside a type-46 (0x2E) container and are XOR-decoded before parsing.

### Client→Server (outbound)

In `I(a, b)` (line 26 — the main command send function):
```javascript
function I(a, b) {
  if (a.h) {
    b = b.h();             // serialize command
    za(b, b[0]);           // XOR encode using length byte as limit
    // ... wrap in Aa, then Xa()→send
  }
}
```

The XOR uses `b[0]` (the length prefix byte) as the encode length. So only the first `length` bytes are XORed, not the trailing padding.

### Exceptions — Text Messages (NOT XOR encoded)

These bypass XOR entirely, sent as raw text:
- `%` AUTH
- `*` Game select
- `+` Join game
- `-` Quit
- `~` Disconnect
- `` ` `` Ping
- `R` Supervisor responses

Server→client text messages also bypass XOR — they arrive on different container codes (43=`+`, 45=`-`, etc.) in the main dispatch switch.

The hardcoded string in tpclient.js is fixed (baked into the compiled JS). Only `tankpit.magic` changes per session — it's injected server-side into the HTML.
