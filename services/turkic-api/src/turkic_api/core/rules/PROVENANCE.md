# Rule file provenance

`turkic-transliteration` is the source of truth for these rules. The files here
are a **vendored copy**, not a fork: nothing in this service edits them, and any
linguistic change is made upstream first and then re-vendored.

Vendoring rather than depending on the library is deliberate. This service ships
a pure-Python rule engine so that deployment needs no ICU C++ library, and it
carries no dependency on the upstream project — not at runtime, not in its test
suite. Depending on the published package would not be a smaller commitment:
`turkic-translit` requires `torch`, `transformers`, `gradio` and, on every
platform but Windows, `PyICU` — the last of which would reimpose the exact libicu
requirement the pure-Python engine exists to remove.

The cost of that independence is that drift cannot be detected by comparing two
working copies at test time, so it is detected by the hashes below plus the
frozen engine goldens in `tests/golden_sweep.py`.

## Vendored from

| | |
|---|---|
| Repository | `turkic-transliteration` |
| Commit | `dd5fc51081319f0c20d8338c7cb0f276fe8bf564` |
| Commit date | 2026-08-14 |
| Vendored on | 2026-08-14 |
| Upstream path | `src/turkic_translit/rules/` |

The twelve vendored files are byte-identical to the ones published in
`turkic-translit` **0.5.4** on PyPI, once line endings are normalised: every
hash in the table below equals the hash of the corresponding member of that
release's wheel. So the copy can be checked against a published artifact, not
only against one developer's checkout.

## Files

Twelve files are vendored. `ru_ipa.rules` is the one exception: it originated
here and has no upstream counterpart, because Russian is in this set as the
contact language the Turkic corpora borrow from rather than as a Turkic
language.

Hashes are SHA-256 over each file's content **with CRLF normalised to LF**.
`.gitattributes` marks `*.rules` as `text`, so git stores LF and checks out
whatever the platform uses; hashing raw bytes would record the checkout's
newline convention rather than the rules, and would agree on Linux while
disagreeing on a fresh Windows clone.

| File | SHA-256 | Origin |
|---|---|---|
| `ar_lat.rules` | `b720d7c4f395b8bfd608643b5f1a06f6e34270b09969a6ad4610840101e5e448` | upstream |
| `az_ipa.rules` | `9f02f13c0f10422d19526ccb341221646294a30695d3c0b4664a0625c80b45d3` | upstream |
| `fi_ipa.rules` | `e79d45c4ac2316bfe0d39bc7ddd32b15ecdb52b85fdca1044b27045d39eeed27` | upstream |
| `kk_ipa.rules` | `f0eaa4c6a8f7c3f6bec220793a085d3d81f385ed69604317e37963a26747a414` | upstream |
| `kk_lat.rules` | `9e42fa6466f7a975bc153d0871db4af546f96ed2d244ebfb2af61a06f30d5fe0` | upstream |
| `ky_ipa.rules` | `ec684b897eebbabda6fe5f690d5bc3f7a36d8238160ef3db91f72eda8be2053a` | upstream |
| `ky_lat.rules` | `f88afc3e2e3da2b99a0bf33a442dee9fb2b53d3e441865aad9081e118aa9883d` | upstream |
| `tr_ipa.rules` | `29ef747111224b1a4bdd0f4e1e61da10f0ce9f3342dd157bc31f87293f223de1` | upstream |
| `tr_lat.rules` | `aa5f749d2b8d38b29481b0313c39c5ca135208e532b51d6b6816ab8b8611bfb7` | upstream |
| `ug_ipa.rules` | `9112bdccddbffa65de7987a7d0bb5044c8f05c5d8c4fa75040f6c2b20fd84f02` | upstream |
| `uz_ipa.rules` | `b0d81023d3e10df2fe5ae7e5f264a4e9ca5fa9e3f4ccd5bd66a6d637d9ce127f` | upstream |
| `uzc_ipa.rules` | `79a5e49e065641ece6081d8b487d49cc86a4abbd54ac9c3b75f69d0b804088e3` | upstream |
| `ru_ipa.rules` | `4e8634b638620109511a25aa5332d070b49286b3e76da8466733d75e938f46bc` | this service |

`tests/test_rule_vendoring.py` asserts every hash in this table, so the table
cannot drift from the files it describes. `tests/test_rule_provenance.py`
asserts that every IPA file's own `Source-*` header is complete and resolvable.

## Re-vendoring

1. Copy `src/turkic_translit/rules/*.rules` over this directory.
2. Update the commit, date, and hashes above. Hash the normalised content, as
   the test does — `sha256sum` on a Windows checkout will not match.
3. Regenerate the engine goldens (see `docs/rule-engine-goldens.md`) — the rules
   and the goldens must move together, or the goldens will encode the old rules.
4. Run the suite. A rule change the pure-Python engine cannot express surfaces
   as a golden mismatch, not as a silent no-op.

## What the upstream history contains

The rules carry machine-readable `Source-*` headers naming the description each
mapping rests on. Between 2026-08-10 and 2026-08-13 upstream checked every file
against its cited source and corroborated it against a second, variety-matched
description, which corrected several mappings — Kyrgyz and Uzbek-Cyrillic `ж`,
the Uzbek `yo` vowel, the Kazakh `у` glide, a Turkish rule that deleted vowels,
and three Finnish defects. Those corrections are the reason this directory was
re-vendored; the previous copy predated all of them.

One defect is still present and is reproduced deliberately. `ar_lat.rules`
contains `ء > ' ;   ع > ' ;`, and ICU reads `'` as a quote delimiter — so the
two apostrophes bracket a literal, ء maps to the text `` ;   ع > ``, and the rule
for ع is swallowed into that literal and never exists, leaving ع
untransliterated. The engine reproduces this rather than repairing it, because
repairing it here would make this service disagree with the project that owns
the rules; the fix belongs upstream. See
`tests/test_rule_engine_goldens.py::test_arabic_quoting_defect_is_reproduced_not_repaired`.
