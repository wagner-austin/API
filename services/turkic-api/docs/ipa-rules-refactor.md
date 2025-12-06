# IPA Rules Refactor

Audit of balanced OSCAR corpora output (2025-12-01) revealed missing transliteration rules causing character leakage.

**Status: COMPLETED 2025-12-02**

## Summary

| File | Priority | Issue | Status |
|------|----------|-------|--------|
| `ug_ipa.rules` | **High** | Missing Arabic letters (ع ح ء) and diacritics | ✅ Done |
| `ky_ipa.rules` | Medium | Missing Cyrillic letters (ѳ ҥ) | ✅ Done |
| `tr_ipa.rules` | Medium | Missing diacritic vowels (â î û) | ✅ Done |
| `translit.py` | Medium | Uzbek needs both Latin + Cyrillic rules | ✅ Done |

---

## 1. Uyghur (`ug_ipa.rules`)

**Problem:** OSCAR Uyghur is 97.7% Arabic script, but rules are missing common characters.

### Missing Consonants

| Char | Unicode | Name | Count | IPA |
|------|---------|------|-------|-----|
| ع | U+0639 | ARABIC LETTER AIN | 68,045 | ʕ |
| ح | U+062D | ARABIC LETTER HAH | 26,895 | ħ |
| ء | U+0621 | ARABIC LETTER HAMZA | 32,760 | ʔ |

### Missing Punctuation

| Char | Unicode | Name | Count | Action |
|------|---------|------|-------|--------|
| ، | U+060C | ARABIC COMMA | 106,112 | → `,` or delete |
| ؟ | U+061F | ARABIC QUESTION MARK | 4,562 | → `?` or delete |
| ـ | U+0640 | ARABIC TATWEEL | 4,638 | delete (kashida) |

### Missing Diacritics (delete all - optional short vowel marks)

| Char | Unicode | Name | Count |
|------|---------|------|-------|
| َ | U+064E | ARABIC FATHA | 17,270 |
| ِ | U+0650 | ARABIC KASRA | 6,698 |
| ُ | U+064F | ARABIC DAMMA | 5,769 |
| ْ | U+0652 | ARABIC SUKUN | 5,311 |
| ّ | U+0651 | ARABIC SHADDA | 3,324 |

### Proposed Additions

```
# Missing consonants
ع > ʕ ;
ح > ħ ;
ء > ʔ ;

# Punctuation normalization
، >   ;       # Arabic comma → space (or comma)
؟ >   ;       # Arabic question → space (or ?)
ـ >   ;       # Tatweel (kashida) → delete

# Diacritics (delete - they mark short vowels already represented)
َ >   ;       # fatha
ِ >   ;       # kasra
ُ >   ;       # damma
ْ >   ;       # sukun
ّ >   ;       # shadda
```

---

## 2. Kyrgyz (`ky_ipa.rules`)

**Problem:** Some archaic/variant Cyrillic letters appear in OSCAR data.

| Char | Unicode | Name | Count | IPA |
|------|---------|------|-------|-----|
| ѳ | U+0473 | CYRILLIC SMALL LETTER FITA | 2,440 | θ or f |
| ҥ | U+04CA | CYRILLIC SMALL LETTER EN WITH TAIL | 595 | ŋ |

### Proposed Additions

```
# Archaic/variant letters
ѳ > θ ;    Ѳ > θ ;    # fita (Greek theta loans)
ҥ > ŋ ;    Ҥ > ŋ ;    # en with tail (velar nasal)
```

---

## 3. Turkish (`tr_ipa.rules`)

**Problem:** Circumflex vowels (from Arabic/Persian loans, older orthography) not handled.

| Char | Unicode | Name | Count | IPA |
|------|---------|------|-------|-----|
| â | U+00E2 | LATIN SMALL LETTER A WITH CIRCUMFLEX | 4,506 | aː or a |
| î | U+00EE | LATIN SMALL LETTER I WITH CIRCUMFLEX | 1,598 | iː or i |
| û | U+00FB | LATIN SMALL LETTER U WITH CIRCUMFLEX | 746 | uː or u |

### Proposed Additions

```
# Circumflex vowels (loan words, indicate length or palatalization)
â > aː ;    Â > aː ;    # or just 'a' if length not needed
î > iː ;    Î > iː ;
û > uː ;    Û > uː ;
```

---

## 4. Uzbek (`translit.py`)

**Problem:** OSCAR Uzbek is 96.8% Latin, 0.8% Cyrillic. Current `to_ipa()` only applies Latin rules.

### Solution Applied

Modified `to_ipa()` in `src/turkic_api/core/translit.py` to apply both `uz_ipa.rules` (Latin) and `uzc_ipa.rules` (Cyrillic) for Uzbek:

```python
# ADDITION 2025-12-02: Uzbek mixed-script support
# OSCAR Uzbek is 96.8% Latin, 0.8% Cyrillic - apply both rule sets
if lang == "uz":
    result = apply_rules(result, _rules("uzc_ipa.rules"))
```

Added test in `tests/test_translit_branches.py`:
```python
def test_to_ipa_uz_cyrillic_script() -> None:
    result = to_ipa("Ўзбекистон", "uz")
    assert "o" in result  # Cyrillic Ў maps to /o/
    assert "ɔ" in result  # Cyrillic О maps to /ɔ/
```

---

## 5. Not Requiring Changes

### Kazakh (`kk_ipa.rules`)
- Leakage is punctuation (« » № –) and minor Arabic contamination from OSCAR
- Rules are complete for Kazakh Cyrillic

### Azerbaijani (`az_ipa.rules`)
- Leakage is punctuation and minor Russian/Arabic contamination
- Rules are complete for Azerbaijani Latin

---

## Audit Methodology

1. Ran `build_balanced_corpora.py` with OSCAR source, 0.95 LID threshold
2. Analyzed output files for non-IPA Unicode characters
3. Cross-referenced with `.rules` files to identify gaps
4. Checked OSCAR source script distribution to confirm data correctness

### Output File Stats (2025-12-01)

| File | Size | Lines | IPA Chars |
|------|------|-------|-----------|
| oscar_az_ipa.txt | 19 MB | 129,581 | 11,709,174 |
| oscar_kk_ipa.txt | 18 MB | 125,116 | 11,709,174 |
| oscar_ky_ipa.txt | 18 MB | 136,749 | 11,709,174 |
| oscar_tr_ipa.txt | 17 MB | 125,809 | 11,709,174 |
| oscar_ug_ipa.txt | 18 MB | 129,395 | 11,709,174 |
| oscar_uz_ipa.txt | 16 MB | 109,231 | 11,709,174 |

Bottleneck language: Uzbek (uz) with 11,709,174 IPA characters.
