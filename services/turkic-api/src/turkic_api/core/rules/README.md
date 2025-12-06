# Transliteration Rules

This directory contains transliteration rule files for converting text between different scripts and phonetic representations.

## File Naming Convention

Rule files follow a strict naming pattern:

- **IPA rules**: `{lang_code}_ipa.rules` - Convert to International Phonetic Alphabet
- **Latin rules**: `{lang_code}_lat.rules` - Convert to Latin script

Where `{lang_code}` is the ISO 639-1 language code (e.g., `kk` for Kazakh, `ky` for Kyrgyz).

## Currently Supported Languages

The following languages have rule files and are exposed through the API:

- **Kazakh (kk)**: `kk_ipa.rules`, `kk_lat.rules`
- **Kyrgyz (ky)**: `ky_ipa.rules`, `ky_lat.rules`
- **Uzbek (uz)**: `uz_ipa.rules`, `uzc_ipa.rules` (Cyrillic variant)
- **Uyghur (ug)**: `ug_ipa.rules`
- **Azerbaijani (az)**: `az_ipa.rules`

Additional rule files exist but are not currently exposed through the API configuration.

## Sources

All rule files and test files are based on peer-reviewed linguistic research and official standards.

### Kazakh

kk_ipa.rules
test_kazakh_northwind.py
test_kazakh_ipa_letters.py
McCollum, A. G. & Chen, S. (2021). "Kazakh". Journal of the International Phonetic Association, 51(2): 276-298. https://doi.org/10.1017/S0025100319000185

kk_lat.rules
Presidential Decree No. 569 (26 October 2017). On the change of the alphabet of the Kazakh language from the Cyrillic to the Latin script. Republic of Kazakhstan. https://adilet.zan.kz/rus/docs/U1700000569

### Kyrgyz

ky_ipa.rules
test_kyrgyz_words_mccollum.py
test_kyrgyz_ipa_letters.py
McCollum, A. G. (2020). "Vowel harmony and positional variation in Kyrgyz". Laboratory Phonology, 11(1): article 25. https://doi.org/10.5334/labphon.247

ky_lat.rules
Modern Practical Latin Transliteration (NFC)

### Uzbek

uz_ipa.rules
uzc_ipa.rules
test_uzbek_northwind.py
test_uzbek_lat_ipa_letters.py
test_uzbek_cyr_ipa_letters.py
Ido, S. (2025). "Uzbek". Journal of the International Phonetic Association, 55(1-2): 152-168. https://doi.org/10.1017/S0025100324000148

### Uyghur

ug_ipa.rules
test_uyghur_ipa_letters.py
McCollum, A. G. (2021). "Transparency, locality, and contrast in Uyghur backness harmony". Laboratory Phonology, 12(1): article 8. https://doi.org/10.5334/labphon.239

### Azerbaijani

az_ipa.rules
test_azerbaijani_northwind.py
test_azerbaijani_ipa_letters.py
Ghaffarvand Mokari, P. & Werner, S. (2017). "Azerbaijani". Journal of the International Phonetic Association, 47(2): 207–212. https://doi.org/10.1017/S0025100317000184

### Turkish

tr_ipa.rules
test_turkish_ipa_letters.py
Zimmer, K. & Orgun, O. (1992). "Turkish". Journal of the International Phonetic Association, 22(1-2): 43-45. https://doi.org/10.1017/S0025100300004588

tr_lat.rules
Arslan, A. (2016). "DeASCIIfication approach to handle diacritics in Turkish information retrieval". Information Processing & Management, 52(2): 326-339. https://doi.org/10.1016/j.ipm.2015.08.004

### Finnish

fi_ipa.rules
https://fi.alegsaonline.com/art/60239

test_finnish_maamme.py
test_finnish_ipa_letters.py
Suomi, K., Toivanen, J., & Ylitalo, R. (2008). Finnish sound structure: Phonetics, phonology, phonotactics and prosody. Studia Humaniora Ouluensia 9. Oulu University Press. ISBN: 978-951-42-8984-2. https://urn.fi/URN:ISBN:9789514289842
