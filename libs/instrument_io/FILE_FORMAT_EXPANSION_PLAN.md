# File Format Expansion Plan for instrument_io

**Created:** 2025-12-04
**Status:** Planning
**Goal:** Add support for all common file formats found in Faiola Lab OneDrive

---

## 📊 Current Format Support

### ✅ Already Implemented
- **Excel:** `.xlsx`, `.xls` (via openpyxl/polars)
- **CSV/TSV:** `.csv`, `.tsv` (via built-in csv module)
- **PDF:** `.pdf` (via pdfplumber 0.11.8)
- **Mass Spec:** `.mzML`, `.mzXML` (via pyteomics)
- **Instrument:** Agilent `.D`, Thermo `.raw`, Waters `.raw` (via rainbow-api/pythonnet)
- **MGF:** `.mgf` (via pyteomics)
- **imzML:** `.imzML` (via pyteomics)

---

## 📈 File Format Statistics from Faiola Lab

Based on scan of `C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab`:

| Format | Count | Priority | Description |
|--------|-------|----------|-------------|
| `.rps` | 1,508 | 🔴 HIGH | SMPS particle size data (tab-delimited) |
| `.docx` | 508 | 🔴 HIGH | Word documents (SOPs, protocols) |
| `.pptx` | 468 | 🟡 MEDIUM | PowerPoint presentations |
| `.txt` | 97 | 🟢 LOW | Plain text files |
| `.mat` | 49 | 🟡 MEDIUM | MATLAB data files |
| `.doc` | 23 | 🟡 MEDIUM | Legacy Word format |
| `.fig` | 118 | 🟢 LOW | MATLAB figure files |
| `.pptm` | 2 | 🟢 LOW | PowerPoint macro-enabled |

**Total new files to support:** 2,773 files

---

## 🎯 Implementation Phases

### Phase 1: Text-Based Formats (Quick Wins)
**Estimated Time:** 2-3 hours
**Files Covered:** 1,605 files (57.9%)

#### 1.1 Plain Text Reader (.txt)
**Count:** 97 files
**Complexity:** ⭐ Trivial
**Library:** Built-in Python `open()`

**Features:**
- `TXTReader.read_text(path: Path) -> str`
- `TXTReader.read_lines(path: Path) -> list[str]`
- Encoding detection (UTF-8, UTF-16, etc.)

**Implementation:**
```python
# _exceptions.py
class TXTReadError(InstrumentIOError): ...

# readers/txt.py
class TXTReader:
    def read_text(self, path: Path) -> str: ...
    def read_lines(self, path: Path) -> list[str]: ...
```

**Test Files:**
- Use existing `.txt` files from Faiola Lab
- Test various encodings

---

#### 1.2 SMPS Data Reader (.rps)
**Count:** 1,508 files ⚡ **LARGEST FORMAT**
**Complexity:** ⭐⭐ Easy
**Library:** Built-in `csv` module (tab-delimited)

**File Structure:**
```
10/30/2025	11:20 AM	nsmps
Lower Voltage Limit [V]	Upper Voltage Limit [V]	...
10	10000	30	90.0	...
Sample Duration [s]	Bin End Voltage [V]	Count [#]	...
3.01	12.62	0.00	8.75	...
```

**Features:**
- `SMPSReader.read_metadata(path: Path) -> dict[str, Any]`
- `SMPSReader.read_data(path: Path) -> list[dict[str, CellValue]]`
- `SMPSReader.read_full(path: Path) -> SMPSData` (metadata + data combined)

**TypedDict:**
```python
class SMPSMetadata(TypedDict):
    timestamp: str
    instrument: str
    lower_voltage_limit: float
    upper_voltage_limit: float
    num_bins: int
    scan_time: float
    aerosol_flow_rate: float
    sheath_flow_rate: float
    temperature: float
    pressure: float

class SMPSData(TypedDict):
    metadata: SMPSMetadata
    data: list[dict[str, CellValue]]
```

**Implementation:**
```python
# _exceptions.py
class SMPSReadError(InstrumentIOError): ...

# _decoders/smps.py
def _decode_smps_metadata(lines: list[str]) -> SMPSMetadata: ...
def _decode_smps_data(lines: list[str]) -> list[dict[str, CellValue]]: ...

# readers/smps.py
class SMPSReader:
    def read_metadata(self, path: Path) -> SMPSMetadata: ...
    def read_data(self, path: Path) -> list[dict[str, CellValue]]: ...
    def read_full(self, path: Path) -> SMPSData: ...
```

**Test Files:**
- `Current Projects/Acyclic Terpenes/SMPS data/limonene/251030/*.rps`
- Verify metadata parsing
- Verify data table parsing
- Test particle size distribution calculations

---

### Phase 2: Office Documents
**Estimated Time:** 6-8 hours
**Files Covered:** 1,001 files (36.1%)

#### 2.1 Word Documents (.docx)
**Count:** 508 files
**Complexity:** ⭐⭐⭐ Medium
**Library:** `python-docx` (latest stable)

**Use Cases:**
- Standard preparation SOPs (`Important Docs/Faiola Lab SOPs/TDGCMS/Standard Preparation.docx`)
- Chemical segregation guidelines
- Lab protocols

**Features:**
- `DOCXReader.read_text(path: Path) -> str`
- `DOCXReader.read_paragraphs(path: Path) -> list[str]`
- `DOCXReader.read_tables(path: Path) -> list[list[dict[str, CellValue]]]`
- `DOCXReader.read_headings(path: Path) -> list[tuple[int, str]]`

**Implementation:**
```python
# pyproject.toml
python-docx = "^1.1.2"  # Latest stable

# _exceptions.py
class DOCXReadError(InstrumentIOError): ...

# _protocols/python_docx.py
class DocumentProtocol(Protocol):
    paragraphs: list[ParagraphProtocol]
    tables: list[TableProtocol]
    ...

# _decoders/docx.py
def _decode_docx_table(table: TableProtocol) -> list[dict[str, CellValue]]: ...
def _decode_docx_paragraph(para: ParagraphProtocol) -> str: ...

# readers/docx.py
class DOCXReader:
    def read_text(self, path: Path) -> str: ...
    def read_paragraphs(self, path: Path) -> list[str]: ...
    def read_tables(self, path: Path) -> list[list[dict[str, CellValue]]]: ...
    def read_headings(self, path: Path) -> list[tuple[int, str]]: ...
```

**Test Files:**
- `Important Docs/Faiola Lab SOPs/TDGCMS/Standard Preparation.docx`
- `Important Docs/Chemical Inventory/CHEMICAL SEGREGATION GUIDELINES_01SEP2020.docx`
- `Important Docs/Faiola Lab SOPs/Aerosol_Composition_SamplePrep/Sample Extraction for HPLC runs.docx`

---

#### 2.2 Legacy Word (.doc)
**Count:** 23 files
**Complexity:** ⭐⭐⭐⭐ Hard
**Library:** `python-docx` (may not work) or `textract` or `antiword`

**Options:**
1. Try `python-docx` (may work for some .doc files)
2. Use `textract` library (requires system dependencies)
3. Use `antiword` command-line tool
4. Convert to .docx first using `LibreOffice` command-line
5. **Recommended:** Just support .docx and note .doc as unsupported for now

**Implementation Decision:**
- Start with .docx only
- Add .doc support later if needed
- Most .doc files can be converted to .docx manually

---

#### 2.3 PowerPoint (.pptx, .pptm)
**Count:** 470 files
**Complexity:** ⭐⭐⭐ Medium
**Library:** `python-pptx` (latest stable)

**Use Cases:**
- SOPs (`Important Docs/Faiola Lab SOPs/OFR_Experiments/Standard Operating Procedure for ACSM.pptx`)
- Training materials
- Data presentations

**Features:**
- `PPTXReader.read_text(path: Path) -> str`
- `PPTXReader.read_slides(path: Path) -> list[str]`
- `PPTXReader.read_tables(path: Path) -> list[list[dict[str, CellValue]]]`
- `PPTXReader.list_slide_titles(path: Path) -> list[str]`
- `PPTXReader.count_slides(path: Path) -> int`

**Implementation:**
```python
# pyproject.toml
python-pptx = "^1.0.2"  # Latest stable

# _exceptions.py
class PPTXReadError(InstrumentIOError): ...

# _protocols/python_pptx.py
class PresentationProtocol(Protocol):
    slides: list[SlideProtocol]
    ...

# _decoders/pptx.py
def _decode_pptx_table(table: TableProtocol) -> list[dict[str, CellValue]]: ...
def _decode_pptx_slide(slide: SlideProtocol) -> str: ...

# readers/pptx.py
class PPTXReader:
    def read_text(self, path: Path) -> str: ...
    def read_slides(self, path: Path) -> list[str]: ...
    def read_tables(self, path: Path) -> list[list[dict[str, CellValue]]]: ...
    def list_slide_titles(self, path: Path) -> list[str]: ...
    def count_slides(self, path: Path) -> int: ...
```

**Test Files:**
- `Important Docs/Faiola Lab SOPs/OFR_Experiments/Standard Operating Procedure for ACSM.pptx`
- `Important Docs/Faiola Lab SOPs/TDGCMS/TDGCMS_Calculation.pptx`

---

### Phase 3: Scientific Data Formats
**Estimated Time:** 4-6 hours
**Files Covered:** 167 files (6.0%)

#### 3.1 MATLAB Data (.mat)
**Count:** 49 files
**Complexity:** ⭐⭐⭐ Medium
**Library:** `scipy.io.loadmat` (already a dependency)

**Features:**
- `MATReader.list_variables(path: Path) -> list[str]`
- `MATReader.read_variable(path: Path, var_name: str) -> Any`
- `MATReader.read_all(path: Path) -> dict[str, Any]`
- `MATReader.get_metadata(path: Path) -> dict[str, str]`

**Implementation:**
```python
# _exceptions.py
class MATReadError(InstrumentIOError): ...

# _protocols/scipy_io.py
# scipy.io.loadmat returns dict, so may not need Protocol wrapper

# _decoders/mat.py
def _decode_mat_array(arr: np.ndarray) -> list[Any]: ...
def _decode_mat_struct(struct: Any) -> dict[str, Any]: ...

# readers/mat.py
class MATReader:
    def list_variables(self, path: Path) -> list[str]: ...
    def read_variable(self, path: Path, var_name: str) -> Any: ...
    def read_all(self, path: Path) -> dict[str, Any]: ...
```

**Test Files:**
- Search Faiola Lab for `.mat` files in current projects

---

#### 3.2 MATLAB Figures (.fig)
**Count:** 118 files
**Complexity:** ⭐⭐⭐⭐⭐ Very Hard
**Library:** Unclear - may need MATLAB engine or custom parser

**Challenges:**
- .fig files are binary MATLAB-specific format
- May require MATLAB Engine for Python (requires MATLAB license)
- Alternative: Extract metadata only, not full figure

**Options:**
1. **MATLAB Engine API** - Requires MATLAB installation + license
2. **matplotlib** - Can read some .fig files (if saved with specific settings)
3. **Read metadata only** - Extract figure properties without rendering

**Implementation Decision:**
- **Defer to Phase 4** (future work)
- Requires investigation of MATLAB file format
- May not be feasible without MATLAB installation

---

## 🔧 Technical Requirements

### Dependencies to Add
```toml
[tool.poetry.dependencies]
python-docx = "^1.1.2"      # Word documents
python-pptx = "^1.0.2"      # PowerPoint
# scipy already present for .mat support
```

### New Exception Classes
```python
# _exceptions.py
class TXTReadError(InstrumentIOError): ...
class SMPSReadError(InstrumentIOError): ...
class DOCXReadError(InstrumentIOError): ...
class PPTXReadError(InstrumentIOError): ...
class MATReadError(InstrumentIOError): ...
```

### New Protocol Wrappers
```python
# _protocols/python_docx.py - Word documents
# _protocols/python_pptx.py - PowerPoint
# _protocols/scipy_io.py - MATLAB (if needed)
```

### New Decoders
```python
# _decoders/smps.py - SMPS data parsing
# _decoders/docx.py - Word table decoding
# _decoders/pptx.py - PowerPoint table decoding
# _decoders/mat.py - MATLAB data decoding
```

### New Readers
```python
# readers/txt.py - Plain text
# readers/smps.py - SMPS data
# readers/docx.py - Word documents
# readers/pptx.py - PowerPoint
# readers/mat.py - MATLAB data
```

---

## ✅ Quality Standards

All implementations must follow existing patterns:

### 1. Strict Typing
- ✅ No `Any` types
- ✅ No `cast()` usage
- ✅ No `type: ignore` comments
- ✅ TypedDict for all structured data
- ✅ Protocol wrappers for external libraries

### 2. Error Handling
- ✅ Custom exception classes for each format
- ✅ No try/except without log + raise (guard rules)
- ✅ Explicit error messages with file paths

### 3. Testing
- ✅ 100% statement coverage
- ✅ 100% branch coverage
- ✅ Unit tests for decoders
- ✅ Integration tests with real files from Faiola Lab
- ✅ Typed mock classes (no MagicMock)

### 4. Documentation
- ✅ Module docstrings
- ✅ Function docstrings
- ✅ Type annotations on all parameters/returns
- ✅ README updates

### 5. Code Review
- ✅ Run `make check` (guard + mypy + ruff + pytest)
- ✅ 0 guard violations
- ✅ 0 mypy errors
- ✅ All tests passing

---

## 📝 Implementation Order

### Recommended Sequence:
1. **Phase 1.1:** TXT reader (30 min)
2. **Phase 1.2:** SMPS/RPS reader (2 hours)
3. **Phase 2.1:** DOCX reader (3-4 hours)
4. **Phase 2.3:** PPTX reader (3-4 hours)
5. **Phase 3.1:** MAT reader (2-3 hours)
6. **Phase 2.2:** DOC reader (defer/investigate)
7. **Phase 3.2:** FIG reader (defer/future work)

### Total Estimated Time: 12-16 hours

---

## 🎯 Success Metrics

After completion, `instrument_io` will support:

**Documents:**
- ✅ 508 Word documents (.docx)
- ✅ 470 PowerPoint presentations (.pptx)
- ✅ 97 text files (.txt)

**Scientific Data:**
- ✅ 1,508 SMPS particle size files (.rps) ⚡ **MAJOR WIN**
- ✅ 49 MATLAB data files (.mat)
- ✅ 460 Excel files (already supported)
- ✅ 456 PDFs (already supported)
- ✅ 146 CSV files (already supported)

**Total Coverage:** 3,694 / 9,774 files = 37.8% of all files in Faiola Lab
**With existing support:** 4,756 / 9,774 files = 48.6% coverage

---

## 🚀 Future Enhancements (Phase 4)

### Low Priority Formats:
- `.doc` - Legacy Word (23 files)
- `.fig` - MATLAB figures (118 files)
- `.pxp` - Igor Pro (20 files)
- `.opju` - OriginLab (3 files)
- `.qza`, `.qzv` - QIIME2 (5 files)
- `.R`, `.Rmd`, `.rda` - R files (108 files)

### Advanced Features:
- Batch processing multiple files
- Format conversion utilities
- Data validation/schema checking
- Export to standardized formats

---

## 📚 References

### Libraries:
- **python-docx:** https://python-docx.readthedocs.io/
- **python-pptx:** https://python-pptx.readthedocs.io/
- **scipy.io:** https://docs.scipy.org/doc/scipy/reference/io.html
- **pdfplumber:** https://github.com/jsvine/pdfplumber

### Standards:
- Guard rules: No try/except without log+raise
- Mypy strict mode
- 100% test coverage required
- Protocol-based external library integration

---

**Last Updated:** 2025-12-04
**Status:** Ready to implement
**Next Step:** Begin Phase 1.1 (TXT reader)
