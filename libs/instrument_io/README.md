# instrument-io

Strictly typed IO for analytical chemistry instrument data formats.

## Installation

```toml
[tool.poetry.dependencies]
instrument-io = { path = "../libs/instrument_io", develop = true }
```

## Readers

### Mass Spectrometry Vendors

```python
from pathlib import Path
from instrument_io import AgilentReader, WatersReader, ThermoReader

# Agilent .D directories
reader = AgilentReader()
tic = reader.read_tic(Path("sample.D"))
eic = reader.read_eic(Path("sample.D"), target_mz=400.0, mz_tolerance=0.5)

# Waters .raw directories
reader = WatersReader()
tic = reader.read_tic(Path("sample.raw"))

# Thermo .raw files
reader = ThermoReader()
tic = reader.read_tic(Path("sample.raw"))

# Iterate spectra
for spectrum in reader.iter_spectra(Path("sample.raw")):
    print(spectrum["meta"]["scan_number"], spectrum["stats"]["num_peaks"])
```

### Open Formats

```python
from instrument_io import MzMLReader

reader = MzMLReader()
tic = reader.read_tic(Path("sample.mzML"))

for spectrum in reader.iter_spectra(Path("sample.mzML")):
    print(spectrum["meta"]["ms_level"], spectrum["stats"]["base_peak_mz"])
```

### Document Formats

```python
from instrument_io import ExcelReader, CSVChromatogramReader, PDFReader, DOCXReader, PPTXReader

# Excel workbooks
reader = ExcelReader()
sheets = reader.list_sheets(Path("data.xlsx"))
rows = reader.read_sheet(Path("data.xlsx"), "Sheet1")

# CSV chromatograms
reader = CSVChromatogramReader()
tic = reader.read_tic(Path("chromatogram.csv"))

# PDF table extraction
reader = PDFReader()
tables = reader.read_tables(Path("report.pdf"))

# Word documents
reader = DOCXReader()
text = reader.read_text(Path("notebook.docx"))
tables = reader.read_tables(Path("report.docx"))

# PowerPoint presentations
reader = PPTXReader()
slides = reader.read_slides(Path("presentation.pptx"))
tables = reader.read_tables(Path("results.pptx"))
```

### Specialized Formats

```python
from instrument_io import MATReader, TXTReader, SMPSReader, MGFReader, ImzMLReader

# MATLAB data files
reader = MATReader()
variables = reader.list_variables(Path("data.mat"))
data = reader.read_variable(Path("data.mat"), "results")

# Plain text files
reader = TXTReader()
content = reader.read_text(Path("log.txt"))
lines = reader.read_lines(Path("data.txt"))

# SMPS particle size data
reader = SMPSReader()
metadata = reader.read_metadata(Path("scan.rps"))
data = reader.read_full(Path("scan.rps"))

# MGF peak lists
reader = MGFReader()
spectra = list(reader.iter_spectra(Path("peaks.mgf")))

# Imaging mass spectrometry
reader = ImzMLReader()
coordinates = reader.read_coordinates(Path("tissue.imzML"))
spectrum = reader.read_spectrum(Path("tissue.imzML"), index=0)
```

## Supported Formats

| Format | Extension | Library |
|--------|-----------|---------|
| Agilent MassHunter/ChemStation | `.D` (directory) | rainbow-api |
| Waters MassLynx | `.raw` (directory) | rainbow-api |
| Thermo RAW | `.raw` (file) | ThermoRawFileParser |
| mzML | `.mzML` | pyteomics |
| mzXML | `.mzXML` | pyteomics |
| MGF | `.mgf` | pyteomics |
| imzML | `.imzML` + `.ibd` | pyimzml |
| Excel | `.xlsx`, `.xls`, `.xlsm` | openpyxl + polars |
| CSV | `.csv` | stdlib |
| PDF | `.pdf` | pdfplumber |
| Word (read) | `.docx` | python-docx |
| Word (write) | `.docx` | python-docx |
| PDF (write) | `.pdf` | reportlab |
| PowerPoint | `.pptx`, `.pptm` | python-pptx |
| MATLAB | `.mat` | scipy |
| Plain Text | `.txt` | stdlib |
| SMPS Particle Size | `.rps` | stdlib |

See [FORMATS.md](FORMATS.md) for platform requirements and Docker support.

## Return Types

All methods return strictly-typed TypedDicts:

| Type | Description |
|------|-------------|
| `TICData` | Total Ion Chromatogram |
| `EICData` | Extracted Ion Chromatogram |
| `MSSpectrum` | Mass spectrum with m/z and intensities |
| `DADData` | Diode Array Detector data (UV-Vis) |

## Writers

### Word Documents

```python
from pathlib import Path
from instrument_io import WordWriter, DocumentContent

content: DocumentContent = [
    {"type": "heading", "text": "Research Report", "level": 1},
    {"type": "paragraph", "text": "Introduction to the study.", "bold": False, "italic": False},
    {"type": "heading", "text": "Methods", "level": 2},
    {"type": "list", "items": ["Step 1", "Step 2", "Step 3"], "ordered": True},
    {"type": "table", "headers": ["Sample", "Result"], "rows": [{"Sample": "A", "Result": 1.5}], "caption": "Table 1"},
    {"type": "page_break"},
    {"type": "heading", "text": "Conclusions", "level": 2},
    {"type": "paragraph", "text": "Key findings summarized.", "bold": True, "italic": False},
]

writer = WordWriter(title="My Report", author="Lab Team")
writer.write_document(content, Path("report.docx"))
```

### PDF Documents

```python
from pathlib import Path
from instrument_io import PDFWriter, DocumentContent

content: DocumentContent = [
    {"type": "heading", "text": "Analysis Results", "level": 1},
    {"type": "paragraph", "text": "Summary of findings.", "bold": False, "italic": False},
    {"type": "table", "headers": ["Compound", "Concentration"], "rows": [{"Compound": "Caffeine", "Concentration": 125.4}], "caption": ""},
]

writer = PDFWriter(page_size="letter", margin_inches=1.0)
writer.write_document(content, Path("results.pdf"))
```

### Document Content Types

Both `WordWriter` and `PDFWriter` accept the same `DocumentContent` type:

| Type | Fields |
|------|--------|
| `HeadingContent` | `type="heading"`, `text`, `level` (1-6) |
| `ParagraphContent` | `type="paragraph"`, `text`, `bold`, `italic` |
| `TableContent` | `type="table"`, `headers`, `rows`, `caption` |
| `FigureContent` | `type="figure"`, `path`, `caption`, `width_inches` |
| `ListContent` | `type="list"`, `items`, `ordered` |
| `PageBreakContent` | `type="page_break"` |

## Error Handling

Each reader has a dedicated exception type:

```python
from instrument_io import (
    AgilentReadError, WatersReadError, ThermoReadError,
    MzMLReadError, ExcelReadError, PDFReadError,
    DOCXReadError, PPTXReadError, MATReadError,
    TXTReadError, SMPSReadError, MGFReadError, ImzMLReadError
)

# Example: Instrument data
try:
    reader = AgilentReader()
    tic = reader.read_tic(Path("sample.D"))
except AgilentReadError as e:
    print(f"Failed: {e.path}: {e.message}")

# Example: Document data
try:
    reader = DOCXReader()
    text = reader.read_text(Path("notebook.docx"))
except DOCXReadError as e:
    print(f"Failed: {e.path}: {e.message}")
```

No recovery or fallback behavior - methods either succeed or raise.

## Development

```bash
make lint    # Run ruff linter
make test    # Run pytest with coverage
make check   # Run both lint and test
```

## Requirements

- Python 3.11+
- 100% test coverage enforced
- No `Any`, `cast`, or `type: ignore`
