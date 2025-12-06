# Supported File Formats

## Overview

`instrument_io` provides unified access to mass spectrometry and chromatography data from multiple vendors and open formats.

---

## Fully Supported Formats

### Vendor Formats

| Vendor | Format | Extension | Library | Platforms | Install |
|--------|--------|-----------|---------|-----------|---------|
| Agilent | MassHunter/ChemStation | `.D` (directory) | rainbow-api | All | `pip install` |
| Waters | MassLynx | `.raw` (directory) | rainbow-api | All | `pip install` |
| Thermo Fisher | RAW | `.raw` (file) | ThermoRawFileParser | All | Bundled |

### Open Formats

| Format | Description | Extension | Library | Platforms |
|--------|-------------|-----------|---------|-----------|
| mzML | HUPO-PSI standard | `.mzML` | pyteomics | All |
| mzXML | Legacy XML format | `.mzXML` | pyteomics | All |
| MGF | Mascot Generic Format | `.mgf` | pyteomics | All |
| imzML | Imaging MS format | `.imzML` + `.ibd` | pyimzml | All |

### Document/Data Formats

| Format | Description | Extension | Library | Platforms |
|--------|-------------|-----------|---------|-----------|
| Excel | Spreadsheets | `.xlsx`, `.xls`, `.xlsm` | openpyxl + polars | All |
| CSV | Chromatogram exports | `.csv` | stdlib | All |
| PDF | Documents with tables | `.pdf` | pdfplumber | All |
| Word | Documents | `.docx` | python-docx | All |
| PowerPoint | Presentations | `.pptx`, `.pptm` | python-pptx | All |
| MATLAB | MAT files | `.mat` | scipy | All |
| Plain Text | Text files | `.txt` | stdlib | All |
| SMPS | Particle size data | `.rps` | stdlib | All |

---

## Planned Support

| Vendor | Format | Extension | Library | Status |
|--------|--------|-----------|---------|--------|
| Bruker | timsTOF | `.d` (directory with `.tdf`) | opentimspy | Planned |
| SCIEX/AB | Analyst | `.wiff` + `.wiff.scan` | alpharaw | Planned |

See `~/.claude/plans/instrument-io-vendor-expansion.md` for implementation details.

---

## Not Supported

| Vendor | Format | Extension | Reason | Workaround |
|--------|--------|-----------|--------|------------|
| SCIEX | SCIEX OS | `.wiff2` | Different format than `.wiff` | Use SCIEX software to convert |
| Shimadzu | LabSolutions | `.lcd` | No Python library available | Convert to mzML with msconvert |
| Shimadzu | GCMS | `.qgd` | No Python library available | Convert to mzML with msconvert |

### Converting Unsupported Formats

For formats without native Python support, use **ProteoWizard msconvert**:

```bash
# Install ProteoWizard (Windows)
# Download from: https://proteowizard.sourceforge.io/download.html

# Convert to mzML
msconvert input.lcd --mzML -o output_dir/

# Then read with instrument_io
from instrument_io import MzMLReader
reader = MzMLReader()
tic = reader.read_tic(Path("output_dir/input.mzML"))
```

---

## Platform Requirements

### All Platforms (Windows, Linux, macOS)

- Agilent `.D` via rainbow-api
- Waters `.raw` via rainbow-api
- All open formats (mzML, mzXML, MGF, imzML)
- All document formats (Excel, CSV, PDF, etc.)

### Windows + Linux Only

- Bruker `.d` via opentimspy (macOS not supported by Bruker SDK)

### Requires Mono Runtime (Linux/macOS)

- Thermo `.raw` via ThermoRawFileParser
- SCIEX `.wiff` via alpharaw (uses pythonnet)

Install Mono on Linux:
```bash
sudo apt install mono-complete
```

Install Mono on macOS:
```bash
brew install mono
```

---

## Docker Support

For cloud deployment or consistent environments, use Docker:

```dockerfile
FROM python:3.11-slim

# Install Mono for Thermo/SCIEX support
RUN apt-get update && apt-get install -y mono-complete && rm -rf /var/lib/apt/lists/*

# Install instrument_io
RUN pip install instrument-io

WORKDIR /data
```

For maximum vendor support including msconvert:

```dockerfile
FROM chambm/pwiz-skyline-i-agree-to-the-vendor-licenses:latest
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install instrument-io
```

---

## Reader API Summary

### Chromatography/MS Readers

All chromatography and mass spec readers follow a consistent API:

```python
from pathlib import Path
from instrument_io import AgilentReader, ThermoReader, MzMLReader

# Check format support
reader = AgilentReader()
if reader.supports_format(path):
    # Read TIC
    tic = reader.read_tic(path)

    # Read EIC
    eic = reader.read_eic(path, target_mz=400.0, mz_tolerance=0.5)

    # Iterate spectra
    for spectrum in reader.iter_spectra(path):
        print(spectrum["meta"]["scan_number"])

    # Read single spectrum
    spec = reader.read_spectrum(path, scan_number=1)
```

### Document Readers

Document readers provide typed access to text, tables, and metadata:

```python
from instrument_io import DOCXReader, ExcelReader, PDFReader, PPTXReader, MATReader, TXTReader

# Word documents
docx_reader = DOCXReader()
text = docx_reader.read_text(Path("notebook.docx"))
tables = docx_reader.read_tables(Path("report.docx"))
headings = docx_reader.read_headings(Path("outline.docx"))

# Excel workbooks
excel_reader = ExcelReader()
sheets = excel_reader.list_sheets(Path("data.xlsx"))
rows = excel_reader.read_sheet(Path("data.xlsx"), "Sheet1")

# PDF documents
pdf_reader = PDFReader()
tables = pdf_reader.read_tables(Path("results.pdf"))
text = pdf_reader.read_text(Path("report.pdf"))

# PowerPoint presentations
pptx_reader = PPTXReader()
slides = pptx_reader.read_slides(Path("talk.pptx"))
titles = pptx_reader.list_slide_titles(Path("talk.pptx"))

# MATLAB files
mat_reader = MATReader()
variables = mat_reader.list_variables(Path("data.mat"))
data = mat_reader.read_variable(Path("data.mat"), "results")

# Text files
txt_reader = TXTReader()
content = txt_reader.read_text(Path("log.txt"))
lines = txt_reader.read_lines(Path("data.txt"))
```

### Return Types

All methods return strictly-typed TypedDicts:

- `TICData` - Total Ion Chromatogram
- `EICData` - Extracted Ion Chromatogram
- `MSSpectrum` - Mass spectrum with m/z and intensities
- `DADData` - Diode Array Detector data (UV-Vis)

---

## Fixture Locations

Test fixtures for each format are in `tests/fixtures/`:

| Format | Fixture |
|--------|---------|
| Agilent | `sample.D/`, `brown.D/` |
| Waters | `waters_sample.raw/` |
| Thermo | `small.RAW` |
| mzML | `tiny.pwiz.1.1.mzML` |
| MGF | `sample.mgf` |
| imzML | `tiny.imzML`, `tiny.ibd` |
| Excel | `sample_chromatogram.xlsx` |
| CSV | `chromatogram.csv` |
