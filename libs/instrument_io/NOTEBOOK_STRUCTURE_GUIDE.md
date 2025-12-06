# Austin Wagner Lab Notebook - Structure Guide

Based on inspection of other lab members' notebooks and your software development work.

## What You've Built

### 1. instrument_io (libs/instrument_io)
**Purpose**: Strictly typed I/O library for analytical chemistry instrument data formats
- **Reads**: Agilent .D, Waters .raw, Thermo .raw, mzML, Excel, PDF, Word, PowerPoint, MATLAB
- **Features**: 100% test coverage, no `Any` types, Protocol-based typing
- **Document Readers** (newly exported):
  - `DOCXReader` - Word documents (.docx)
  - `PPTXReader` - PowerPoint (.pptx)
  - `MATReader` - MATLAB data files (.mat)
  - `TXTReader` - Plain text (.txt)
  - `SMPSReader` - SMPS particle size data (.rps)
  - `ExcelReader` - Excel workbooks (.xlsx)
  - `PDFReader` - PDF documents

### 2. Tree Bot (PROJECTS/Tree Bot)
**Purpose**: Offline Windows app for validating/transforming Excel workbooks (tree GCMS data)
- **Features**:
  - Schema validation (old/new formats)
  - Data standardization and quality reports
  - NiceGUI-based local web UI
  - CLI for batch processing
  - Installer for Windows distribution
- **Tech**: Polars, openpyxl, PyYAML, rainbow-api, 100% test coverage

### 3. Tree Data (PROJECTS/Tree Data)
**Purpose**: Research data repository
- CompoundIDs (multiple versions for different sites)
- Sample metadata and summaries
- Literature review notes
- Field notebooks
- Chiral terpenes literature review

## What Other Lab Members Have in Their Notebooks

### Emily Truong (PhD Student - Metabolomics Focus)
**2025 Lab Notebook** contains:
- Daily entries with dates (format: MM/DD/YY)
- GCMS run notes and decisions
- Data processing notes (ProGenesis, blanks removal, normalization)
- Conversations with Felix (PI/supervisor)
- Sample preparation notes
- Standards preparation references
- Meeting notes

### Jasmine OseiEnin (Computational/Data Analysis Focus)
**Organization**:
- **GCMS_Run_2024.xlsx** - Run log with sheets per date
  - Template sheet
  - Individual run sheets (dates: 101724, 100624, etc.)
  - Tracks start/finish times, samples, conditions
- **Coding commands.xlsx** - Personal reference
  - Linux/Bash commands with examples
  - Command functions and use cases
- **Data Management Plan** - Project-level documentation
- **Project Plan Template** - Reusable planning structure
- **GoalTemplate.xlsx** - Goal tracking
- **Response factors.xlsx** - Instrument calibration data
- **Literature organization** - Reading lists and notes
- **R packages.docx** - Software environment documentation

### Claire Freimark
**Organization**:
- Annual lab notebooks (2023-2024, 2024-2025)
- Project-specific reflections (Seagrass VOC)

### Juan Flores
**Organization**:
- Protocol documentation (BVOC Sample Collection)

## Recommended Structure for Your Notebook

### Core Documents

#### 1. **2025 Lab Notebook.docx** (Main chronological notebook)
Daily/weekly entries with:
- Date headers
- Software development progress
  - instrument_io updates (new readers, bug fixes)
  - Tree Bot features/improvements
  - Testing and validation results
- Data processing activities
  - Which datasets processed
  - Issues encountered and solutions
- Meetings with PI/collaborators
- Technical decisions and rationale
- Code review notes

#### 2. **Software Development Log.xlsx**
Sheets:
- **instrument_io Development** - Feature additions, version history
- **Tree Bot Development** - UI updates, validation rule changes
- **Bug Tracker** - Issues found and resolved
- **Testing Log** - Test coverage, integration tests

#### 3. **Data Processing Notebook.xlsx** or **.docx**
- Datasets processed (with dates and locations)
- Processing pipeline versions used
- Quality control checks
- Issues and resolutions
- Output locations

#### 4. **Coding Reference.xlsx** (like Jasmine's)
Sheets:
- **Python Commands** - Common operations
- **instrument_io Examples** - Code snippets for each reader
- **Polars Operations** - DataFrame manipulation
- **Testing Patterns** - pytest examples
- **Git Commands** - Version control workflows

#### 5. **instrument_io Documentation/**
Folder with:
- **Reader Usage Examples.md** - How to use each reader
- **API Reference.md** - Key functions and types
- **Development Notes.md** - Design decisions
- **Binary Format Notes.md** - Insights about .D, .raw formats

#### 6. **Tree Bot Documentation/**
Folder with:
- **User Guide.docx** - How to use the app
- **Validation Rules.md** - Quality control criteria
- **Schema Definitions.md** - Data structure documentation
- **Deployment Notes.md** - Packaging and installation

#### 7. **Meeting Notes.docx**
- Weekly/biweekly meetings
- Decisions made
- Action items
- Technical discussions

#### 8. **Literature Notes/** (Optional)
- Mass spectrometry data formats
- Data validation techniques
- Software engineering papers
- Analytical chemistry methods

#### 9. **Project Plans/**
Folder with:
- **instrument_io Roadmap.md**
- **Tree Bot Feature Backlog.md**
- **Integration Plans.md** (connecting systems)

### Existing: Chemical Inventory and Standards/
Keep your existing folder structure - this is valuable reference material

## Daily Workflow Suggestions

### When coding:
1. Date entry in main lab notebook
2. What you're working on
3. Technical decisions made
4. Challenges encountered
5. Solutions implemented
6. Testing results

### When processing data:
1. Dataset identifier and location
2. Processing date
3. Software/version used
4. Parameters
5. QC results
6. Output location

### When updating instrument_io:
1. What reader you're working on
2. Test files used
3. Edge cases discovered
4. Documentation updated

## Tips from Other Notebooks

1. **Use dated entries** - MM/DD/YY format works well
2. **Reference external locations** - "see standards notebook for more info"
3. **Track decisions** - Why you chose one approach over another
4. **Document conversations** - Who said what, when
5. **Keep running lists** - Commands, packages, useful resources
6. **Link to code** - Reference Git commits or file locations

## Software-Specific Suggestions

Since you're the software developer:
- **Version everything** - Track software versions used in analyses
- **Document dependencies** - What packages broke, what versions work
- **Keep examples** - Working code snippets for common tasks
- **Track test data** - Where sample files are located
- **Note performance** - Processing times, memory usage for large files
- **API changes** - When you update interfaces, document migration notes

## Integration with Your Repos

Your notebook should bridge your code and the lab's science:
- **instrument_io** → Binary file reading for other lab members
- **Tree Bot** → Data validation/QC for research datasets
- **Tree Data** → Real-world test cases for your software

Document how these connect and how lab members can use them!

## Example Notebook Entries

### Example 1: Daily Software Development
```
12/05/25

instrument_io - Added DOCXReader, PPTXReader, MATReader exports
- Updated src/instrument_io/__init__.py to export document readers
- AIs were getting confused trying to import DOCXReader
- All document readers now accessible from main package
- Tested on Emily's lab notebook - successfully reads .docx files
- Next: Add usage examples to README

Tree Bot - No updates today

Meeting Notes:
- Discussed with lab about notebook structure
- Looked at Jasmine's coding reference spreadsheet - good model
- Need to create similar reference for instrument_io
```

### Example 2: Data Processing Session
```
12/04/25

Processed BlueOak GCMS Dataset
- Location: Tree Data/CompoundIDs - BlueOak.xlsx
- Used: instrument_io v0.1.0, ExcelReader
- Found: 15 sheets with compound identifications
- QC: All sheets read successfully, no missing data
- Output: Saved to processed/blueoak_compounds_20251204.parquet
- Notes: Some compound names needed normalization (see Tree Bot normalize.py)
```

### Example 3: Reader Development
```
11/28/25

Waters .raw Reader Testing
- Test file: Sample_data/Sample001.raw (directory format)
- Using rainbow-api backend
- Successfully extracted TIC, EIC, and DAD chromatograms
- Edge case: Empty scans cause IndexError - added guard
- Added test: tests/test_waters_reader.py::test_empty_scans
- Coverage: 100% maintained
- Committed: abc123def
```
