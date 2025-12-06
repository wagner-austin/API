# Faiola Lab Chemical Inventory & Standards Documentation

**Generated:** 2025-12-04
**Location:** `C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab`

---

## 📋 Chemical Inventory Files

### 1. Emily Truong - Chem_Inv.xlsx
**Path:** `Notebooks\Emily Truong Notebook\Chem_Inv.xlsx`
**Type:** Chemical Inventory (Current/Updated)
**Sheets:** 1 (Sheet1)
**Total Entries:** 129 chemicals

**Format:** Compound-focused listing with verification status
**Sample Chemicals:**
- (+)-Aromadendrene ✓ Verified
  - CAS: 489-39-4
  - Physical State: liquid
  - GHS: H227
  - Containers: 1

- (+)-Camphor ✓ Verified
  - CAS: 464-49-3
  - Physical State: solid
  - GHS: H228, H302, H315, H319, H335, H402
  - Containers: 1

**Notes:** This appears to be the most current chemical inventory maintained by Emily Truong.

---

### 2. Main Chemical Inventory (February 2021)
**Path:** `Important Docs\Chemical Inventory\02252021-Chemical Inventory.xlsx`
**Type:** Official UCI Chemical Inventory
**Last Updated:** February 25, 2021
**Sheets:** 2 (CiBR-Trac, 428)

#### Sheet: CiBR-Trac (12 chemicals)
**Columns:**
- id, Chemical_Name, CAS, Container_Size, Units, Container_Number
- Chemical_Physical_State, Container_Type, HazMat_Type
- Storage_Pressure, Storage_Temperature

**Sample Entries:**
1. **1-BUTANOL**
   - CAS: 71-36-3
   - Size: 1 L
   - Type: Liquid, Glass Bottle
   - Storage: Ambient temperature/pressure

2. **ACETONITRILE**
   - CAS: 75-05-8
   - Size: 4 L
   - Type: Liquid, Glass Bottle
   - Storage: Ambient temperature/pressure

3. **Air, compressed**
   - CAS: 132259-10-0
   - Size: 250 ft³ (2 cylinders)
   - Type: Gas, Cylinder
   - Storage: Ambient temperature/pressure

#### Sheet: 428 (27 chemicals - Room 0428 specific)
**Columns:**
- Chemical Name, Substance Name, CAS, Molecular Formula
- Physical State, Room Number, Sublocation
- Size, Amount, Units, Container Type
- Concentration, Barcode

**Sample Entries:**
1. **Toluene**
   - CAS: 108-88-3
   - Formula: C7H8
   - Location: Room 0428, fridge
   - Size: 500 mL, Glass Bottle
   - Barcode: CA000000000000000003F908

2. **Isoprene (≥99%)**
   - CAS: 78-79-5
   - Formula: C5H8
   - Location: Room 0428, fridge
   - Size: 1000 mL, Glass Bottle
   - Barcode: CA000000000000000003F909

3. **Eucalyptol (99%)**
   - CAS: 470-82-6
   - Formula: C10H18O
   - Location: Room 0428, fridge
   - Size: 100 mL, Glass Bottle
   - Barcode: CA000000000000000003F90A

---

## 🧪 Standards & Calibration Files

### 3. Jasmine OseiEnin - Response Factors.xlsx
**Path:** `Notebooks\Jasmine OseiEnin Lab Notebook\Response factors.xlsx`
**Type:** GC-MS Response Factor Calculations
**Sheets:** 2 (Template (1ul), JO_STD013024)

#### Sheet: Template (1ul) - Standard Preparation Template (9 compounds)
**Columns:**
- Chemical name, Density (g/mL)
- Volume in Standard (µL) - based on bottle
- Mass in Standard (g)
- Standard Total Volume (mL)
- Mass in Total (g/mL)
- Volume used (µL) - Based on syringe
- Mass used (g), Mass used (ng)

**Standard Compounds:**
1. **alpha-pinene**
   - Density: 0.858 g/mL
   - Volume in std: 1 µL
   - Total volume: 40 mL
   - Mass concentration: 21.45 µg/mL
   - Mass used (1 µL): 21.45 ng

2. **beta-pinene**
   - Density: 0.872 g/mL
   - Volume in std: 1 µL
   - Total volume: 40 mL
   - Mass concentration: 21.8 µg/mL
   - Mass used (1 µL): 21.8 ng

3. **bisabolene**
   - Density: 0.89 g/mL
   - Volume in std: 1 µL
   - Total volume: 40 mL
   - Mass concentration: 22.25 µg/mL
   - Mass used (1 µL): 22.25 ng

#### Sheet: JO_STD013024 - January 30, 2024 Standard (66 entries)
Multi-level calibration with varying injection volumes (1 µL, 2 µL, etc.) for:
- alpha-pinene
- beta-pinene
- Multiple terpene standards

**Purpose:** GC-MS response factor calibration for VOC quantitation

---

### 4. Jasmine OseiEnin - GCMS_Run_2024.xlsx
**Path:** `Notebooks\Jasmine OseiEnin Lab Notebook\GCMS_Run_2024.xlsx`
**Type:** GC-MS Run Log
**Sheets:** 14 sheets (Template + dated runs)

**Date Range:** August 2024 - October 2024
**Total Runs Tracked:** 13 runs

**Sample Runs:**
- 091624: 20 cartridges
- 090824: 22 cartridges
- 100624: 17 cartridges

**Format:** Run logs tracking:
- Start/finish times
- Total number of cartridges analyzed
- Sample tracking per run

---

### 5. Avisa Lab Notebook - Standard Calculations (1).xlsx
**Path:** `Notebooks\Avisa Lab Notebook\Standard Calculations (1).xlsx`
**Type:** Standard Dilution Calculations
**Sheets:** 1 (Sheet1)
**Rows:** 27 calculations

**Columns:**
- Compound
- Concentration (ppm)
- Volume (µL)
- Dilution factor
- Final concentration

**Purpose:** Documents standard dilution series for limonene chirality experiments and volatility studies.

---

### 6. Soil VOC Quantitation.xlsx
**Path:** `Current Projects\Soil VOC quantitation.xlsx`
**Type:** VOC Quantitation Data & Calibration
**Sheets:** Multiple (project-specific)

**Content:**
- Calibration curves for soil VOC analysis
- Quantitation data
- Standards used for soil volatiles project

---

## 📚 Related SOP Documents (Word/PowerPoint)

The `instrument_io` library was successfully used to read the content of the SOP documents.

### Standards Preparation SOPs
**Path:** `Important Docs\Faiola Lab SOPs\TDGCMS\`

1.  **Standard Preparation.docx**
    -   Protocol for preparing analytical standards
    -   Standard stock solutions
    -   Dilution procedures

2.  **Standard Cartridge Loading Protocol.docx**
    -   Loading standards onto TD cartridges
    -   Procedure for GC-MS calibration

3.  **TDGCMS_Calculation.pptx**
    -   Calculations for TD-GC-MS
    -   Calibration curve generation
    -   Response factor calculations

### Other Chemical Documentation
**Path:** `Important Docs\Chemical Inventory\`

1.  **CHEMICAL SEGREGATION GUIDELINES_01SEP2020.docx**
    -   Chemical storage compatibility
    -   Segregation requirements

2.  **Feb 2021_UC Chemicals User Quick Reference.docx**
    -   UC chemical management system guide
    -   CiBR-Trac usage instructions

### HPLC Standards
**Path:** `Important Docs\Faiola Lab SOPs\Aerosol_Composition_SamplePrep\`

1.  **Sample Extraction for HPLC runs.docx**
    -   HPLC sample preparation
    -   Standards for aerosol composition analysis

---

## 🔗 Online Resources

### Emily Truong - Standards Notebook (OneNote)
**URL:** `https://ucirvine-my.sharepoint.com/personal/cfaiola_ad_uci_edu/...`
**Type:** SharePoint OneNote
**Access:** Requires UCI credentials
**Content:** Detailed standards notebook (online, not downloaded locally)

### Jack Richards - Lab Notebook (OneNote)
**URL:** `https://ucirvine-my.sharepoint.com/personal/cfaiola_ad_uci_edu/...`
**Type:** SharePoint OneNote
**Access:** Requires UCI credentials

---

## 📊 Summary Statistics

**Total Chemical Inventory Files:** 2 Excel files (129 + 39 chemicals = 168 total entries)
**Standards/Calibration Files:** 4 Excel files
**SOP Documents (unread):** 6 Word/PowerPoint files
**Online Resources:** 2 OneNote notebooks

**Key Standards Tracked:**
- Terpenes (alpha-pinene, beta-pinene, bisabolene, etc.)
- Solvents (toluene, acetonitrile, 1-butanol)
- VOCs (isoprene, eucalyptol)

**Storage Locations:**
- Room 0428 (main lab storage)
- CiBR-Trac system (UCI chemical tracking)
- Fridge storage (volatile compounds)

---

## 🔬 File Content Analysis (2025-12-05)

This section summarizes the findings from a detailed analysis of the files listed above, performed using the `instrument_io` library.

### Excel Files (`.xlsx`)

-   **`Chem_Inv.xlsx`**: This file is not in a standard tabular format. It appears to be a free-form text file, and the `ExcelReader` could not parse it into a structured format. Further processing would require a custom parsing script.

-   **`02252021-Chemical Inventory.xlsx`**: This is a well-structured Excel file with two sheets:
    -   `CiBR-Trac`: Contains chemical inventory data with clear headers.
    -   `428`: Contains room-specific chemical inventory data, also well-structured.

-   **`Response factors.xlsx`**: This file is well-structured and contains two sheets for response factor calculations:
    -   `Template (1ul)`: A template for standard preparation.
    -   `JO_STD013024`: Data for a specific standard from January 30, 2024.

-   **`GCMS_Run_2024.xlsx`**: This workbook contains many sheets.
    -   The `Template` sheet defines the data structure for the run logs.
    -   The data sheets (e.g., `101724`) appear to be mostly empty or use a non-standard header position, preventing easy data extraction with default methods.

-   **`Standard Calculations (1).xlsx`**: The sheets in this workbook have a complex, non-tabular structure that is not easily parsed by automated tools.

-   **`Soil VOC quantitation.xlsx`**: This workbook contains many sheets, but the `Standard list` sheet is well-structured and provides a clear list of standards used.

### SOP & Other Documents (`.docx`, `.pptx`)

The `instrument_io` library was successfully used to read the content of the SOP documents.

-   **`Standard Preparation.docx`**: Successfully read. Contains detailed instructions for preparing chemical standards, including compound details, densities, and dilution calculations.

-   **`Standard Cartridge Loading Protocol.docx`**: Successfully read. Provides a step-by-step guide for loading standard cartridges for analysis.

-   **`TDGCMS_Calculation.pptx`**: Successfully read. Explains the full process of calculating emission rates from TDGCMS data, including response factors and quality control.

-   **`CHEMICAL SEGREGATION GUIDELINES_01SEP2020.docx`**: Successfully read. The primary content is a comprehensive table detailing chemical segregation and storage guidelines.

-   **`Feb 2021_UC Chemicals User Quick Reference.docx`**: Successfully read. A user guide for the UC Chemicals inventory management system.

-   **`Sample Extraction for HPLC runs.docx`**: Successfully read. An SOP for preparing Teflon filter samples for HPLC-PDA-ESIMS analysis.

---

## ⚠️ Notes

1.  Emily's `Chem_Inv.xlsx` appears to be more current than the 2021 official inventory but is not in a machine-readable format.
2.  Multiple standards notebooks are maintained by different lab members.
3.  Online OneNote notebooks contain additional standards information and require credentials to access.
4.  Response factors and calibration data are maintained in Jasmine's notebook.
5.  Standard preparation follows systematic dilution procedures (40 mL total volumes common).

---

## 🔄 Next Steps

1.  Develop a custom parser for the poorly-structured Excel files (`Chem_Inv.xlsx`, `Standard Calculations (1).xlsx`).
2.  Cross-reference Emily's inventory with the 2021 official inventory once `Chem_Inv.xlsx` is parsed.
3.  Consolidate response factors and standards information from all relevant files into a unified dataset.
4.  Fully parse and extract structured data from the `GCMS_Run_2024.xlsx` run logs.
