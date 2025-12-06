#!/usr/bin/env python
"""Test script to read notebook files."""

from pathlib import Path
from instrument_io import DOCXReader, ExcelReader

# Test DOCXReader
print("=" * 80)
print("Testing DOCXReader on Emily's 2025 Notebook")
print("=" * 80)
docx_reader = DOCXReader()
emily_path = Path(r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab\Notebooks\Emily Truong Notebook\ETruong_E-Notebook 2025.docx")
if emily_path.exists():
    text = docx_reader.read_text(emily_path)
    print(f"Document length: {len(text)} characters")
    print("\nFirst 1500 characters:")
    print(text[:1500])
else:
    print(f"File not found: {emily_path}")

# Test ExcelReader on Jasmine's coding commands
print("\n" + "=" * 80)
print("Testing ExcelReader on Jasmine's Coding Commands")
print("=" * 80)
excel_reader = ExcelReader()
coding_path = Path(r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab\Notebooks\Jasmine OseiEnin Lab Notebook\Coding commands.xlsx")
if coding_path.exists():
    sheets = excel_reader.list_sheets(coding_path)
    print(f"Sheets: {sheets}")

    for sheet in sheets[:2]:  # Read first 2 sheets
        print(f"\n--- Sheet: {sheet} ---")
        data = excel_reader.read_sheet(coding_path, sheet)
        print(f"Rows: {len(data)}")
        if data:
            print(f"Columns: {list(data[0].keys())}")
            print("\nFirst 3 rows:")
            for i, row in enumerate(data[:3]):
                print(f"Row {i+1}: {row}")
else:
    print(f"File not found: {coding_path}")

# Test ExcelReader on GCMS Run log
print("\n" + "=" * 80)
print("Testing ExcelReader on Jasmine's GCMS Run 2024")
print("=" * 80)
gcms_path = Path(r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab\Notebooks\Jasmine OseiEnin Lab Notebook\GCMS_Run_2024.xlsx")
if gcms_path.exists():
    sheets = excel_reader.list_sheets(gcms_path)
    print(f"Sheets: {sheets}")

    if sheets:
        data = excel_reader.read_sheet(gcms_path, sheets[0])
        print(f"\nSheet '{sheets[0]}' has {len(data)} rows")
        if data:
            print(f"Columns: {list(data[0].keys())}")
            print("\nFirst 2 rows:")
            for row in data[:2]:
                print(row)
else:
    print(f"File not found: {gcms_path}")
