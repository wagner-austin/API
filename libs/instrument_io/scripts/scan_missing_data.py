
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table

def scan_for_missing_data():
    console = Console()
    base_path = Path(r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab")
    
    # 1. Define Known/Processed Files (normalized relative strings)
    # These are the files we have explicitly written scripts to scrape/process.
    processed_files = {
        # Standards (extract_standards.py)
        "Notebooks/Jasmine OseiEnin Lab Notebook/Response factors.xlsx",
        "Current Projects/Soil VOC quantitation.xlsx",
        "Notebooks/Avisa Lab Notebook/Standard Calculations (1).xlsx",
        "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/8mix_calc.xlsx",
        "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/std_tidy.xlsx",
        
        # Inventories (create_2021_inventory_excel.py & clean_emily_data.py context)
        "Important Docs/Chemical Inventory/02252021-Chemical Inventory.xlsx",
        "Notebooks/Emily Truong Notebook/Chem_Inv.xlsx", # The source for 2025
        
        # Generated Outputs (We don't want to "process" our own reports)
        "Notebooks/Emily Truong Notebook/Chemical_Inventory_List_2025.xlsx",
        "Notebooks/Emily Truong Notebook/Chemical_Inventory_List_2021.xlsx",
        "Notebooks/Emily Truong Notebook/Chemical_Ghost_Inventory_Report_2025.xlsx",
        "Notebooks/Emily Truong Notebook/Chemical_Standards_List_2025.xlsx",
        "Notebooks/Emily Truong Notebook/Chemical_Standards_Audit_2025.xlsx",
    }

    # Normalize processed paths for comparison
    processed_paths = set()
    for p in processed_files:
        full = (base_path / p).resolve()
        processed_paths.add(str(full).lower())

    # 2. Find all Excel files
    console.print(f"[bold blue]Scanning directory:[/bold blue] {base_path}")
    all_excel = list(base_path.rglob("*.xlsx")) + list(base_path.rglob("*.xls"))
    
    candidates = []

    # Keywords that suggest a file might contain chemical data
    keywords = ["inventory", "chem", "std", "standard", "stock", "mix", "analyte", "compound", "calibration"]

    for f in all_excel:
        # Skip temporary files (~)
        if f.name.startswith("~"):
            continue
            
        f_str = str(f.resolve()).lower()
        
        # If it's already processed, skip
        if f_str in processed_paths:
            continue
            
        # Check for keywords in filename
        name_lower = f.name.lower()
        if any(kw in name_lower for kw in keywords):
            candidates.append(f)

    # 3. Report
    if not candidates:
        console.print("[green]No new potential inventory/standard files found![/green]")
        return

    table = Table(title=f"Potential Unprocessed Data Files ({len(candidates)})")
    table.add_column("File Name", style="cyan")
    table.add_column("Relative Path", style="dim")

    for c in candidates:
        try:
            rel_path = c.relative_to(base_path)
        except:
            rel_path = c
        table.add_row(c.name, str(rel_path))

    console.print(table)
    console.print("\n[yellow]Review this list. These files have names matching keywords like 'inventory' or 'standard' but haven't been explicitly processed yet.[/yellow]")

if __name__ == "__main__":
    scan_for_missing_data()
