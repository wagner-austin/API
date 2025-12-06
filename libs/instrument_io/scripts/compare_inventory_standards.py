
import polars as pl
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Fix Windows console encoding for Greek letters
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def compare_inventory_and_standards():
    base_path = Path(r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab\Notebooks\Emily Truong Notebook")
    inventory_excel_path = base_path / "Chemical_Inventory_List_2025.xlsx"
    standards_excel_path = base_path / "Chemical_Standards_List_2025.xlsx"

    # Load Data from Excel
    console = Console(force_terminal=True, legacy_windows=False)

    try:
        df_inventory = pl.read_excel(source=inventory_excel_path, engine="openpyxl")
        console.print(f"Loaded inventory from: {inventory_excel_path}")
    except Exception as e:
        console.print(f"[red]Error loading inventory Excel '{inventory_excel_path}': {e}[/red]")
        return
    
    try:
        df_standards = pl.read_excel(source=standards_excel_path, engine="openpyxl")
        console.print(f"Loaded standards from: {standards_excel_path}")
    except Exception as e:
        console.print(f"[red]Error loading standards Excel '{standards_excel_path}': {e}[/red]")
        return

    # Normalize Helper
    def normalize(name):
        if not name or not isinstance(name, str):
            return ""
        return name.strip().lower()

    # Extract chemical names from DataFrames and create sets for comparison
    # Handle potential missing 'Chemical Name' column gracefully
    inventory_chemicals_raw = df_inventory.get_column("Chemical Name").to_list() if "Chemical Name" in df_inventory.columns else []
    standards_chemicals_raw = df_standards.get_column("Chemical Name").to_list() if "Chemical Name" in df_standards.columns else []

    inv_map = {normalize(name): name for name in inventory_chemicals_raw if name}
    std_map = {normalize(name): name for name in standards_chemicals_raw if name}

    inv_names = set(inv_map.keys())
    std_names = set(std_map.keys())

    # Analysis
    common = inv_names.intersection(std_names)
    missing_in_inventory = std_names - inv_names
    no_standard = inv_names - std_names
    
    console.print(f"\n[bold blue]Cross-Reference Report[/bold blue]")
    console.print(f"Total Inventory Items: {len(inv_names)}")
    console.print(f"Total Standards: {len(std_names)}")
    console.print(f"Matches found: {len(common)}\n")

    # Table 1: Critical Gaps (Have Standard, No Inventory)
    table_missing = Table(title="[red]CRITICAL: Standards Missing from Inventory[/red]")
    table_missing.add_column("Chemical Name (Standard)", style="cyan")
    table_missing.add_column("Source", style="magenta")
    
    for name_norm in sorted(missing_in_inventory):
        # We need to find the original entry to get the source
        original_entry = next((item for item in standards_chemicals_raw if normalize(item) == name_norm), None)
        if original_entry:
            # Find the full row data from the DataFrame using the original name
            full_row_data = df_standards.filter(pl.col("Chemical Name") == original_entry).head(1)
            source = full_row_data.get_column("Source").item() if "Source" in full_row_data.columns else "N/A"
            table_missing.add_row(
                original_entry,
                source
            )
    
    if missing_in_inventory:
        console.print(table_missing)
    else:
        console.print("[green]Good news! All standards have corresponding inventory entries.[/green]")

    print("\n")

    # Table 2: Potential Opportunities (Have Inventory, No Standard)
    # (Showing top 15 to avoid spamming)
    table_opportunity = Table(title="[yellow]Inventory Items without Standards (Top 15)[/yellow]")
    table_opportunity.add_column("Chemical Name (Inventory)", style="green")
    table_opportunity.add_column("CAS", style="white")
    
    for i, name_norm in enumerate(sorted(no_standard)):
        if i >= 15: break
        # Find the original entry to get the CAS
        original_entry = next((item for item in inventory_chemicals_raw if normalize(item) == name_norm), None)
        if original_entry:
            full_row_data = df_inventory.filter(pl.col("Chemical Name") == original_entry).head(1)
            cas = full_row_data.get_column("CAS").item() if "CAS" in full_row_data.columns else "N/A"
            table_opportunity.add_row(
                original_entry,
                str(cas) # Ensure CAS is string for display
            )
        
    console.print(table_opportunity)
    if len(no_standard) > 15:
        console.print(f"... and {len(no_standard) - 15} more.")

if __name__ == "__main__":
    compare_inventory_and_standards()
