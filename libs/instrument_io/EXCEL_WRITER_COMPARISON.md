# Excel Writer Comparison

## ExcelWriter (src/instrument_io/writers/excel.py)

### Strengths ✅
1. **Clean, Simple API**
   ```python
   writer = ExcelWriter()
   writer.write_sheet(rows, Path("output.xlsx"), "Data")
   # Done!
   ```

2. **Strictly Typed** - Protocol-based, no `Any` types
3. **Reusable** - Import once, use everywhere
4. **Automatic Features**
   - Auto-sized columns
   - Excel tables with `TableStyleMedium9`
   - Multiple sheets support
   - Safe table naming

4. **Consistent with Readers** - Same design pattern
5. **Error Handling** - Raises `WriterError` with clear messages
6. **Testable** - Easy to write unit tests

### Weaknesses ❌
1. **NO Custom Styling**
   - Can't change cell colors
   - Can't apply conditional formatting
   - Can't customize fonts (bold, colors, sizes)
   - Can't set cell alignment
   - Limited to default table styles

2. **NO Advanced Features**
   - Can't freeze panes
   - Can't merge cells
   - Can't add formulas
   - Can't set custom column widths per column

### Best For
- Quick data exports
- Standardized reports
- Simple tabular data
- When you want "just works" behavior

---

## Scripts (create_audit_excel.py, create_ghost_excel.py)

### Strengths ✅
1. **Full Styling Control**
   ```python
   # Conditional formatting with colors
   if "OK" in entry["Status"]:
       cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
       cell.font = Font(color="006100")
   elif "Warning" in entry["Status"]:
       cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
       cell.font = Font(color="9C5700")
   ```

2. **Advanced Features**
   - Custom table styles (`TableStyleMedium2`, `TableStyleMedium9`)
   - PatternFill for cell backgrounds
   - Font styling (bold, colors, sizes)
   - Cell alignment
   - Precise column width control

3. **Report-Quality Output**
   - Color-coded status indicators
   - Professional-looking reports
   - Visual hierarchy

### Weaknesses ❌
1. **Lots of Boilerplate**
   ```python
   wb = Workbook()
   ws = wb.active
   ws.title = "Report"

   # Write headers
   for col_idx, header in enumerate(headers, 1):
       ws.cell(row=1, column=col_idx, value=header)

   # Write data
   for row_idx, entry in enumerate(data, 2):
       ws.cell(row=row_idx, column=1, value=entry["name"])
       # ... repeat for each column

   # Create table
   ref = f"A1:G{last_row}"
   tab = Table(displayName="MyTable", ref=ref)
   # ... more setup

   # Auto-width columns
   for col in ws.columns:
       max_length = 0
       # ... calculate width

   wb.save(path)
   ```

2. **Code Duplication**
   - Same patterns repeated across scripts
   - Column width logic duplicated
   - Table creation duplicated

3. **Not Reusable**
   - Hard-coded for specific use cases
   - Can't easily adapt for new reports

4. **Less Type Safety**
   - Uses openpyxl types directly
   - No custom TypedDicts
   - Harder to validate data structure

5. **Harder to Maintain**
   - Changes require updating multiple scripts
   - More code to test

### Best For
- Styled reports with colors
- Conditional formatting needs
- One-off custom exports
- When appearance matters

---

## Side-by-Side Comparison

| Feature | ExcelWriter | Scripts |
|---------|-------------|---------|
| **API Simplicity** | ✅ 1 line | ❌ ~50 lines |
| **Type Safety** | ✅ Strict | ⚠️ Basic |
| **Reusability** | ✅ High | ❌ Low |
| **Auto Tables** | ✅ Yes | ✅ Yes |
| **Auto Column Width** | ✅ Yes | ✅ Yes (manual) |
| **Cell Colors** | ❌ No | ✅ Yes |
| **Conditional Formatting** | ❌ No | ✅ Yes |
| **Custom Fonts** | ❌ No | ✅ Yes |
| **Custom Alignment** | ❌ No | ✅ Yes |
| **Pattern Fills** | ❌ No | ✅ Yes |
| **Boilerplate Code** | ✅ Minimal | ❌ Extensive |
| **Code Duplication** | ✅ None | ❌ High |
| **Maintainability** | ✅ High | ⚠️ Medium |

---

## The Verdict

### For Data Export → **ExcelWriter Wins**
When you just need to dump data to Excel with basic formatting:
```python
writer = ExcelWriter()
writer.write_sheet(standards_data, Path("standards.xlsx"))
```
Clean, simple, done.

### For Styled Reports → **Scripts Win**
When you need color-coded status indicators, custom styling:
```python
# Create green cells for "OK", yellow for "Warning", red for "Critical"
status_cell.fill = PatternFill(start_color="C6EFCE", ...)
status_cell.font = Font(color="006100")
```
More code, but more control.

---

## The Ideal Solution

Enhance `ExcelWriter` to support styling while keeping the clean API:

```python
# Proposed API
writer = ExcelWriter()
writer.write_sheet(
    rows=audit_data,
    out_path=Path("audit.xlsx"),
    sheet_name="Audit",
    styles={
        "Status": lambda val: CellStyle(
            fill="green" if "OK" in val else "yellow" if "Warning" in val else "red",
            font_color="dark_green" if "OK" in val else "dark_yellow" else "dark_red"
        )
    }
)
```

This would give you:
- ✅ Clean API like ExcelWriter
- ✅ Styling power like scripts
- ✅ Type safety
- ✅ Reusability

---

## Current Recommendation

**Use ExcelWriter when:**
- Exporting data for analysis
- Creating simple tables
- You don't care about colors
- Speed matters (less code to write)

**Use openpyxl scripts when:**
- Creating reports for presentations
- Need visual indicators (red/yellow/green)
- Conditional formatting required
- Appearance is important

**Long-term goal:**
Enhance ExcelWriter to support styling, then deprecate the script pattern.
