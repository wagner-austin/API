import pdfplumber

pdf = pdfplumber.open("tests/fixtures/research_paper.pdf")
print(f"Total pages: {len(pdf.pages)}")

for i, page in enumerate(pdf.pages[:10]):
    text = page.extract_text()
    tables = page.extract_tables()
    print(f"Page {i + 1}: text={'Yes' if text else 'No'}, tables={len(tables)}")
    if tables:
        for j, table in enumerate(tables):
            print(f"  Table {j + 1}: {len(table)} rows x {len(table[0]) if table else 0} cols")

pdf.close()
