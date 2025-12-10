#!/usr/bin/env python
"""Comprehensive demo of all instrument_io writers.

Demonstrates:
- ExcelWriter: Multiple sheets, auto-table formatting, auto-width
- WordWriter: Headings, paragraphs (bold/italic), tables, lists, page breaks
- PDFWriter: All content types plus figures with captions, page sizes
"""

from __future__ import annotations

from pathlib import Path

from instrument_io import (
    DocumentContent,
    ExcelWriter,
    PDFWriter,
    WordWriter,
)


def create_sample_image(path: Path) -> None:
    """Create a realistic LC-MS/MS chromatogram image for figure demonstration."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Create time axis (0-10 minutes)
    time = np.linspace(0, 10, 1000)

    # Create gaussian peaks for different analytes
    def gaussian_peak(t: np.ndarray, center: float, height: float, width: float) -> np.ndarray:
        return height * np.exp(-((t - center) ** 2) / (2 * width**2))

    # Baseline with slight drift
    baseline = 1000 + 50 * np.sin(time * 0.5) + np.random.normal(0, 20, len(time))

    # Add analyte peaks
    signal = baseline.copy()
    signal += gaussian_peak(time, 2.3, 45000, 0.08)  # Compound A
    signal += gaussian_peak(time, 3.8, 62000, 0.10)  # Compound B
    signal += gaussian_peak(time, 5.1, 38000, 0.07)  # Internal Std
    signal += gaussian_peak(time, 6.9, 55000, 0.09)  # Compound C

    # Create figure
    _fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(time, signal, "b-", linewidth=0.8)
    ax.set_xlabel("Retention Time (min)", fontsize=10)
    ax.set_ylabel("Intensity (cps)", fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 70000)

    # Add peak labels
    ax.annotate("Compound A", xy=(2.3, 46000), ha="center", fontsize=8)
    ax.annotate("Compound B", xy=(3.8, 63000), ha="center", fontsize=8)
    ax.annotate("IS", xy=(5.1, 39000), ha="center", fontsize=8)
    ax.annotate("Compound C", xy=(6.9, 56000), ha="center", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def demo_excel_writer(output_dir: Path) -> None:
    """Demonstrate ExcelWriter capabilities."""
    print("=" * 60)
    print("ExcelWriter Demo")
    print("=" * 60)

    writer = ExcelWriter(auto_table=True, auto_width=True)

    # Sheet 1: Sample analysis results
    results_data = [
        {
            "Sample ID": "S001",
            "Compound": "Caffeine",
            "RT (min)": 3.21,
            "Area": 125432,
            "Concentration (ng/mL)": 125.4,
        },
        {
            "Sample ID": "S002",
            "Compound": "Caffeine",
            "RT (min)": 3.19,
            "Area": 98234,
            "Concentration (ng/mL)": 98.2,
        },
        {
            "Sample ID": "S003",
            "Compound": "Caffeine",
            "RT (min)": 3.22,
            "Area": 156789,
            "Concentration (ng/mL)": 156.8,
        },
        {
            "Sample ID": "S001",
            "Compound": "Theobromine",
            "RT (min)": 2.85,
            "Area": 45123,
            "Concentration (ng/mL)": 45.1,
        },
        {
            "Sample ID": "S002",
            "Compound": "Theobromine",
            "RT (min)": 2.83,
            "Area": 52341,
            "Concentration (ng/mL)": 52.3,
        },
        {
            "Sample ID": "S003",
            "Compound": "Theobromine",
            "RT (min)": 2.86,
            "Area": 38921,
            "Concentration (ng/mL)": 38.9,
        },
    ]

    # Sheet 2: Calibration curve data
    calibration_data = [
        {"Level": 1, "Concentration (ng/mL)": 10.0, "Response": 10234, "Accuracy (%)": 102.3},
        {"Level": 2, "Concentration (ng/mL)": 25.0, "Response": 25891, "Accuracy (%)": 103.6},
        {"Level": 3, "Concentration (ng/mL)": 50.0, "Response": 49123, "Accuracy (%)": 98.2},
        {"Level": 4, "Concentration (ng/mL)": 100.0, "Response": 101456, "Accuracy (%)": 101.5},
        {"Level": 5, "Concentration (ng/mL)": 250.0, "Response": 248901, "Accuracy (%)": 99.6},
        {"Level": 6, "Concentration (ng/mL)": 500.0, "Response": 502341, "Accuracy (%)": 100.5},
    ]

    # Sheet 3: Instrument parameters
    instrument_data = [
        {"Parameter": "Ion Source", "Value": "ESI+", "Unit": "-"},
        {"Parameter": "Capillary Voltage", "Value": 3500, "Unit": "V"},
        {"Parameter": "Nebulizer Pressure", "Value": 45, "Unit": "psi"},
        {"Parameter": "Gas Temperature", "Value": 300, "Unit": "°C"},
        {"Parameter": "Gas Flow", "Value": 10, "Unit": "L/min"},
        {"Parameter": "Collision Energy", "Value": 20, "Unit": "eV"},
    ]

    # Write multi-sheet workbook
    out_path = output_dir / "demo_analysis_results.xlsx"
    writer.write_sheets(
        {
            "Analysis Results": results_data,
            "Calibration": calibration_data,
            "Instrument Parameters": instrument_data,
        },
        out_path,
    )
    print(f"Created: {out_path}")
    print("  - Sheet 'Analysis Results': 6 rows of sample data")
    print("  - Sheet 'Calibration': 6-point calibration curve")
    print("  - Sheet 'Instrument Parameters': MS instrument settings")
    print()


def demo_word_writer(output_dir: Path) -> None:
    """Demonstrate WordWriter capabilities."""
    print("=" * 60)
    print("WordWriter Demo")
    print("=" * 60)

    content: DocumentContent = [
        # Title
        {"type": "heading", "text": "LC-MS/MS Method Validation Report", "level": 1},
        # Abstract
        {"type": "heading", "text": "Abstract", "level": 2},
        {
            "type": "paragraph",
            "text": "This report describes the validation of an LC-MS/MS method for the "
            "quantification of methylxanthines in biological matrices. The method "
            "demonstrates excellent linearity, precision, and accuracy within "
            "regulatory guidelines.",
            "bold": False,
            "italic": False,
        },
        # Introduction
        {"type": "heading", "text": "1. Introduction", "level": 2},
        {
            "type": "paragraph",
            "text": "Methylxanthines are a class of alkaloids that include caffeine, "
            "theobromine, and theophylline. These compounds are widely consumed "
            "and have significant pharmacological effects.",
            "bold": False,
            "italic": False,
        },
        {
            "type": "paragraph",
            "text": "Key objectives of this validation study:",
            "bold": True,
            "italic": False,
        },
        {
            "type": "list",
            "items": [
                "Establish linearity range (10-500 ng/mL)",
                "Determine precision (intra-day and inter-day)",
                "Assess accuracy at QC levels",
                "Evaluate matrix effects",
            ],
            "ordered": True,
        },
        # Methods
        {"type": "heading", "text": "2. Materials and Methods", "level": 2},
        {"type": "heading", "text": "2.1 Chemicals and Reagents", "level": 3},
        {
            "type": "paragraph",
            "text": "All reference standards were purchased from Sigma-Aldrich (purity >98%). "
            "HPLC-grade solvents were obtained from Fisher Scientific.",
            "bold": False,
            "italic": False,
        },
        {"type": "heading", "text": "2.2 Instrument Configuration", "level": 3},
        {
            "type": "table",
            "headers": ["Component", "Specification", "Vendor"],
            "rows": [
                {
                    "Component": "LC System",
                    "Specification": "1290 Infinity II",
                    "Vendor": "Agilent",
                },
                {"Component": "MS Detector", "Specification": "6495C QQQ", "Vendor": "Agilent"},
                {
                    "Component": "Column",
                    "Specification": "C18, 2.1×100mm, 1.8μm",
                    "Vendor": "Waters",
                },
                {
                    "Component": "Autosampler",
                    "Specification": "Multisampler, 4°C",
                    "Vendor": "Agilent",
                },
            ],
            "caption": "Table 1: Instrument configuration for LC-MS/MS analysis",
        },
        {"type": "heading", "text": "2.3 Chromatographic Conditions", "level": 3},
        {
            "type": "list",
            "items": [
                "Mobile Phase A: 0.1% formic acid in water",
                "Mobile Phase B: 0.1% formic acid in acetonitrile",
                "Flow rate: 0.4 mL/min",
                "Injection volume: 5 μL",
                "Column temperature: 40°C",
            ],
            "ordered": False,
        },
        # Results - Page break before
        {"type": "page_break"},
        {"type": "heading", "text": "3. Results", "level": 2},
        {"type": "heading", "text": "3.1 Calibration Curve", "level": 3},
        {
            "type": "paragraph",
            "text": "The calibration curve demonstrated excellent linearity (R² > 0.999) "
            "over the concentration range of 10-500 ng/mL.",
            "bold": False,
            "italic": False,
        },
        {
            "type": "table",
            "headers": ["Analyte", "Slope", "Intercept", "R²"],
            "rows": [
                {"Analyte": "Caffeine", "Slope": 1023.4, "Intercept": 234.5, "R²": 0.9998},
                {"Analyte": "Theobromine", "Slope": 892.1, "Intercept": 178.3, "R²": 0.9995},
                {"Analyte": "Theophylline", "Slope": 956.7, "Intercept": 201.2, "R²": 0.9997},
            ],
            "caption": "Table 2: Calibration curve regression parameters",
        },
        {"type": "heading", "text": "3.2 Precision and Accuracy", "level": 3},
        {
            "type": "paragraph",
            "text": "Intra-day and inter-day precision were evaluated at three QC levels "
            "(low, medium, high). All %CV values were below 15%, meeting "
            "FDA bioanalytical method validation criteria.",
            "bold": False,
            "italic": True,
        },
        # Conclusions
        {"type": "page_break"},
        {"type": "heading", "text": "4. Conclusions", "level": 2},
        {
            "type": "paragraph",
            "text": "The developed LC-MS/MS method meets all validation criteria and is "
            "suitable for routine analysis of methylxanthines in biological samples.",
            "bold": True,
            "italic": False,
        },
        # References
        {"type": "heading", "text": "5. References", "level": 2},
        {
            "type": "list",
            "items": [
                "FDA Guidance for Industry: Bioanalytical Method Validation (2018)",
                "ICH M10: Bioanalytical Method Validation (2019)",
                "Smith J, et al. J Chromatogr B. 2023;1234:123456",
            ],
            "ordered": True,
        },
    ]

    writer = WordWriter(title="Method Validation Report", author="Analytical Lab")
    out_path = output_dir / "demo_validation_report.docx"
    writer.write_document(content, out_path)

    print(f"Created: {out_path}")
    print("  - 3 heading levels (H1, H2, H3)")
    print("  - Regular, bold, and italic paragraphs")
    print("  - 2 tables with captions")
    print("  - Ordered and unordered lists")
    print("  - 2 page breaks")
    print()


def demo_pdf_writer(output_dir: Path, image_path: Path) -> None:
    """Demonstrate PDFWriter capabilities."""
    print("=" * 60)
    print("PDFWriter Demo")
    print("=" * 60)

    content: DocumentContent = [
        # Title
        {"type": "heading", "text": "Analytical Chemistry Research Report", "level": 1},
        # Introduction
        {"type": "heading", "text": "Executive Summary", "level": 2},
        {
            "type": "paragraph",
            "text": "This report presents the results of a comprehensive analytical study "
            "utilizing liquid chromatography-tandem mass spectrometry (LC-MS/MS) "
            "for the detection and quantification of target analytes.",
            "bold": False,
            "italic": False,
        },
        # Key findings with bold
        {
            "type": "paragraph",
            "text": "Key Findings:",
            "bold": True,
            "italic": False,
        },
        {
            "type": "list",
            "items": [
                "Method detection limit: 0.5 ng/mL",
                "Linear range: 1-1000 ng/mL (R² > 0.999)",
                "Intra-day precision: <5% CV",
                "Recovery: 95-105%",
            ],
            "ordered": False,
        },
        # Methodology
        {"type": "heading", "text": "Methodology", "level": 2},
        {
            "type": "paragraph",
            "text": "The analytical method was developed following FDA and EMA guidelines "
            "for bioanalytical method validation. Sample preparation involved "
            "protein precipitation followed by solid-phase extraction.",
            "bold": False,
            "italic": False,
        },
        # Numbered procedure
        {"type": "heading", "text": "Sample Preparation Protocol", "level": 3},
        {
            "type": "list",
            "items": [
                "Add 100 μL internal standard solution to 100 μL plasma sample",
                "Add 300 μL acetonitrile for protein precipitation",
                "Vortex for 30 seconds, centrifuge at 14,000 rpm for 10 minutes",
                "Transfer supernatant to SPE cartridge (pre-conditioned)",
                "Wash with 1 mL 5% methanol in water",
                "Elute with 1 mL methanol, evaporate to dryness",
                "Reconstitute in 100 μL mobile phase A",
            ],
            "ordered": True,
        },
        # Results section with table
        {"type": "heading", "text": "Results", "level": 2},
        {
            "type": "paragraph",
            "text": "Calibration curve parameters for all target analytes:",
            "bold": False,
            "italic": True,
        },
        {
            "type": "table",
            "headers": ["Analyte", "LLOQ (ng/mL)", "ULOQ (ng/mL)", "Slope", "R²"],
            "rows": [
                {
                    "Analyte": "Compound A",
                    "LLOQ (ng/mL)": 1.0,
                    "ULOQ (ng/mL)": 1000,
                    "Slope": 0.0234,
                    "R²": 0.9998,
                },
                {
                    "Analyte": "Compound B",
                    "LLOQ (ng/mL)": 0.5,
                    "ULOQ (ng/mL)": 500,
                    "Slope": 0.0456,
                    "R²": 0.9995,
                },
                {
                    "Analyte": "Compound C",
                    "LLOQ (ng/mL)": 2.0,
                    "ULOQ (ng/mL)": 2000,
                    "Slope": 0.0178,
                    "R²": 0.9997,
                },
                {
                    "Analyte": "Internal Std",
                    "LLOQ (ng/mL)": "-",
                    "ULOQ (ng/mL)": "-",
                    "Slope": "-",
                    "R²": "-",
                },
            ],
            "caption": "Table 1: Calibration curve regression parameters for target analytes",
        },
        # Figure demonstration
        {"type": "heading", "text": "Chromatographic Separation", "level": 3},
        {
            "type": "paragraph",
            "text": "Representative chromatogram showing baseline separation of all analytes:",
            "bold": False,
            "italic": False,
        },
        {
            "type": "figure",
            "path": image_path,
            "caption": "Figure 1: Representative LC-MS/MS chromatogram (MRM mode)",
            "width_inches": 5.0,
        },
        # Page break before QC data
        {"type": "page_break"},
        {"type": "heading", "text": "Quality Control Results", "level": 2},
        {
            "type": "table",
            "headers": [
                "QC Level",
                "Nominal (ng/mL)",
                "Mean Found",
                "Accuracy (%)",
                "Precision (%CV)",
            ],
            "rows": [
                {
                    "QC Level": "LLOQ",
                    "Nominal (ng/mL)": 1.0,
                    "Mean Found": 1.02,
                    "Accuracy (%)": 102.0,
                    "Precision (%CV)": 8.5,
                },
                {
                    "QC Level": "Low",
                    "Nominal (ng/mL)": 3.0,
                    "Mean Found": 2.95,
                    "Accuracy (%)": 98.3,
                    "Precision (%CV)": 4.2,
                },
                {
                    "QC Level": "Medium",
                    "Nominal (ng/mL)": 100,
                    "Mean Found": 101.5,
                    "Accuracy (%)": 101.5,
                    "Precision (%CV)": 3.1,
                },
                {
                    "QC Level": "High",
                    "Nominal (ng/mL)": 800,
                    "Mean Found": 792,
                    "Accuracy (%)": 99.0,
                    "Precision (%CV)": 2.8,
                },
            ],
            "caption": "Table 2: Quality control sample accuracy and precision (n=6 replicates)",
        },
        # Second figure with different width
        {
            "type": "paragraph",
            "text": "Stability evaluation results:",
            "bold": True,
            "italic": False,
        },
        {
            "type": "figure",
            "path": image_path,
            "caption": "Figure 2: Analyte stability under various storage conditions",
            "width_inches": 5.0,
        },
        {
            "type": "paragraph",
            "text": "Additional visualization:",
            "bold": False,
            "italic": False,
        },
        {
            "type": "figure",
            "path": image_path,
            "caption": "Figure 3: Supplementary chromatogram data",
            "width_inches": 5.0,
        },
        # Conclusions
        {"type": "page_break"},
        {"type": "heading", "text": "Conclusions", "level": 2},
        {
            "type": "paragraph",
            "text": "The validated LC-MS/MS method demonstrates excellent performance "
            "characteristics suitable for routine bioanalytical applications. "
            "All validation parameters meet regulatory acceptance criteria.",
            "bold": False,
            "italic": False,
        },
        # Bold italic combination
        {
            "type": "paragraph",
            "text": "This method is recommended for implementation in the clinical laboratory.",
            "bold": True,
            "italic": True,
        },
    ]

    # Letter size PDF
    writer_letter = PDFWriter(page_size="letter", margin_inches=1.0)
    out_path_letter = output_dir / "demo_research_report_letter.pdf"
    writer_letter.write_document(content, out_path_letter)
    print(f"Created: {out_path_letter}")
    print("  - Page size: Letter (8.5 x 11 inches)")

    # A4 size PDF
    writer_a4 = PDFWriter(page_size="a4", margin_inches=0.75)
    out_path_a4 = output_dir / "demo_research_report_a4.pdf"
    writer_a4.write_document(content, out_path_a4)
    print(f"Created: {out_path_a4}")
    print("  - Page size: A4 (210 x 297 mm)")

    print()
    print("PDF Features Demonstrated:")
    print("  - 3 heading levels")
    print("  - Regular, bold, italic, and bold+italic paragraphs")
    print("  - 2 tables with captions")
    print("  - Ordered and unordered lists")
    print("  - 3 figures (2 with captions, 1 without, various widths)")
    print("  - 2 page breaks")
    print("  - 2 page sizes (letter and A4)")
    print()


def main() -> None:
    """Run all writer demonstrations."""
    print()
    print("=" * 60)
    print("instrument_io Writers - Comprehensive Demo")
    print("=" * 60)
    print()

    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Create sample image for figures
    image_path = output_dir / "sample_chromatogram.png"
    create_sample_image(image_path)
    print(f"Created sample image: {image_path}")
    print()

    # Run demos
    demo_excel_writer(output_dir)
    demo_word_writer(output_dir)
    demo_pdf_writer(output_dir, image_path)

    # Summary
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"\nAll output files are in: {output_dir.absolute()}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
