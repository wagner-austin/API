"""Tests for types.spectrum module."""

from __future__ import annotations

from instrument_io.types.spectrum import (
    MS2Spectrum,
    MS3Spectrum,
    MSSpectrum,
    PrecursorInfo,
    SpectrumData,
    SpectrumMeta,
    SpectrumStats,
)


def test_spectrum_meta_construction() -> None:
    meta = SpectrumMeta(
        source_path="/data/sample.mzML",
        scan_number=100,
        retention_time=5.5,
        ms_level=1,
        polarity="positive",
        total_ion_current=1000000.0,
    )
    assert meta["source_path"] == "/data/sample.mzML"
    assert meta["scan_number"] == 100
    assert meta["retention_time"] == 5.5
    assert meta["ms_level"] == 1
    assert meta["polarity"] == "positive"


def test_spectrum_data_construction() -> None:
    data = SpectrumData(
        mz_values=[100.0, 150.0, 200.0],
        intensities=[1000.0, 5000.0, 2000.0],
    )
    assert data["mz_values"] == [100.0, 150.0, 200.0]
    assert data["intensities"] == [1000.0, 5000.0, 2000.0]


def test_spectrum_stats_construction() -> None:
    stats = SpectrumStats(
        num_peaks=50,
        mz_min=100.0,
        mz_max=1000.0,
        base_peak_mz=450.0,
        base_peak_intensity=100000.0,
    )
    assert stats["num_peaks"] == 50
    assert stats["mz_min"] == 100.0
    assert stats["base_peak_mz"] == 450.0


def test_ms_spectrum_construction() -> None:
    meta = SpectrumMeta(
        source_path="/test.mzML",
        scan_number=1,
        retention_time=1.0,
        ms_level=1,
        polarity="positive",
        total_ion_current=500000.0,
    )
    data = SpectrumData(
        mz_values=[100.0, 200.0],
        intensities=[1000.0, 2000.0],
    )
    stats = SpectrumStats(
        num_peaks=2,
        mz_min=100.0,
        mz_max=200.0,
        base_peak_mz=200.0,
        base_peak_intensity=2000.0,
    )
    spectrum = MSSpectrum(meta=meta, data=data, stats=stats)
    assert spectrum["meta"]["ms_level"] == 1
    assert spectrum["data"]["mz_values"] == [100.0, 200.0]


def test_precursor_info_construction() -> None:
    precursor = PrecursorInfo(
        mz=500.5,
        charge=2,
        intensity=50000.0,
        isolation_window=1.0,
    )
    assert precursor["mz"] == 500.5
    assert precursor["charge"] == 2
    assert precursor["intensity"] == 50000.0


def test_precursor_info_with_none_values() -> None:
    precursor = PrecursorInfo(
        mz=500.5,
        charge=None,
        intensity=None,
        isolation_window=None,
    )
    assert precursor["mz"] == 500.5
    assert precursor["charge"] is None


def test_ms2_spectrum_construction() -> None:
    meta = SpectrumMeta(
        source_path="/test.mzML",
        scan_number=10,
        retention_time=2.5,
        ms_level=2,
        polarity="positive",
        total_ion_current=100000.0,
    )
    precursor = PrecursorInfo(
        mz=450.0,
        charge=2,
        intensity=50000.0,
        isolation_window=1.5,
    )
    data = SpectrumData(
        mz_values=[150.0, 250.0, 350.0],
        intensities=[5000.0, 10000.0, 3000.0],
    )
    stats = SpectrumStats(
        num_peaks=3,
        mz_min=150.0,
        mz_max=350.0,
        base_peak_mz=250.0,
        base_peak_intensity=10000.0,
    )
    spectrum = MS2Spectrum(meta=meta, precursor=precursor, data=data, stats=stats)
    assert spectrum["meta"]["ms_level"] == 2
    assert spectrum["precursor"]["mz"] == 450.0
    assert spectrum["precursor"]["charge"] == 2


def test_ms3_spectrum_construction() -> None:
    meta = SpectrumMeta(
        source_path="/test.mzML",
        scan_number=15,
        retention_time=3.0,
        ms_level=3,
        polarity="positive",
        total_ion_current=50000.0,
    )
    # MS3 has a precursor chain: parent (MS1->MS2) -> child (MS2->MS3)
    parent_precursor = PrecursorInfo(
        mz=500.0,
        charge=2,
        intensity=100000.0,
        isolation_window=2.0,
    )
    child_precursor = PrecursorInfo(
        mz=300.0,
        charge=1,
        intensity=25000.0,
        isolation_window=1.0,
    )
    data = SpectrumData(
        mz_values=[100.0, 150.0],
        intensities=[2000.0, 3000.0],
    )
    stats = SpectrumStats(
        num_peaks=2,
        mz_min=100.0,
        mz_max=150.0,
        base_peak_mz=150.0,
        base_peak_intensity=3000.0,
    )
    # precursors list contains the chain from parent to child
    spectrum = MS3Spectrum(
        meta=meta,
        precursors=[parent_precursor, child_precursor],
        data=data,
        stats=stats,
    )
    assert spectrum["meta"]["ms_level"] == 3
    assert spectrum["precursors"][0]["mz"] == 500.0  # parent
    assert spectrum["precursors"][1]["mz"] == 300.0  # child
