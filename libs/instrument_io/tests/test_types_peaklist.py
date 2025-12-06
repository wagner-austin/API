"""Tests for types.peaklist module."""

from __future__ import annotations

from instrument_io.types.peaklist import (
    AnnotatedMassPeak,
    AnnotatedMassPeakList,
    ChromatogramPeak,
    ChromatogramPeakList,
    MassPeak,
    MassPeakList,
    PeakListMeta,
)


def test_chromatogram_peak_construction() -> None:
    peak = ChromatogramPeak(
        peak_id=1,
        rt_start=5.25,
        rt_apex=5.5,
        rt_end=5.75,
        area=500000.0,
        height=100000.0,
        width_at_half_height=0.25,
    )
    assert peak["peak_id"] == 1
    assert peak["rt_apex"] == 5.5
    assert peak["height"] == 100000.0
    assert peak["area"] == 500000.0
    assert peak["width_at_half_height"] == 0.25


def test_peak_list_meta_construction() -> None:
    meta = PeakListMeta(
        source_path="/data/sample.D",
        num_peaks=10,
        processing_method="threshold",
    )
    assert meta["source_path"] == "/data/sample.D"
    assert meta["num_peaks"] == 10
    assert meta["processing_method"] == "threshold"


def test_chromatogram_peak_list_construction() -> None:
    meta = PeakListMeta(
        source_path="/test.D",
        num_peaks=2,
        processing_method="gradient",
    )
    peaks = [
        ChromatogramPeak(
            peak_id=1,
            rt_start=1.85,
            rt_apex=2.0,
            rt_end=2.15,
            area=200000.0,
            height=50000.0,
            width_at_half_height=0.15,
        ),
        ChromatogramPeak(
            peak_id=2,
            rt_start=4.8,
            rt_apex=5.0,
            rt_end=5.2,
            area=350000.0,
            height=80000.0,
            width_at_half_height=0.2,
        ),
    ]
    peak_list = ChromatogramPeakList(meta=meta, peaks=peaks)
    assert peak_list["meta"]["num_peaks"] == 2
    assert len(peak_list["peaks"]) == 2
    assert peak_list["peaks"][0]["rt_apex"] == 2.0


def test_mass_peak_construction() -> None:
    peak = MassPeak(
        mz=150.0789,
        intensity=100000.0,
        relative_intensity=100.0,
    )
    assert peak["mz"] == 150.0789
    assert peak["intensity"] == 100000.0
    assert peak["relative_intensity"] == 100.0


def test_mass_peak_list_construction() -> None:
    meta = PeakListMeta(
        source_path="/test.mzML",
        num_peaks=3,
        processing_method="centroid",
    )
    peaks = [
        MassPeak(mz=100.0, intensity=5000.0, relative_intensity=50.0),
        MassPeak(mz=150.0, intensity=10000.0, relative_intensity=100.0),
        MassPeak(mz=200.0, intensity=3000.0, relative_intensity=30.0),
    ]
    peak_list = MassPeakList(
        meta=meta,
        scan_number=1,
        retention_time=5.0,
        peaks=peaks,
    )
    assert peak_list["meta"]["num_peaks"] == 3
    assert len(peak_list["peaks"]) == 3
    assert peak_list["scan_number"] == 1
    assert peak_list["retention_time"] == 5.0


def test_annotated_mass_peak_construction() -> None:
    peak = AnnotatedMassPeak(
        mz=150.0789,
        intensity=100000.0,
        relative_intensity=100.0,
        annotation="[M+H]+",
        mass_error_ppm=2.5,
    )
    assert peak["mz"] == 150.0789
    assert peak["annotation"] == "[M+H]+"
    assert peak["mass_error_ppm"] == 2.5


def test_annotated_mass_peak_with_none_values() -> None:
    peak = AnnotatedMassPeak(
        mz=150.0789,
        intensity=100000.0,
        relative_intensity=100.0,
        annotation=None,
        mass_error_ppm=None,
    )
    assert peak["annotation"] is None
    assert peak["mass_error_ppm"] is None


def test_annotated_mass_peak_list_construction() -> None:
    meta = PeakListMeta(
        source_path="/test.mzML",
        num_peaks=1,
        processing_method="centroid",
    )
    peaks = [
        AnnotatedMassPeak(
            mz=195.0877,
            intensity=50000.0,
            relative_intensity=100.0,
            annotation="Caffeine [M+H]+",
            mass_error_ppm=1.2,
        ),
    ]
    peak_list = AnnotatedMassPeakList(
        meta=meta,
        scan_number=10,
        retention_time=8.5,
        peaks=peaks,
    )
    assert peak_list["peaks"][0]["annotation"] == "Caffeine [M+H]+"
    assert peak_list["scan_number"] == 10
