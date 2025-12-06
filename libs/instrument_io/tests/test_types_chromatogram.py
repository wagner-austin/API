"""Tests for types.chromatogram module."""

from __future__ import annotations

from instrument_io.types.chromatogram import (
    ChromatogramData,
    ChromatogramMeta,
    ChromatogramStats,
    DADData,
    DADSlice,
    EICData,
    EICParams,
    TICData,
)


def test_chromatogram_meta_construction() -> None:
    meta = ChromatogramMeta(
        source_path="/data/sample.D",
        instrument="Agilent 1260",
        method_name="default.M",
        sample_name="Sample001",
        acquisition_date="2024-01-15",
        signal_type="TIC",
        detector="MS",
    )
    assert meta["source_path"] == "/data/sample.D"
    assert meta["instrument"] == "Agilent 1260"
    assert meta["signal_type"] == "TIC"


def test_chromatogram_data_construction() -> None:
    data = ChromatogramData(
        retention_times=[0.0, 1.0, 2.0, 3.0],
        intensities=[100.0, 250.0, 180.0, 120.0],
    )
    assert data["retention_times"] == [0.0, 1.0, 2.0, 3.0]
    assert data["intensities"] == [100.0, 250.0, 180.0, 120.0]


def test_chromatogram_stats_construction() -> None:
    stats = ChromatogramStats(
        num_points=100,
        rt_min=0.0,
        rt_max=30.0,
        rt_step_mean=0.3,
        intensity_min=0.0,
        intensity_max=1000000.0,
        intensity_mean=500000.0,
        intensity_p99=950000.0,
    )
    assert stats["num_points"] == 100
    assert stats["rt_max"] == 30.0
    assert stats["intensity_max"] == 1000000.0
    assert stats["rt_step_mean"] == 0.3
    assert stats["intensity_mean"] == 500000.0
    assert stats["intensity_p99"] == 950000.0


def test_tic_data_construction() -> None:
    meta = ChromatogramMeta(
        source_path="/test.D",
        instrument="",
        method_name="",
        sample_name="",
        acquisition_date="",
        signal_type="TIC",
        detector="MS",
    )
    data = ChromatogramData(
        retention_times=[1.0, 2.0],
        intensities=[100.0, 200.0],
    )
    stats = ChromatogramStats(
        num_points=2,
        rt_min=1.0,
        rt_max=2.0,
        rt_step_mean=1.0,
        intensity_min=100.0,
        intensity_max=200.0,
        intensity_mean=150.0,
        intensity_p99=200.0,
    )
    tic = TICData(meta=meta, data=data, stats=stats)
    assert tic["meta"]["signal_type"] == "TIC"
    assert tic["data"]["retention_times"] == [1.0, 2.0]


def test_eic_data_construction() -> None:
    meta = ChromatogramMeta(
        source_path="/test.D",
        instrument="",
        method_name="",
        sample_name="",
        acquisition_date="",
        signal_type="EIC",
        detector="MS",
    )
    params = EICParams(target_mz=150.0, mz_tolerance=0.5)
    data = ChromatogramData(
        retention_times=[1.0, 2.0],
        intensities=[50.0, 75.0],
    )
    stats = ChromatogramStats(
        num_points=2,
        rt_min=1.0,
        rt_max=2.0,
        rt_step_mean=1.0,
        intensity_min=50.0,
        intensity_max=75.0,
        intensity_mean=62.5,
        intensity_p99=75.0,
    )
    eic = EICData(meta=meta, params=params, data=data, stats=stats)
    assert eic["params"]["target_mz"] == 150.0
    assert eic["params"]["mz_tolerance"] == 0.5


def test_dad_slice_construction() -> None:
    meta = ChromatogramMeta(
        source_path="/test.D",
        instrument="",
        method_name="",
        sample_name="",
        acquisition_date="",
        signal_type="DAD",
        detector="DAD",
    )
    data = ChromatogramData(
        retention_times=[1.0, 2.0],
        intensities=[1000.0, 1500.0],
    )
    stats = ChromatogramStats(
        num_points=2,
        rt_min=1.0,
        rt_max=2.0,
        rt_step_mean=1.0,
        intensity_min=1000.0,
        intensity_max=1500.0,
        intensity_mean=1250.0,
        intensity_p99=1500.0,
    )
    dad_slice = DADSlice(
        meta=meta,
        wavelength_nm=254.0,
        data=data,
        stats=stats,
    )
    assert dad_slice["wavelength_nm"] == 254.0


def test_dad_data_construction() -> None:
    meta = ChromatogramMeta(
        source_path="/test.D",
        instrument="",
        method_name="",
        sample_name="",
        acquisition_date="",
        signal_type="DAD",
        detector="DAD",
    )
    dad = DADData(
        meta=meta,
        wavelengths=[200.0, 254.0, 280.0],
        retention_times=[1.0, 2.0],
        intensity_matrix=[[100.0, 150.0], [200.0, 250.0], [180.0, 220.0]],
    )
    assert dad["wavelengths"] == [200.0, 254.0, 280.0]
    assert len(dad["intensity_matrix"]) == 3
