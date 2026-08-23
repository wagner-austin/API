"""ClearGBM-versus-LightGBM benchmarking subsystem.

A measurement harness for comparing the two gradient-boosting learners this
library depends on, built so its numbers stay comparable across sessions and
machines.

Three properties of the protocol are deliberate, because each one has
previously produced a wrong conclusion when absent:

* **Both learners are measured in the same run.** A fixed reference measured
  now is the only way to separate a code change from a machine-state change.
  A LightGBM number carried forward from an older manifest cannot do that,
  and dividing a fresh time by a stale one manufactures a gap that is not
  there.
* **The canonical statistic is the median, not the minimum.** The first fits
  after an idle period run with full turbo headroom -- a different power
  regime, not noise -- so a minimum reports a cold-start outlier in place of
  the steady state that sustained training experiences.
* **Results are normalized by tree size.** At a fixed ``max_depth`` a
  depth-wise learner grows a full balanced tree while a leaf-wise one stops
  at ``num_leaves``. A raw wall-clock ratio conflates "slower per unit of
  work" with "doing more work per tree"; the per-leaf ratio does not.

The layers are independent: :mod:`types` defines the record shapes and their
codecs, :mod:`protocols` names every injected boundary, :mod:`timing`,
:mod:`splitting`, :mod:`quality`, :mod:`model_shape` and :mod:`reporting`
are pure, :mod:`dataset` is the only module that touches a file,
:mod:`adapters` holds the two concrete learners, :mod:`runner` owns the
measurement protocol, and :mod:`factory` is the only place that names a
concrete implementation.
"""

from __future__ import annotations

from covenant_ml.benchmarking.types import (
    MANIFEST_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkManifest,
    BenchmarkModelName,
    DatasetInfo,
    QualityMetrics,
    SeedResult,
    TimingSummary,
)
from covenant_ml.benchmarking.types_codec import (
    decode_benchmark_manifest,
    encode_benchmark_manifest,
)

from .dataset import LoadedDataset, load_bankruptcy_dataset
from .factory import (
    DEFAULT_REPEATS,
    DEFAULT_SEEDS,
    DEFAULT_WARMUPS,
    make_baseline_trainers,
    make_benchmark_config,
    make_split_factory,
    make_trainers,
)
from .multiclass_quality import (
    MulticlassArmResult,
    MulticlassBenchConfig,
    MulticlassManifest,
    MulticlassQuality,
    encode_multiclass_manifest,
    make_synthetic_multiclass,
    run_multiclass_benchmark,
)
from .power import disable_power_throttling, opt_out_of_power_throttling
from .protocols import (
    DataSplit,
    PowerThrottlingOptOutProto,
    ProcessInformationSetterProto,
    SplitFactoryProto,
    TrainedModelProto,
    TrainerProto,
)
from .quality import compute_quality
from .reporting import GapSummary, ModelSummary, render_report, summarize_gap
from .runner import measure_trainer, run_benchmark
from .splitting import company_disjoint_split
from .timing import summarize_timings

__all__ = [
    "DEFAULT_REPEATS",
    "DEFAULT_SEEDS",
    "DEFAULT_WARMUPS",
    "MANIFEST_SCHEMA_VERSION",
    "BenchmarkConfig",
    "BenchmarkManifest",
    "BenchmarkModelName",
    "DataSplit",
    "DatasetInfo",
    "GapSummary",
    "LoadedDataset",
    "ModelSummary",
    "MulticlassArmResult",
    "MulticlassBenchConfig",
    "MulticlassManifest",
    "MulticlassQuality",
    "PowerThrottlingOptOutProto",
    "ProcessInformationSetterProto",
    "QualityMetrics",
    "SeedResult",
    "SplitFactoryProto",
    "TimingSummary",
    "TrainedModelProto",
    "TrainerProto",
    "company_disjoint_split",
    "compute_quality",
    "decode_benchmark_manifest",
    "disable_power_throttling",
    "encode_benchmark_manifest",
    "encode_multiclass_manifest",
    "load_bankruptcy_dataset",
    "make_baseline_trainers",
    "make_benchmark_config",
    "make_split_factory",
    "make_synthetic_multiclass",
    "make_trainers",
    "measure_trainer",
    "opt_out_of_power_throttling",
    "render_report",
    "run_benchmark",
    "run_multiclass_benchmark",
    "summarize_gap",
    "summarize_timings",
]
