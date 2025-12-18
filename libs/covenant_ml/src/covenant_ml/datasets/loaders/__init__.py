"""Format-specific dataset loaders.

Provides loaders for different file formats (CSV, ARFF, Excel).
Each loader implements the DatasetLoaderProtocol.

Loaders:
    CSVLoader: Standard CSV files with single observation per entity.
    ARFFLoader: ARFF (Weka) format files.
    TimeSeriesCSVLoader: CSV files with multiple observations per entity over time.
"""

from covenant_ml.datasets.loaders.arff_loader import ARFFLoader
from covenant_ml.datasets.loaders.csv_loader import CSVLoader
from covenant_ml.datasets.loaders.timeseries_csv_loader import TimeSeriesCSVLoader

__all__ = [
    "ARFFLoader",
    "CSVLoader",
    "TimeSeriesCSVLoader",
]
