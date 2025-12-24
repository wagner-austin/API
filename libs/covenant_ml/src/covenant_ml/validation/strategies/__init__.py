"""Cross-validation strategy implementations.

Provides pluggable CV splitter implementations that satisfy CVSplitterProtocol.
Each strategy can be registered in CVSplitterRegistry and used interchangeably.

Strategies:
- StratifiedKFoldSplitter: Maintains class proportions across folds
- GroupStratifiedKFoldSplitter: Group-aware stratified splitting
- ShuffleSplitSplitter: Random stratified splits with configurable sizes
- TimeSeriesSplitter: Temporal ordering preserved for time series data
"""

from .group_stratified_kfold import (
    GroupStratifiedKFoldSplitter,
    create_group_stratified_kfold_splitter,
)
from .shuffle_split import (
    ShuffleSplitSplitter,
    create_shuffle_split_splitter,
)
from .stratified_kfold import (
    StratifiedKFoldSplitter,
    create_stratified_kfold_splitter,
)
from .time_series import (
    TimeSeriesSplitter,
    create_time_series_splitter,
)

__all__ = [
    "GroupStratifiedKFoldSplitter",
    "ShuffleSplitSplitter",
    "StratifiedKFoldSplitter",
    "TimeSeriesSplitter",
    "create_group_stratified_kfold_splitter",
    "create_shuffle_split_splitter",
    "create_stratified_kfold_splitter",
    "create_time_series_splitter",
]
