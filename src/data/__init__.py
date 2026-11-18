from typing import Any

from src.data.dataset_builder import DatasetBuilder, DatasetBuilderConfig
from src.data.loader import (
    DataLoader,
    ProcessedDatasetBundle,
    RawDatasetBundle,
    load_processed_datasets,
    load_raw_datasets,
)

_PREPROCESSING_EXPORTS = {
    "ItemPreprocessor",
    "UserPreprocessor",
    "build_integer_lookup",
    "build_normalizer",
    "build_string_lookup",
}

__all__ = [
    "DataLoader",
    "DatasetBuilder",
    "DatasetBuilderConfig",
    "ItemPreprocessor",
    "ProcessedDatasetBundle",
    "RawDatasetBundle",
    "UserPreprocessor",
    "build_integer_lookup",
    "build_normalizer",
    "build_string_lookup",
    "load_processed_datasets",
    "load_raw_datasets",
]


def __getattr__(name: str) -> Any:
  if name in _PREPROCESSING_EXPORTS:
    from src.data import preprocessing

    return getattr(preprocessing, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
