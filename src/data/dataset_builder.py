import json
import logging
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

try:
  from src.data.loader import DataLoader, ProcessedDatasetBundle, RawDatasetBundle
except ImportError:
  from loader import DataLoader, ProcessedDatasetBundle, RawDatasetBundle

logger = logging.getLogger(__name__)

DEFAULT_EVENT_WEIGHTS: dict[str, float] = {
    "view": 1.0,
    "wishlist": 2.0,
    "add_to_cart": 3.0,
    "rating": 4.0,
    "purchase": 5.0,
}


@dataclass(frozen=True)
class DatasetBuilderConfig:
   raw_dir: Path = Path("data/raw")
   processed_dir: Path = Path("data/processed")
   train_quantile: float = 0.70
   val_quantile: float = 0.85
   ipw_clip_max: float = 10.0
   ctr_smoothing_prior: float = 20.0
   high_rating_threshold: float = 4.0
   high_rating_bonus: float = 1.0
   event_weights: dict[str, float] = field(
      default_factory=lambda: dict(DEFAULT_EVENT_WEIGHTS) 
   )

   def __post_init__(self) -> None:
     if not (0.0 < self.train_quantile < self.val_quantile < 1.0):
        raise ValueError(
          "Quantiles must satisfy 0.0 < train_quantile < val_quantile < 1.0; "
          f"got train={self.train_quantile}, val={self.val_quantile}"
        )