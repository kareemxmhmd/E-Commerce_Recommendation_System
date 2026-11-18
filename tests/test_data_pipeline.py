import json
import unittest

import numpy as np
import polars as pl

from src.config import PathsConfig
from src.data.preprocessing import ItemPreprocessor, UserPreprocessor

_PATHS = PathsConfig()
_HAS_PROCESSED_DATA = (
    (_PATHS.processed_dir / "split_metadata.json").exists()
    and (_PATHS.processed_dir / "train_interactions.parquet").exists()
    and (_PATHS.processed_dir / "users_enriched.parquet").exists()
    and (_PATHS.processed_dir / "items_enriched.parquet").exists()
)


@unittest.skipUnless(
    _HAS_PROCESSED_DATA,
    "Requires processed Parquet datasets in data/processed.",
)
class TestDataPipeline(unittest.TestCase):

  def setUp(self) -> None:
    self.paths = PathsConfig()

  def test_chronological_split_has_zero_temporal_leakage(self) -> None:
    meta_path = self.paths.processed_dir / "split_metadata.json"
    self.assertTrue(meta_path.exists(), "split_metadata.json must exist")
    with open(meta_path, "r", encoding="utf-8") as f:
      meta = json.load(f)

    train_df = pl.read_parquet(
        self.paths.processed_dir / "train_interactions.parquet"
    )
    val_df = pl.read_parquet(
        self.paths.processed_dir / "val_interactions.parquet"
    )
    test_df = pl.read_parquet(
        self.paths.processed_dir / "test_interactions.parquet"
    )

    max_train_ts = train_df["timestamp"].max()
    min_val_ts = val_df["timestamp"].min()
    max_val_ts = val_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()

    self.assertLessEqual(max_train_ts, min_val_ts)
    self.assertLessEqual(max_val_ts, min_test_ts)
    total_rows = (
        meta["interactions"]["train_rows"]
        + meta["interactions"]["val_rows"]
        + meta["interactions"]["test_rows"]
    )
    self.assertEqual(len(train_df) + len(val_df) + len(test_df), total_rows)

  def test_user_and_item_preprocessors_fit_and_transform(self) -> None:
    users_df = pl.read_parquet(
        self.paths.processed_dir / "users_enriched.parquet"
    ).head(25)
    items_df = pl.read_parquet(
        self.paths.processed_dir / "items_enriched.parquet"
    ).head(25)

    user_prep = UserPreprocessor(user_df=users_df)
    user_inputs = {
        "user_id": users_df["user_id"].to_numpy().astype(np.int64),
        "city": users_df["city"].cast(pl.Utf8).to_numpy().astype(str),
        "device": users_df["device"].cast(pl.Utf8).to_numpy().astype(str),
        "language": users_df["language"].cast(pl.Utf8).to_numpy().astype(str),
    }
    user_out = user_prep(user_inputs)
    self.assertIn("user_idx", user_out)
    self.assertEqual(len(user_out["user_idx"]), 25)

    item_prep = ItemPreprocessor(item_df=items_df)
    item_inputs = {
        "item_id": items_df["item_id"].to_numpy().astype(np.int64),
        "category_id": items_df["category_id"].to_numpy().astype(np.int64),
        "root_category_id": (
            items_df["root_category_id"].to_numpy().astype(np.int64)
        ),
        "seller_id": items_df["seller_id"].to_numpy().astype(np.int64),
        "log_price": items_df["log_price"].to_numpy().astype(np.float32),
        "train_smoothed_ctr": (
            items_df["train_smoothed_ctr"].to_numpy().astype(np.float32)
        ),
        "train_avg_rating": (
            items_df["train_avg_rating"].to_numpy().astype(np.float32)
        ),
    }
    item_out = item_prep(item_inputs)
    self.assertIn("item_idx", item_out)
    self.assertIn("log_price_norm", item_out)
    self.assertEqual(len(item_out["item_idx"]), 25)


if __name__ == "__main__":
  unittest.main()
