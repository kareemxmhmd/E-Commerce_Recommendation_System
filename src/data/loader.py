import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


EXPECTED_RAW_SCHEMAS: dict[str, set[str]] = {
    "items": {
        "item_id",
        "title",
        "category_id",
        "brand",
        "price",
        "seller_id",
        "created_at",
        "stock",
    },
    "users": {
        "user_id",
        "signup_date",
        "city",
        "device",
        "language",
    },
    "interactions": {
        "user_id",
        "item_id",
        "event_type",
        "timestamp",
        "session_id",
        "rating",
    },
    "impressions": {
        "user_id",
        "item_id",
        "session_id",
        "position",
        "clicked",
        "timestamp",
    },
    "categories": {
        "category_id",
        "name",
        "parent_id",
    },
    "sellers": {
        "seller_id",
        "name",
        "rating",
        "joined_at",
    },
}


@dataclass(frozen=True)
class RawDatasetBundle:
    
    items: pl.DataFrame
    users: pl.DataFrame
    interactions: pl.DataFrame
    impressions: pl.DataFrame
    categories: pl.DataFrame
    sellers: pl.DataFrame

    def as_dict(self) -> dict[str, pl.DataFrame]:
        return {
            "items": self.items,
            "users": self.users,
            "interactions": self.interactions,
            "impressions": self.impressions,
            "categories": self.categories,
            "sellers": self.sellers,
        }


@dataclass(frozen=True)
class ProcessedDatasetBundle:
    
    train_interactions: pl.DataFrame
    val_interactions: pl.DataFrame
    test_interactions: pl.DataFrame
    train_impressions: pl.DataFrame
    val_impressions: pl.DataFrame
    test_impressions: pl.DataFrame
    train_user_item_pairs: pl.DataFrame
    items_enriched: pl.DataFrame
    users_enriched: pl.DataFrame
    position_propensities: pl.DataFrame
    metadata: dict[str, Any]


class DataLoader:

  def __init__(
      self,
      raw_dir: str | Path = "data/raw",
      processed_dir: str | Path = "data/processed",
  ) -> None:
    self.raw_dir = Path(raw_dir)
    self.processed_dir = Path(processed_dir)

  def load_raw_table(self, table_name: str) -> pl.DataFrame:

     if table_name not in EXPECTED_RAW_SCHEMAS:
        raise ValueError(
           f"Unknown raw table '{table_name}'. Expected one of:"
           f" {sorted(EXPECTED_RAW_SCHEMAS)}"
        )

     file_path = self.raw_dir / f"{table_name}.parquet"
     if not file_path.exists():
       raise FileNotFoundError(f"Raw Parquet file not found: {file_path}")

     df = pl.read_parquet(file_path)
     missing_cols = EXPECTED_RAW_SCHEMAS[table_name] - set(df.columns)
     if missing_cols:
        raise ValueError(
           f"Table '{table_name}' is missing required columns:"
           f" {sorted(missing_cols)}"
        )

     if df.height == 0:
        raise ValueError(f"Table '{table_name}' at {file_path} is empty.")

     return df

  def load_raw(self, validate_integrity: bool = True) -> RawDatasetBundle:
     
     bundle = RawDatasetBundle(
        items=self.load_raw_table("items"),
        users=self.load_raw_table("users"),
        interactions=self.load_raw_table("interactions"),
        impressions=self.load_raw_table("impressions"),
        categories=self.load_raw_table("categories"),
        sellers=self.load_raw_table("sellers"),
    )

     if validate_integrity:
        self.validate_raw_integrity(bundle)
        
     logger.info(
        "Loaded raw datasets: items=%s, users=%s, interactions=%s,"
        " impressions=%s, categories=%s, sellers=%s",
        bundle.items.shape,
        bundle.users.shape,
        bundle.interactions.shape,
        bundle.impressions.shape,
        bundle.categories.shape,
        bundle.sellers.shape,
    )
     return bundle

  @staticmethod
  def validate_raw_integrity(bundle: RawDatasetBundle) -> None:
     
     for name, df in bundle.as_dict().items():
        dup_count = int(df.is_duplicated().sum())
        if dup_count > 0:
           raise ValueError(f"Raw table '{name}' contains {dup_count} duplicate rows.")

        fk_violations = {
           "interactions.user_id -> users.user_id": (
              bundle.interactions.filter(
                 ~pl.col("user_id").is_in(bundle.users["user_id"])
              ).height
           ),
           "interactions.item_id -> items.item_id": (
              bundle.interactions.filter(
                 ~pl.col("item_id").is_in(bundle.items["item_id"])
              ).height
           ),
           "impressions.user_id -> users.user_id": (
              bundle.impressions.filter(
                 ~pl.col("user_id").is_in(bundle.users["user_id"])
              ).height
           ),
           "impressions.item_id -> items.item_id": (
              bundle.impressions.filter(
                 ~pl.col("item_id").is_in(bundle.items["item_id"])
              ).height
           ),
           "items.category_id -> categories.category_id": (
              bundle.items.filter(
                 ~pl.col("category_id").is_in(bundle.categories["category_id"])
              ).height
           ),
           "items.seller_id -> sellers.seller_id": (
              bundle.items.filter(
                 ~pl.col("seller_id").is_in(bundle.sellers["seller_id"])
              ).height
           ),
        }

        failed = {k: v for k, v in fk_violations.items() if v > 0}
        if failed:
           raise ValueError(f"Foreign-key integrity violations detected: {failed}")

  def load_processed(self) -> ProcessedDatasetBundle:

     required_files = [
        "train_interactions.parquet",
        "val_interactions.parquet",
        "test_interactions.parquet",
        "train_impressions.parquet",
        "val_impressions.parquet",
        "test_impressions.parquet",
        "train_user_item_pairs.parquet",
        "items_enriched.parquet",
        "users_enriched.parquet",
        "position_propensities.parquet",
        "split_metadata.json",
     ]

     missing = [f for f in required_files if not (self.processed_dir / f).exists()]

     if missing:
        raise FileNotFoundError(
           f"Missing processed dataset files in {self.processed_dir}: {missing}."
           " Run dataset_builder.py first."
        )

     with (self.processed_dir / "split_metadata.json").open(
        "r", encoding="utf-8"
     ) as f:
        metadata = json.load(f)

     return ProcessedDatasetBundle(
        train_interactions=pl.read_parquet(
           self.processed_dir / "train_interactions.parquet"
        ),
        val_interactions=pl.read_parquet(
           self.processed_dir / "val_interactions.parquet"
        ),
        test_interactions=pl.read_parquet(
           self.processed_dir / "test_interactions.parquet"
        ),
        train_impressions=pl.read_parquet(
           self.processed_dir / "train_impressions.parquet"
        ),
        val_impressions=pl.read_parquet(
           self.processed_dir / "val_impressions.parquet"
        ),
        test_impressions=pl.read_parquet(
           self.processed_dir / "test_impressions.parquet"
        ),
        train_user_item_pairs=pl.read_parquet(
           self.processed_dir / "train_user_item_pairs.parquet"
        ),
        items_enriched=pl.read_parquet(
           self.processed_dir / "items_enriched.parquet"
        ),
        users_enriched=pl.read_parquet(
           self.processed_dir / "users_enriched.parquet"
        ),
        position_propensities=pl.read_parquet(
           self.processed_dir / "position_propensities.parquet"
        ),
        metadata=metadata,
     )

  def load_split(
        self,
        split: str = "train",
        table: str = "interactions"
  ) -> pl.DataFrame:

     if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

     if split not in {"interactions", "impressions"}:
        raise ValueError(f"Invalid table '{table}'. Expected 'interactions' or 'impressions'.")

     path = self.processed_dir / f"{split}_{table}.parquet"
     if not path.exists():
        raise FileNotFoundError(f"Processed split file not found: {path}")
     return pl.read_parquet(path)


def load_raw_datasets(
    raw_dir: str | Path = "data/raw",
    validate_integrity: bool = True,
) -> RawDatasetBundle:

  return DataLoader(raw_dir=raw_dir).load_raw(
      validate_integrity=validate_integrity
  )


def load_processed_datasets(
    processed_dir: str | Path = "data/processed",
) -> ProcessedDatasetBundle:

  return DataLoader(processed_dir=processed_dir).load_processed()


        
       