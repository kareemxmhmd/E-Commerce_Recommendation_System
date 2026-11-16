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


class DatasetBuilder:

  def __init__(self, config: DatasetBuilderConfig | None = None) -> None:
    self.config = config or DatasetBuilderConfig()
    self.loader = DataLoader(
        raw_dir=self.config.raw_dir,
        processed_dir=self.config.processed_dir,
    )

  def compute_temporal_cutoffs(
      self, interactions: pl.DataFrame
  ) -> tuple[datetime, datetime]:

    sorted_ts = interactions.sort("timestamp")["timestamp"]
    n_rows = len(sorted_ts)
    train_idx = int(n_rows * self.config.train_quantile)
    val_idx = int(n_rows * self.config.val_quantile)

    train_cutoff = sorted_ts[train_idx]
    val_cutoff = sorted_ts[val_idx]
    return train_cutoff, val_cutoff

  def add_interaction_weights(self, interactions: pl.DataFrame) -> pl.DataFrame:

    weight_expr: pl.Expr = pl.lit(1.0, dtype=pl.Float64)
    for event_name, weight_val in self.config.event_weights.items():
      weight_expr = (
          pl.when(pl.col("event_type") == event_name)
          .then(pl.lit(float(weight_val), dtype=pl.Float64))
          .otherwise(weight_expr)
      )

    rating_bonus_expr = (
        pl.when(
            (pl.col("event_type") == "rating")
            & (pl.col("rating").is_not_null())
            & (pl.col("rating") >= self.config.high_rating_threshold)
        )
        .then(pl.lit(float(self.config.high_rating_bonus), dtype=pl.Float64))
        .otherwise(pl.lit(0.0, dtype=pl.Float64))
    )
    return interactions.with_columns(
        (weight_expr + rating_bonus_expr)
        .alias("confidence_weight")
        .cast(pl.Float64)
    )

  @staticmethod
  def split_by_timestamp(
      df: pl.DataFrame,
      train_cutoff: datetime,
      val_cutoff: datetime,
  ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:

    train_df = df.filter(pl.col("timestamp") <= train_cutoff).sort("timestamp")
    val_df = df.filter(
        (pl.col("timestamp") > train_cutoff)
        & (pl.col("timestamp") <= val_cutoff)
    ).sort("timestamp")
    test_df = df.filter(pl.col("timestamp") > val_cutoff).sort("timestamp")
    return train_df, val_df, test_df

  def compute_position_propensities(
      self, train_impressions: pl.DataFrame
  ) -> pl.DataFrame:

    pos_df = (
        train_impressions.group_by("position")
        .agg(
            [
                pl.len().alias("train_impressions"),
                pl.col("clicked").sum().alias("train_clicks"),
                pl.col("clicked").mean().alias("train_ctr"),
            ]
        )
        .sort("position")
    )

    pos1_ctr = float(pos_df["train_ctr"].first())
    if pos1_ctr <= 0.0:
      pos1_ctr = 1e-6

    pos_df = pos_df.with_columns(
        (pl.col("train_ctr") / pos1_ctr)
        .clip(1e-4, 1.0)
        .round(6)
        .alias("rel_propensity")
    ).with_columns(
        (1.0 / pl.col("rel_propensity"))
        .clip(1.0, self.config.ipw_clip_max)
        .round(6)
        .alias("ipw_weight")
    )
    return pos_df

  @staticmethod
  def attach_ipw_weights(
      impressions_df: pl.DataFrame,
      position_propensities: pl.DataFrame,
  ) -> pl.DataFrame:

    prop_cols = position_propensities.select(
        ["position", "rel_propensity", "ipw_weight"]
    )
    return impressions_df.join(prop_cols, on="position", how="left").with_columns(
        [
            pl.col("rel_propensity").fill_null(1.0),
            pl.col("ipw_weight").fill_null(1.0),
        ]
    )

  @staticmethod
  def build_train_user_item_pairs(
      train_interactions: pl.DataFrame,
  ) -> pl.DataFrame:

    return (
        train_interactions.group_by(["user_id", "item_id"])
        .agg(
            [
                pl.len().alias("interaction_count"),
                pl.col("confidence_weight").sum().alias("total_weight"),
                pl.col("confidence_weight").max().alias("max_weight"),
                (pl.col("event_type") == "purchase")
                .any()
                .cast(pl.Int8)
                .alias("has_purchase"),
                (pl.col("event_type") == "add_to_cart")
                .any()
                .cast(pl.Int8)
                .alias("has_add_to_cart"),
                pl.col("rating").mean().alias("mean_explicit_rating"),
                pl.col("timestamp").max().alias("last_interaction_ts"),
            ]
        )
        .sort(["user_id", "total_weight"], descending=[False, True])
    )

  def build_enriched_items(
      self,
      raw: RawDatasetBundle,
      train_interactions: pl.DataFrame,
      train_impressions: pl.DataFrame,
  ) -> pl.DataFrame:

    parent_cats = raw.categories.select(
        [
            pl.col("category_id").alias("parent_id"),
            pl.col("name").alias("parent_category_name"),
        ]
    )
    cat_hierarchy = (
        raw.categories.rename({"name": "category_name"})
        .join(parent_cats, on="parent_id", how="left")
        .with_columns(
            [
                pl.coalesce(["parent_id", "category_id"]).alias(
                    "root_category_id"
                ),
                pl.coalesce(["parent_category_name", "category_name"]).alias(
                    "root_category_name"
                ),
            ]
        )
        .select(["category_id", "category_name", "root_category_id", "root_category_name"])
    )

    seller_meta = raw.sellers.select(
        [
            "seller_id",
            pl.col("name").alias("seller_name"),
            pl.col("rating").alias("seller_rating"),
            pl.col("joined_at").alias("seller_joined_at"),
        ]
    )

    global_avg_rating = float(
        train_interactions["rating"].drop_nulls().mean() or 4.0
    )
    global_train_ctr = float(train_impressions["clicked"].mean() or 0.10)
    prior = float(self.config.ctr_smoothing_prior)

    item_inter_stats = train_interactions.group_by("item_id").agg(
        [
            pl.len().alias("train_interaction_count"),
            pl.col("confidence_weight").sum().alias("train_weight_sum"),
            (pl.col("event_type") == "purchase")
            .sum()
            .alias("train_purchase_count"),
            (pl.col("event_type") == "add_to_cart")
            .sum()
            .alias("train_cart_count"),
            (pl.col("event_type") == "view").sum().alias("train_view_count"),
            pl.col("rating").mean().alias("train_avg_rating"),
        ]
    )

    item_impr_stats = train_impressions.group_by("item_id").agg(
        [
            pl.len().alias("train_impression_count"),
            pl.col("clicked").sum().alias("train_click_count"),
            pl.col("clicked").mean().alias("train_ctr"),
        ]
    )

    return (
        raw.items.join(cat_hierarchy, on="category_id", how="left")
        .join(seller_meta, on="seller_id", how="left")
        .join(item_inter_stats, on="item_id", how="left")
        .join(item_impr_stats, on="item_id", how="left")
        .with_columns(
            [
                (pl.col("stock") > 0).alias("in_stock"),
                pl.col("price").log1p().round(6).alias("log_price"),
                pl.col("train_interaction_count").fill_null(0),
                pl.col("train_weight_sum").fill_null(0.0),
                pl.col("train_purchase_count").fill_null(0),
                pl.col("train_cart_count").fill_null(0),
                pl.col("train_view_count").fill_null(0),
                pl.col("train_avg_rating")
                .fill_null(global_avg_rating)
                .round(6),
                pl.col("train_impression_count").fill_null(0),
                pl.col("train_click_count").fill_null(0),
                pl.col("train_ctr").fill_null(global_train_ctr).round(6),
            ]
        )
        .with_columns(
            (
                (
                    pl.col("train_click_count")
                    + prior * global_train_ctr
                )
                / (pl.col("train_impression_count") + prior)
            )
            .round(6)
            .alias("train_smoothed_ctr")
        )
        .sort("item_id")
    )

  @staticmethod
  def build_enriched_users(
      raw: RawDatasetBundle,
      train_interactions: pl.DataFrame,
      train_impressions: pl.DataFrame,
  ) -> pl.DataFrame:

    user_inter_stats = train_interactions.group_by("user_id").agg(
        [
            pl.len().alias("train_interaction_count"),
            pl.col("confidence_weight").sum().alias("train_weight_sum"),
            (pl.col("event_type") == "purchase")
            .sum()
            .alias("train_purchase_count"),
            pl.col("item_id").n_unique().alias("train_unique_items"),
            pl.col("session_id").n_unique().alias("train_session_count"),
            pl.col("timestamp").max().alias("last_train_interaction_ts"),
        ]
    )

    user_impr_stats = train_impressions.group_by("user_id").agg(
        [
            pl.len().alias("train_impression_count"),
            pl.col("clicked").sum().alias("train_click_count"),
            pl.col("clicked").mean().alias("train_user_ctr"),
        ]
    )

    return (
        raw.users.join(user_inter_stats, on="user_id", how="left")
        .join(user_impr_stats, on="user_id", how="left")
        .with_columns(
            [
                pl.col("train_interaction_count").fill_null(0),
                pl.col("train_weight_sum").fill_null(0.0),
                pl.col("train_purchase_count").fill_null(0),
                pl.col("train_unique_items").fill_null(0),
                pl.col("train_session_count").fill_null(0),
                pl.col("train_impression_count").fill_null(0),
                pl.col("train_click_count").fill_null(0),
                pl.col("train_user_ctr").fill_null(0.0).round(6),
            ]
        )
        .with_columns(
            (pl.col("train_interaction_count") == 0).alias("is_cold_start_user")
        )
        .sort("user_id")
    )

  @staticmethod
  def build_metadata(
      train_cutoff: datetime,
      val_cutoff: datetime,
      train_inter: pl.DataFrame,
      val_inter: pl.DataFrame,
      test_inter: pl.DataFrame,
      train_impr: pl.DataFrame,
      val_impr: pl.DataFrame,
      test_impr: pl.DataFrame,
      users_enriched: pl.DataFrame,
      items_enriched: pl.DataFrame,
  ) -> dict[str, Any]:

    train_users = set(train_inter["user_id"].unique())
    train_items = set(train_inter["item_id"].unique())

    return {
        "train_cutoff_timestamp": train_cutoff.isoformat(),
        "val_cutoff_timestamp": val_cutoff.isoformat(),
        "interactions": {
            "train_rows": train_inter.height,
            "val_rows": val_inter.height,
            "test_rows": test_inter.height,
            "train_unique_users": len(train_users),
            "val_unique_users": val_inter["user_id"].n_unique(),
            "test_unique_users": test_inter["user_id"].n_unique(),
            "val_unseen_users_vs_train": (
                val_inter.filter(~pl.col("user_id").is_in(list(train_users)))[
                    "user_id"
                ].n_unique()
            ),
            "test_unseen_users_vs_train": (
                test_inter.filter(~pl.col("user_id").is_in(list(train_users)))[
                    "user_id"
                ].n_unique()
            ),
            "val_unseen_items_vs_train": (
                val_inter.filter(~pl.col("item_id").is_in(list(train_items)))[
                    "item_id"
                ].n_unique()
            ),
            "test_unseen_items_vs_train": (
                test_inter.filter(~pl.col("item_id").is_in(list(train_items)))[
                    "item_id"
                ].n_unique()
            ),
        },
        "impressions": {
            "train_rows": train_impr.height,
            "val_rows": val_impr.height,
            "test_rows": test_impr.height,
        },
        "catalog_summary": {
            "total_users": users_enriched.height,
            "cold_start_users_in_train": int(
                users_enriched["is_cold_start_user"].sum()
            ),
            "total_items": items_enriched.height,
            "in_stock_items": int(items_enriched["in_stock"].sum()),
            "out_of_stock_items": int((~items_enriched["in_stock"]).sum()),
        },
    }

  def build(
      self, save: bool = True, validate_integrity: bool = True
  ) -> ProcessedDatasetBundle:

    raw = self.loader.load_raw(validate_integrity=validate_integrity)

    weighted_interactions = self.add_interaction_weights(raw.interactions)
    train_cutoff, val_cutoff = self.compute_temporal_cutoffs(
        weighted_interactions
    )

    train_inter, val_inter, test_inter = self.split_by_timestamp(
        weighted_interactions, train_cutoff, val_cutoff
    )
    train_impr_raw, val_impr_raw, test_impr_raw = self.split_by_timestamp(
        raw.impressions, train_cutoff, val_cutoff
    )

    position_propensities = self.compute_position_propensities(train_impr_raw)
    train_impr = self.attach_ipw_weights(train_impr_raw, position_propensities)
    val_impr = self.attach_ipw_weights(val_impr_raw, position_propensities)
    test_impr = self.attach_ipw_weights(test_impr_raw, position_propensities)

    train_pairs = self.build_train_user_item_pairs(train_inter)
    items_enriched = self.build_enriched_items(raw, train_inter, train_impr)
    users_enriched = self.build_enriched_users(raw, train_inter, train_impr)

    metadata = self.build_metadata(
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        train_inter=train_inter,
        val_inter=val_inter,
        test_inter=test_inter,
        train_impr=train_impr,
        val_impr=val_impr,
        test_impr=test_impr,
        users_enriched=users_enriched,
        items_enriched=items_enriched,
    )

    bundle = ProcessedDatasetBundle(
        train_interactions=train_inter,
        val_interactions=val_inter,
        test_interactions=test_inter,
        train_impressions=train_impr,
        val_impressions=val_impr,
        test_impressions=test_impr,
        train_user_item_pairs=train_pairs,
        items_enriched=items_enriched,
        users_enriched=users_enriched,
        position_propensities=position_propensities,
        metadata=metadata,
    )

    if save:
      self.save_bundle(bundle)

    return bundle

  def save_bundle(self, bundle: ProcessedDatasetBundle) -> None:

    out_dir = self.config.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_tables: dict[str, pl.DataFrame] = {
        "train_interactions.parquet": bundle.train_interactions,
        "val_interactions.parquet": bundle.val_interactions,
        "test_interactions.parquet": bundle.test_interactions,
        "train_impressions.parquet": bundle.train_impressions,
        "val_impressions.parquet": bundle.val_impressions,
        "test_impressions.parquet": bundle.test_impressions,
        "train_user_item_pairs.parquet": bundle.train_user_item_pairs,
        "items_enriched.parquet": bundle.items_enriched,
        "users_enriched.parquet": bundle.users_enriched,
        "position_propensities.parquet": bundle.position_propensities,
    }

    for filename, df in parquet_tables.items():
      df.write_parquet(out_dir / filename)

    metadata_path = out_dir / "split_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
      json.dump(bundle.metadata, f, indent=2)
      f.write("\n")

    logger.info("Saved processed dataset bundle to %s", out_dir)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Build leakage-free temporal datasets for the recommender system."
  )
  parser.add_argument(
      "raw-dir",
      type=Path,
      default=Path("data/raw"),
      help="Directory containing the six raw Parquet files.",
  )
  parser.add_argument(
      "processed-dir",
      type=Path,
      default=Path("data/processed"),
      help="Destination directory for processed Parquet splits.",
  )
  parser.add_argument(
      "train-quantile",
      type=float,
      default=0.70,
      help="Chronological quantile for the end of the training window.",
  )
  parser.add_argument(
      "val-quantile",
      type=float,
      default=0.85,
      help="Chronological quantile for the end of the validation window.",
  )
  return parser.parse_args()


def main() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
  )
  args = parse_args()
  config = DatasetBuilderConfig(
    raw_dir=args.raw_dir,
    processed_dir=args.processed_dir,
    train_quantile=args.train_quantile,
    val_quantile=args.val_quantile,
  )
  builder = DatasetBuilder(config=config)
  bundle = builder.build(save=True, validate_integrity=True)
  print(json.dumps(bundle.metadata, indent=2))


if __name__ == "__main__":
  main()