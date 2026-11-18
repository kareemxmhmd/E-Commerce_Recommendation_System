from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import polars as pl


@dataclass
class PopularityBaseline:

     global_ranking: list[int] = field(default_factory=list)
     item_scores: dict[int, float] = field(default_factory=dict)
     segment_rankings: dict[tuple[str, str], list[int]] = field(default_factory=dict)
     city_rankings: dict[str, list[int]] = field(default_factory=dict)
     in_stock_items: set[int] = field(default_factory=set)
     user_seen_items: dict[int, set[int]] = field(default_factory=dict)
     user_demographics: dict[int, tuple[str, str]] = field(default_factory=dict)

     def fit(
           self,
           train_interactions: pl.DataFrame,
           items_enriched: pl.DataFrame,
           users_enriched: pl.DataFrame | None = None,
     ) -> PopularityBaseline:
        
        in_stock_df = items_enriched.filter(pl.col("in_stock"))
        self.in_stock_items = set(in_stock_df["item_id"].to_list())
        
        max_weight = float(items_enriched["train_weight_sum"].max() or 1.0)
        scored_df = items_enriched.with_columns(
           (
              0.70 * (pl.col("train_weight_sum") / max(max_weight, 1.0))
              + 0.20 * pl.col("train_smoothed_ctr")
              + 0.10 * (pl.col("train_avg_rating") / 5.0)
           ).alias("pop_score")
        ).sort("pop_score", descending=True)

        self.item_scores = {
           int(row["item_id"]): float(row["pop_score"])
           for row in scored_df.select(["item_id", "pop_score"]).iter_rows(named=True)
        }
        
        self.global_ranking = [int(x) for x in scored_df.filter(pl.col("in_stock"))["item_id"].to_list()]
        
        seen_df = train_interactions.group_by("user_id").agg(
           pl.col("item_id").unique().alias("seen_items")
        )
        
        self.user_seen_items = {
          int(row["user_id"]): {int(i) for i in row["seen_items"]}
          for row in seen_df.iter_rows(named=True)
        }
        
        if users_enriched is not None:
          self.user_demographics = {
            int(row["user_id"]): (str(row["city"]), str(row["language"]))
            for row in users_enriched.select(
              ["user_id", "city", "language"]
            ).iter_rows(named=True)
          }
          
          joined = train_interactions.join(
            users_enriched.select(["user_id", "city", "language"]),
            on="user_id",
            how="inner",
          )

          seg_df = (
            joined.group_by(["city", "language", "item_id"])
            .agg(pl.col("confidence_weight").sum().alias("seg_weight"))
            .filter(pl.col("item_id").is_in(list(self.in_stock_items)))
            .sort(["city", "language", "seg_weight"], descending=[False, False, True])
          )
          
          for row in (
            seg_df.group_by(["city", "language"])
            .agg(pl.col("item_id").head(200).alias("top_items"))
            .iter_rows(named=True)
          ):
            key = (str(row["city"]), str(row["language"]))
            self.segment_rankings[key] = [int(i) for i in row["top_items"]]
            
            
          city_df = (
            joined.group_by(["city", "item_id"])
            .agg(pl.col("confidence_weight").sum().alias("city_weight"))
            .filter(pl.col("item_id").is_in(list(self.in_stock_items)))
            .sort(["city", "city_weight"], descending=[False, True])
          )
          
          for row in (
            city_df.group_by("city")
            .agg(pl.col("item_id").head(200).alias("top_items"))
            .iter_rows(named=True)
          ):
            self.city_rankings[str(row["city"])] = [int(i) for i in row["top_items"]]

        return self
        
     def recommend(
           self,
           user_id: int | None = None,
           top_k: int = 10,
           city: str | None = None,
           language: str | None = None,
           exclude_seen: bool = True,
     ) -> list[int]:
        
        seen: set[int] = set()

        if exclude_seen and user_id is not None:
           seen = self.user_seen_items.get(int(user_id), set())

        if (city is None or language is None) and user_id is not None:
           demo = self.user_demographics.get(int(user_id))
           if demo is not None:
              city = city or demo[0]
              language = language or demo[1]

        candidates: list[int] = []
        if city and language and (city, language) in self.segment_rankings:
           candidates.extend(self.segment_rankings[(city, language)])
        elif city and city in self.city_rankings:
           candidates.extend(self.city_rankings[city])

        candidates.extend(self.global_ranking)

        result: list[int] = []
        added: set[int] = set()
        for item_id in candidates:
          if item_id in added or item_id in seen:
             continue
          if item_id not in self.in_stock_items:
             continue
          result.append(item_id)
          added.add(item_id)
          if len(result) >= top_k:
             break
        return result

     def recommend_batch(
           self,
           user_ids: list[int] | np.ndarray,
           top_k: int = 10,
           exclude_seen: bool = True,
     ) -> dict[int, list[int]]:
        return {
           int(uid): self.recommend(user_id=int(uid), top_k=top_k, exclude_seen=exclude_seen)
           for uid in user_ids
        }

     def to_dict(self) -> dict[str, Any]:
        return {
           "global_ranking": self.global_ranking[:500],
           "item_scores": {str(k): v for k, v in self.item_scores.items()},
           "segment_rankings": {
              f"{c}|{l}": items[:100]
              for (c, l), items in self.segment_rankings.items()
           },
           "city_rankings": {c: items[:100] for c, items in self.city_rankings.items()},
        }
