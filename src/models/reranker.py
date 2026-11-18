from dataclasses import dataclass
from typing import Any
import polars as pl

from src.config import RerankConfig


@dataclass
class RankedCandidate:

  item_id: int
  title: str
  category_id: int
  category_name: str
  root_category_id: int
  root_category_name: str
  brand: str
  price: float
  seller_id: int
  seller_name: str
  seller_rating: float
  stock: int
  score: float
  stage: str = "keras_nn_debiased"

  def to_dict(self) -> dict[str, Any]:
    return {
        "item_id": self.item_id,
        "title": self.title,
        "category_id": self.category_id,
        "category_name": self.category_name,
        "root_category_id": self.root_category_id,
        "root_category_name": self.root_category_name,
        "brand": self.brand,
        "price": round(self.price, 2),
        "seller_id": self.seller_id,
        "seller_name": self.seller_name,
        "seller_rating": round(self.seller_rating, 2),
        "stock": self.stock,
        "score": round(self.score, 6),
        "stage": self.stage,
    }


class CatalogReranker:

  def __init__(
      self,
      items_enriched: pl.DataFrame,
      config: RerankConfig | None = None,
  ) -> None:
    self.config = config or RerankConfig()
    self.catalog_map: dict[int, dict[str, Any]] = {}
    for row in items_enriched.iter_rows(named=True):
      self.catalog_map[int(row["item_id"])] = dict(row)

  def rerank(
      self,
      candidate_ids: list[int],
      candidate_scores: list[float],
      seen_items: set[int] | None = None,
      top_k: int | None = None,
      filter_out_of_stock: bool | None = None,
      max_items_per_seller: int | None = None,
      max_items_per_leaf_category: int | None = None,
      stage_label: str = "keras_nn_debiased",
  ) -> list[RankedCandidate]:

    k = top_k if top_k is not None else self.config.top_k
    check_stock = (
        filter_out_of_stock
        if filter_out_of_stock is not None
        else self.config.filter_out_of_stock
    )
    seller_cap = (
        max_items_per_seller
        if max_items_per_seller is not None
        else self.config.max_items_per_seller
    )
    cat_cap = (
        max_items_per_leaf_category
        if max_items_per_leaf_category is not None
        else self.config.max_items_per_leaf_category
    )
    seen = seen_items or set()

    paired = sorted(
        zip(candidate_ids, candidate_scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    selected: list[RankedCandidate] = []
    overflow_pool: list[tuple[dict[str, Any], float]] = []
    seller_counts: dict[int, int] = {}
    cat_counts: dict[int, int] = {}

    for item_id, raw_score in paired:
      iid = int(item_id)
      if self.config.exclude_seen_items and iid in seen:
        continue
      meta = self.catalog_map.get(iid)
      if meta is None:
        continue
      if check_stock and int(meta.get("stock", 0)) <= 0:
        continue

      seller_id = int(meta["seller_id"])
      cat_id = int(meta["category_id"])

      s_count = seller_counts.get(seller_id, 0)
      c_count = cat_counts.get(cat_id, 0)

      if s_count >= seller_cap or c_count >= cat_cap:
        overflow_pool.append((meta, float(raw_score)))
        continue

      adjusted_score = float(raw_score) * (
          1.0 - self.config.diversity_penalty * c_count
      )
      selected.append(
          self._build_candidate(meta, adjusted_score, stage_label)
      )
      seller_counts[seller_id] = s_count + 1
      cat_counts[cat_id] = c_count + 1

      if len(selected) >= k:
        break

    if len(selected) < k and overflow_pool:
      for meta, raw_score in overflow_pool:
        selected.append(
            self._build_candidate(meta, float(raw_score) * 0.85, stage_label)
        )
        if len(selected) >= k:
          break

    return selected

  @staticmethod
  def _build_candidate(
      meta: dict[str, Any], score: float, stage: str
  ) -> RankedCandidate:
    
    return RankedCandidate(
        item_id=int(meta["item_id"]),
        title=str(meta.get("title", f"Item #{meta['item_id']}")),
        category_id=int(meta.get("category_id", 0)),
        category_name=str(meta.get("category_name", "Unknown")),
        root_category_id=int(meta.get("root_category_id", 0)),
        root_category_name=str(meta.get("root_category_name", "Unknown")),
        brand=str(meta.get("brand", "Generic")),
        price=float(meta.get("price", 0.0)),
        seller_id=int(meta.get("seller_id", 0)),
        seller_name=str(meta.get("seller_name", "Marketplace Seller")),
        seller_rating=float(meta.get("seller_rating", 4.5)),
        stock=int(meta.get("stock", 0)),
        score=float(score),
        stage=stage,
    )
