import unittest
import polars as pl

from src.config import PathsConfig, RerankConfig
from src.inference.predictor import ProductionRecommender
from src.models.reranker import CatalogReranker

_PATHS = PathsConfig()
_HAS_SERVING_ARTIFACTS = (
    (_PATHS.processed_dir / "split_metadata.json").exists()
    and (_PATHS.models_dir / "serving_bundle.npz").exists()
)


class TestModelAndInference(unittest.TestCase):

  def test_catalog_reranker_enforces_stock_and_diversity_caps(self) -> None:
    items_df = pl.DataFrame(
        [
            {
                "item_id": 1,
                "title": "Item 1",
                "seller_id": 10,
                "seller_name": "S10",
                "seller_rating": 4.8,
                "category_id": 100,
                "category_name": "C100",
                "root_category_id": 1,
                "root_category_name": "R1",
                "brand": "B1",
                "stock": 5,
                "price": 20.0,
            },
            {
                "item_id": 2,
                "title": "Item 2",
                "seller_id": 10,
                "seller_name": "S10",
                "seller_rating": 4.8,
                "category_id": 100,
                "category_name": "C100",
                "root_category_id": 1,
                "root_category_name": "R1",
                "brand": "B1",
                "stock": 5,
                "price": 25.0,
            },
            {
                "item_id": 3,
                "title": "Item 3",
                "seller_id": 10,
                "seller_name": "S10",
                "seller_rating": 4.8,
                "category_id": 100,
                "category_name": "C100",
                "root_category_id": 1,
                "root_category_name": "R1",
                "brand": "B1",
                "stock": 5,
                "price": 30.0,
            },
            {
                "item_id": 4,
                "title": "Item 4",
                "seller_id": 20,
                "seller_name": "S20",
                "seller_rating": 4.5,
                "category_id": 200,
                "category_name": "C200",
                "root_category_id": 2,
                "root_category_name": "R2",
                "brand": "B2",
                "stock": 0,
                "price": 15.0,
            },
            {
                "item_id": 5,
                "title": "Item 5",
                "seller_id": 30,
                "seller_name": "S30",
                "seller_rating": 4.9,
                "category_id": 300,
                "category_name": "C300",
                "root_category_id": 3,
                "root_category_name": "R3",
                "brand": "B3",
                "stock": 12,
                "price": 40.0,
            },
        ]
    )
    reranker = CatalogReranker(
        items_enriched=items_df,
        config=RerankConfig(
            max_items_per_seller=2,
            max_items_per_leaf_category=2,
            filter_out_of_stock=True,
        ),
    )
    ranked = reranker.rerank(
        candidate_ids=[1, 2, 3, 4, 5],
        candidate_scores=[0.99, 0.95, 0.90, 0.85, 0.80],
        top_k=3,
    )
    ranked_ids = [r.item_id for r in ranked]
    self.assertEqual(ranked_ids, [1, 2, 5])

  @unittest.skipUnless(
      _HAS_SERVING_ARTIFACTS,
      "Requires processed datasets and trained serving_bundle.npz.",
  )
  def test_production_recommender_serves_warm_and_cold_users(self) -> None:
    svc = ProductionRecommender(paths=PathsConfig()).load()

    warm_recs = svc.recommend(user_id=1, top_k=10, exclude_seen=True)
    self.assertEqual(len(warm_recs), 10)
    for r in warm_recs:
      self.assertGreater(
          r.stock, 0, "Out-of-stock items must never be recommended"
      )

    cold_recs = svc.recommend(
        user_id=999999, top_k=10, city="Cairo", language="ar"
    )
    self.assertEqual(len(cold_recs), 10)
    for r in cold_recs:
      self.assertGreater(r.stock, 0)


if __name__ == "__main__":
  unittest.main()
