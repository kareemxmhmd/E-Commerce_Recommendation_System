import unittest
from fastapi.testclient import TestClient
from api.main import app
from src.config import PathsConfig

_PATHS = PathsConfig()
_HAS_API_ARTIFACTS = (
    (_PATHS.processed_dir / "split_metadata.json").exists()
    and (_PATHS.models_dir / "serving_bundle.npz").exists()
    and (_PATHS.reports_dir / "evaluation_metrics.json").exists()
)


@unittest.skipUnless(
    _HAS_API_ARTIFACTS,
    "Requires processed data, trained serving_bundle.npz, and evaluation_metrics.json.",
)
class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["catalog_size"], 4000)
        self.assertEqual(data["in_stock_items"], 3676)

    def test_recommend_post_and_get(self) -> None:
        resp = self.client.post(
            "/recommend",
            json={
                "user_id": 42,
                "top_k": 5,
                "exclude_seen": True,
                "apply_diversity_rerank": True,
                "debiased_position": 1,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["user_id"], 42)
        self.assertEqual(body["count"], 5)
        self.assertEqual(len(body["recommendations"]), 5)
        self.assertIn("X-Process-Time-Ms", resp.headers)

        get_resp = self.client.get("/recommend/42?top_k=5")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(len(get_resp.json()["recommendations"]), 5)

    def test_similar_items_and_rank_endpoints(self) -> None:
        sim_resp = self.client.get("/similar-items/10?top_k=5")
        self.assertEqual(sim_resp.status_code, 200)
        self.assertEqual(len(sim_resp.json()), 5)

        rank_resp = self.client.post(
            "/rank",
            json={
                "user_id": 15,
                "candidate_item_ids": [1, 2, 3, 10, 20],
                "debiased_position": 1,
            },
        )
        self.assertEqual(rank_resp.status_code, 200)
        self.assertGreaterEqual(len(rank_resp.json()), 3)

    def test_user_profile_and_metrics_endpoints(self) -> None:
        prof_resp = self.client.get("/users/42/profile")
        self.assertEqual(prof_resp.status_code, 200)
        self.assertEqual(prof_resp.json()["user_id"], 42)

        metrics_resp = self.client.get("/metrics")
        self.assertEqual(metrics_resp.status_code, 200)
        self.assertIn("models", metrics_resp.json())


if __name__ == "__main__":
    unittest.main()
