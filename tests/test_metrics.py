import math
import unittest

from src.evaluation.metrics import (
    average_precision_at_k,
    catalog_coverage,
    category_kl_divergence,
    evaluate_ranking_predictions,
    hit_rate_at_k,
    mean_self_information_novelty,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class TestEvaluationMetrics(unittest.TestCase):

  def test_exact_ranking_metrics(self) -> None:
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 40}

    self.assertAlmostEqual(recall_at_k(recommended, relevant, k=5), 1.0)
    self.assertAlmostEqual(recall_at_k(recommended, relevant, k=2), 0.5)
    self.assertAlmostEqual(hit_rate_at_k(recommended, relevant, k=5), 1.0)
    self.assertAlmostEqual(mrr_at_k(recommended, relevant, k=5), 0.5)

    expected_ap = (1.0 / 2.0 + 2.0 / 4.0) / 2.0
    self.assertAlmostEqual(
        average_precision_at_k(recommended, relevant, k=5), expected_ap
    )

    expected_dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(5))
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    self.assertAlmostEqual(
        ndcg_at_k(recommended, relevant, k=5), expected_dcg / expected_idcg
    )

  def test_beyond_accuracy_metrics(self) -> None:
    recs = [[1, 2, 3], [2, 3, 4]]
    cov = catalog_coverage(recs, total_in_stock_items=10, k=3)
    self.assertAlmostEqual(cov, 0.4)

    nov = mean_self_information_novelty(
        recs,
        item_interaction_counts={1: 100, 2: 50, 3: 10, 4: 1},
        total_interactions=200,
        k=3,
    )
    self.assertGreater(nov, 1.0)

    item_to_cat = {101: 9, 102: 9, 103: 10, 104: 11}
    kl = category_kl_divergence(
        [101, 102, 103, 104],
        item_to_leaf_category=item_to_cat,
        user_true_leaf_dist={9: 0.5, 10: 0.25, 11: 0.25},
        k=4,
    )
    self.assertAlmostEqual(kl, 0.0, places=3)

  def test_aggregate_evaluation(self) -> None:
    preds = {1: [101, 102, 103], 2: [201, 202, 203]}
    gt = {1: {101}, 2: {999}}
    summary = evaluate_ranking_predictions(
        recommendations_by_user=preds,
        ground_truth_by_user=gt,
        total_in_stock_items=100,
        item_interaction_counts={101: 10, 102: 5, 201: 8},
        total_interactions=100,
        k_values=(3,),
    )
    self.assertAlmostEqual(summary["hit_rate@3"], 0.5)
    self.assertAlmostEqual(summary["recall@3"], 0.5)


if __name__ == "__main__":
  unittest.main()
