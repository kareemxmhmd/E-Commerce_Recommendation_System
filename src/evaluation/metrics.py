import math
from typing import Iterable

import numpy as np


def recall_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:

  if not relevant or k <= 0:
    return 0.0
  top_k = recommended[:k]
  hits = sum(1 for item in top_k if item in relevant)
  return float(hits) / float(len(relevant))


def hit_rate_at_k(
    recommended: list[int], relevant: set[int], k: int = 10
) -> float:

  if not relevant or k <= 0:
    return 0.0
  for item in recommended[:k]:
    if item in relevant:
      return 1.0
  return 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:

  if not relevant or k <= 0:
    return 0.0
  dcg = 0.0
  for rank_idx, item in enumerate(recommended[:k]):
    if item in relevant:
      dcg += 1.0 / math.log2(rank_idx + 2.0)

  ideal_hits = min(len(relevant), k)
  idcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal_hits))
  return float(dcg / idcg) if idcg > 0.0 else 0.0


def average_precision_at_k(
    recommended: list[int], relevant: set[int], k: int = 10
) -> float:

  if not relevant or k <= 0:
    return 0.0
  hits = 0.0
  sum_precisions = 0.0
  for rank_idx, item in enumerate(recommended[:k]):
    if item in relevant:
      hits += 1.0
      sum_precisions += hits / float(rank_idx + 1)
  return float(sum_precisions / min(len(relevant), k))


def mrr_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:

  if not relevant or k <= 0:
    return 0.0
  for rank_idx, item in enumerate(recommended[:k]):
    if item in relevant:
      return 1.0 / float(rank_idx + 1)
  return 0.0


def catalog_coverage(
    all_recommendations: Iterable[list[int]],
    total_in_stock_items: int,
    k: int = 10,
) -> float:

  if total_in_stock_items <= 0:
    return 0.0
  unique_surfaced: set[int] = set()
  for recs in all_recommendations:
    unique_surfaced.update(recs[:k])
  return float(len(unique_surfaced)) / float(total_in_stock_items)


def mean_self_information_novelty(
    all_recommendations: Iterable[list[int]],
    item_interaction_counts: dict[int, int],
    total_interactions: int,
    k: int = 10,
) -> float:

  if total_interactions <= 0:
    return 0.0
  scores: list[float] = []
  denom = float(total_interactions)
  for recs in all_recommendations:
    for item_id in recs[:k]:
      cnt = max(int(item_interaction_counts.get(int(item_id), 1)), 1)
      prob = cnt / denom
      scores.append(-math.log2(prob))
  return float(np.mean(scores)) if scores else 0.0


def category_kl_divergence(
    recommended: list[int],
    item_to_leaf_category: dict[int, int],
    user_true_leaf_dist: dict[int, float],
    k: int = 10,
    smoothing: float = 1e-4,
) -> float:

  if not recommended or not user_true_leaf_dist:
    return 0.0
  top_k = recommended[:k]
  emp_counts: dict[int, float] = {cat: 0.0 for cat in user_true_leaf_dist}
  valid = 0
  for item_id in top_k:
    cat = item_to_leaf_category.get(int(item_id))
    if cat in emp_counts:
      emp_counts[cat] += 1.0
      valid += 1
  if valid == 0:
    return 0.0

  num_cats = len(user_true_leaf_dist)
  kl = 0.0
  for cat, p_true in user_true_leaf_dist.items():
    p = max(float(p_true), 1e-12)
    q = (emp_counts.get(cat, 0.0) + smoothing) / (
        float(valid) + smoothing * num_cats
    )
    kl += p * math.log(p / q)
  return max(0.0, float(kl))


def evaluate_ranking_predictions(
    recommendations_by_user: dict[int, list[int]],
    ground_truth_by_user: dict[int, set[int]],
    total_in_stock_items: int,
    item_interaction_counts: dict[int, int],
    total_interactions: int,
    k_values: tuple[int, ...] = (5, 10, 20),
    item_to_leaf_category: dict[int, int] | None = None,
    user_leaf_ground_truth: dict[int, dict[int, float]] | None = None,
) -> dict[str, float]:

  eval_users = [
      u
      for u, rel in ground_truth_by_user.items()
      if rel and u in recommendations_by_user
  ]
  if not eval_users:
    return {}

  results: dict[str, float] = {"evaluated_users": float(len(eval_users))}
  for k in k_values:
    recalls = [
        recall_at_k(recommendations_by_user[u], ground_truth_by_user[u], k=k)
        for u in eval_users
    ]
    ndcgs = [
        ndcg_at_k(recommendations_by_user[u], ground_truth_by_user[u], k=k)
        for u in eval_users
    ]
    maps = [
        average_precision_at_k(
            recommendations_by_user[u], ground_truth_by_user[u], k=k
        )
        for u in eval_users
    ]
    hits = [
        hit_rate_at_k(recommendations_by_user[u], ground_truth_by_user[u], k=k)
        for u in eval_users
    ]
    mrrs = [
        mrr_at_k(recommendations_by_user[u], ground_truth_by_user[u], k=k)
        for u in eval_users
    ]

    user_slates = [recommendations_by_user[u] for u in eval_users]
    cov = catalog_coverage(user_slates, total_in_stock_items, k=k)
    nov = mean_self_information_novelty(
        user_slates, item_interaction_counts, total_interactions, k=k
    )

    results[f"recall@{k}"] = round(float(np.mean(recalls)), 5)
    results[f"ndcg@{k}"] = round(float(np.mean(ndcgs)), 5)
    results[f"map@{k}"] = round(float(np.mean(maps)), 5)
    results[f"hit_rate@{k}"] = round(float(np.mean(hits)), 5)
    results[f"mrr@{k}"] = round(float(np.mean(mrrs)), 5)
    results[f"catalog_coverage@{k}"] = round(float(cov), 5)
    results[f"novelty@{k}"] = round(float(nov), 4)

    if (
        k == 10
        and item_to_leaf_category is not None
        and user_leaf_ground_truth is not None
    ):
      kls = [
          category_kl_divergence(
              recommendations_by_user[u],
              item_to_leaf_category,
              user_leaf_ground_truth[u],
              k=10,
          )
          for u in eval_users
          if u in user_leaf_ground_truth
      ]
      if kls:
        results["category_kl_div@10"] = round(float(np.mean(kls)), 5)

  return results
