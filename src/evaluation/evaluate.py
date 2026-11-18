import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import polars as pl

from src.config import MLflowConfig, PathsConfig, RerankConfig
from src.data.loader import load_processed_datasets
from src.evaluation.metrics import evaluate_ranking_predictions
from src.inference.predictor import ProductionRecommender
from src.models.baseline import PopularityBaseline
from src.tracking import log_numeric_metrics, setup_mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def _extract_ground_truth(
    interactions_df: pl.DataFrame,
    positive_events: tuple[str, ...] = (
        "purchase",
        "add_to_cart",
        "wishlist",
        "rating",
    ),
) -> Dict[int, Set[int]]:

  filtered = interactions_df.filter(
      pl.col("event_type").is_in(list(positive_events))
  )
  grouped = filtered.group_by("user_id").agg(pl.col("item_id").alias("items"))
  gt: Dict[int, Set[int]] = {}
  for row in grouped.iter_rows(named=True):
    uid = int(row["user_id"])
    items_set = {int(x) for x in row["items"]}
    if items_set:
      gt[uid] = items_set
  return gt


def _load_leaf_preferences(
    ground_truth_csv: Path,
) -> Dict[int, Dict[int, float]]:

  if not ground_truth_csv.exists():
    return {}
  df = pl.read_csv(ground_truth_csv)
  leaf_cols = [c for c in df.columns if c.startswith("leaf_")]
  prefs: Dict[int, Dict[int, float]] = {}
  for row in df.iter_rows(named=True):
    uid = int(row["user_id"])
    prefs[uid] = {
        int(col.replace("leaf_", "")): float(row[col]) for col in leaf_cols
    }
  return prefs


def run_offline_evaluation(
    paths: PathsConfig | None = None,
    max_eval_users: int = 1000,
    mlflow_config: MLflowConfig | None = None,
    run_name: str | None = None,
) -> Dict[str, Any]:

  paths = paths or PathsConfig()
  paths.reports_dir.mkdir(parents=True, exist_ok=True)

  logger.info("Loading processed datasets for offline evaluation...")
  bundle = load_processed_datasets(paths.processed_dir)
  test_gt = _extract_ground_truth(bundle.test_interactions)

  item_interaction_counts = {
      int(r["item_id"]): int(r["train_interaction_count"])
      for r in bundle.items_enriched.select(
          ["item_id", "train_interaction_count"]
      ).iter_rows(named=True)
  }
  total_train_interactions = int(bundle.train_interactions.height)
  item_to_leaf_cat = {
      int(r["item_id"]): int(r["category_id"])
      for r in bundle.items_enriched.select(
          ["item_id", "category_id"]
      ).iter_rows(named=True)
  }
  total_in_stock = int(bundle.items_enriched["in_stock"].sum())

  recommender = ProductionRecommender(
      paths=paths, rerank_config=RerankConfig()
  ).load()
  baseline: PopularityBaseline = recommender.baseline
  leaf_prefs = _load_leaf_preferences(paths.ground_truth_path)

  rng = np.random.default_rng(42)
  all_test_users = sorted(test_gt.keys())
  if len(all_test_users) > max_eval_users:
    eval_user_ids = sorted(
        int(x)
        for x in rng.choice(all_test_users, size=max_eval_users, replace=False)
    )
  else:
    eval_user_ids = all_test_users

  eval_gt = {uid: test_gt[uid] for uid in eval_user_ids}
  warm_user_ids = [
      uid
      for uid in eval_user_ids
      if not recommender.user_profiles.get(uid, {}).get(
          "is_cold_start_user", True
      )
  ]
  cold_user_ids = [
      uid
      for uid in eval_user_ids
      if recommender.user_profiles.get(uid, {}).get("is_cold_start_user", True)
  ]

  logger.info(
      "Evaluating %d test users (%d warm, %d cold-start)...",
      len(eval_user_ids),
      len(warm_user_ids),
      len(cold_user_ids),
  )

  pop_preds: Dict[int, List[int]] = {}
  twotower_preds: Dict[int, List[int]] = {}
  undebiased_preds: Dict[int, List[int]] = {}
  debiased_preds: Dict[int, List[int]] = {}

  for uid in eval_user_ids:
    pop_preds[uid] = baseline.recommend(user_id=uid, top_k=20, exclude_seen=True)
    twotower_preds[uid] = recommender.retrieve_two_tower(
        user_id=uid, top_k=20, exclude_seen=True
    )
    noisy_slates = recommender.recommend(
        user_id=uid,
        top_k=20,
        serving_position=8,
        exclude_seen=True,
        apply_diversity_rerank=False,
    )
    undebiased_preds[uid] = [c.item_id for c in noisy_slates]

    debiased_slates = recommender.recommend(
        user_id=uid,
        top_k=20,
        serving_position=1,
        exclude_seen=True,
        apply_diversity_rerank=True,
    )
    debiased_preds[uid] = [c.item_id for c in debiased_slates]

  models_preds = {
      "popularity_baseline": pop_preds,
      "keras_two_tower_retrieval": twotower_preds,
      "keras_neural_ranker_undebiased_pos8": undebiased_preds,
      "keras_debiased_neural_ranker": debiased_preds,
  }

  results: Dict[str, Any] = {
      "num_evaluated_users": len(eval_user_ids),
      "num_warm_users": len(warm_user_ids),
      "num_cold_users": len(cold_user_ids),
      "total_in_stock_items": total_in_stock,
      "models": {},
  }

  warm_gt = {u: eval_gt[u] for u in warm_user_ids}
  cold_gt = {u: eval_gt[u] for u in cold_user_ids}

  for model_name, preds in models_preds.items():
    metrics = evaluate_ranking_predictions(
        recommendations_by_user=preds,
        ground_truth_by_user=eval_gt,
        total_in_stock_items=total_in_stock,
        item_interaction_counts=item_interaction_counts,
        total_interactions=total_train_interactions,
        k_values=(5, 10, 20),
        item_to_leaf_category=item_to_leaf_cat,
        user_leaf_ground_truth=leaf_prefs,
    )

    warm_m = (
        evaluate_ranking_predictions(
            recommendations_by_user={u: preds[u] for u in warm_user_ids},
            ground_truth_by_user=warm_gt,
            total_in_stock_items=total_in_stock,
            item_interaction_counts=item_interaction_counts,
            total_interactions=total_train_interactions,
            k_values=(10,),
        )
        if warm_gt
        else {}
    )
    cold_m = (
        evaluate_ranking_predictions(
            recommendations_by_user={u: preds[u] for u in cold_user_ids},
            ground_truth_by_user=cold_gt,
            total_in_stock_items=total_in_stock,
            item_interaction_counts=item_interaction_counts,
            total_interactions=total_train_interactions,
            k_values=(10,),
        )
        if cold_gt
        else {}
    )

    metrics["warm_recall@10"] = warm_m.get("recall@10", 0.0)
    metrics["warm_ndcg@10"] = warm_m.get("ndcg@10", 0.0)
    metrics["cold_recall@10"] = cold_m.get("recall@10", 0.0)
    metrics["cold_ndcg@10"] = cold_m.get("ndcg@10", 0.0)
    results["models"][model_name] = metrics

  json_path = paths.reports_dir / "evaluation_metrics.json"
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
    f.write("\n")

  md_path = paths.reports_dir / "metrics_report.md"
  _write_markdown_report(results, md_path)
  logger.info("Saved offline evaluation reports to %s and %s", json_path, md_path)

  mlflow_mod = setup_mlflow(mlflow_config or MLflowConfig())
  if mlflow_mod is not None:
    trained_run_id = recommender.metadata.get("mlflow_run_id")
    eval_run_name = run_name or f"offline-eval-u{len(eval_user_ids)}"
    with mlflow_mod.start_run(run_name=eval_run_name):
      tags = {
          "stage": "evaluation",
          "split": "test",
      }
      if trained_run_id:
        tags["trained_run_id"] = str(trained_run_id)
      mlflow_mod.set_tags(tags)
      mlflow_mod.log_params(
          {
              "max_eval_users": max_eval_users,
              "num_evaluated_users": results["num_evaluated_users"],
              "num_warm_users": results["num_warm_users"],
              "num_cold_users": results["num_cold_users"],
              "total_in_stock_items": results["total_in_stock_items"],
          }
      )

      champion_metrics = results["models"].get(
          "keras_debiased_neural_ranker", {}
      )
      log_numeric_metrics(
          mlflow_mod, champion_metrics, prefix="debiased_"
      )

      for model_name, metrics in results["models"].items():
        with mlflow_mod.start_run(run_name=model_name, nested=True):
          mlflow_mod.set_tags(
              {
                  "stage": "evaluation_variant",
                  "model_variant": model_name,
              }
          )
          log_numeric_metrics(mlflow_mod, metrics)

      mlflow_mod.log_artifact(str(json_path), artifact_path="reports")
      mlflow_mod.log_artifact(str(md_path), artifact_path="reports")
      logger.info("Logged offline evaluation metrics and reports to MLflow.")

  return results


def _write_markdown_report(results: Dict[str, Any], output_path: Path) -> None:

  models = results["models"]
  lines = [
      "# Offline Ranking & Position-Debiasing Evaluation Report",
      "",
      f"- **Evaluated Test Users**: `{results['num_evaluated_users']}` "
      f"(`{results['num_warm_users']}` warm users, `{results['num_cold_users']}` strict cold-start users)",
      f"- **Active In-Stock Catalog Size**: `{results['total_in_stock_items']}` items (`324` out-of-stock items excluded)",
      "",
      "## Primary Ranking & Marketplace Health Scorecard (`@10`)",
      "",
      "| Model Variant | Recall@10 | NDCG@10 | MAP@10 | HitRate@10 | MRR@10 | Coverage@10 | Novelty@10 | Category KL@10 (↓) |",
      "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ]
  display_names = {
      "popularity_baseline": "Demographic Popularity Baseline",
      "keras_two_tower_retrieval": "Keras Two-Tower Retrieval",
      "keras_neural_ranker_undebiased_pos8": "Keras Neural Ranker (Noisy Pos=8)",
      "keras_debiased_neural_ranker": (
          "Keras Debiased Neural Ranker (Pos=1 + Diversity)"
      ),
  }
  for key, label in display_names.items():
    if key not in models:
      continue
    m = models[key]
    lines.append(
        f"| **{label}** | {m.get('recall@10', 0.0):.4f} |"
        f" {m.get('ndcg@10', 0.0):.4f} | {m.get('map@10', 0.0):.4f} |"
        f" {m.get('hit_rate@10', 0.0):.4f} | {m.get('mrr@10', 0.0):.4f} |"
        f" {m.get('catalog_coverage@10', 0.0):.4f} |"
        f" {m.get('novelty@10', 0.0):.3f} |"
        f" {m.get('category_kl_div@10', 0.0):.4f} |"
    )

  lines.extend(
      [
          "",
          "## Warm vs. Strict Cold-Start Breakdown (`@10`)",
          "",
          "| Model Variant | Warm Recall@10 | Warm NDCG@10 | Cold Recall@10 | Cold NDCG@10 |",
          "| :--- | :---: | :---: | :---: | :---: |",
      ]
  )
  for key, label in display_names.items():
    if key not in models:
      continue
    m = models[key]
    lines.append(
        f"| **{label}** | {m.get('warm_recall@10', 0.0):.4f} |"
        f" {m.get('warm_ndcg@10', 0.0):.4f} |"
        f" {m.get('cold_recall@10', 0.0):.4f} |"
        f" {m.get('cold_ndcg@10', 0.0):.4f} |"
    )

  output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Run offline ranking evaluation and log results to MLflow."
  )
  parser.add_argument("--max-eval-users", type=int, default=1000)
  parser.add_argument(
      "--no-mlflow",
      action="store_true",
      help="Disable MLflow experiment tracking for this evaluation run.",
  )
  parser.add_argument("--tracking-uri", type=str, default=None)
  parser.add_argument("--experiment-name", type=str, default=None)
  parser.add_argument("--run-name", type=str, default=None)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  default_mlflow = MLflowConfig()
  mlflow_cfg = MLflowConfig(
      enabled=not args.no_mlflow,
      store_dir=default_mlflow.store_dir,
      tracking_uri=args.tracking_uri or default_mlflow.tracking_uri,
      experiment_name=args.experiment_name or default_mlflow.experiment_name,
      artifact_location=default_mlflow.artifact_location,
  )
  run_offline_evaluation(
      max_eval_users=args.max_eval_users,
      mlflow_config=mlflow_cfg,
      run_name=args.run_name,
  )


if __name__ == "__main__":
  main()
