import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from src.config import PathsConfig
from src.data.loader import load_processed_datasets

logger = logging.getLogger(__name__)


def classify_psi(psi: float) -> str:
  if psi < 0.10:
    return "no_drift"
  if psi < 0.25:
    return "moderate_drift"
  return "significant_drift"


def compute_numeric_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> dict[str, Any]:
  ref = np.asarray(reference, dtype=np.float64)
  cur = np.asarray(current, dtype=np.float64)
  ref = ref[np.isfinite(ref)]
  cur = cur[np.isfinite(cur)]

  if len(ref) == 0 or len(cur) == 0:
    return {
        "psi": 0.0,
        "status": "no_drift",
        "ks_statistic": 0.0,
        "ks_p_value": 1.0,
        "reference_mean": 0.0,
        "current_mean": 0.0,
    }

  quantiles = np.linspace(0.0, 1.0, n_bins + 1)
  edges = np.unique(np.quantile(ref, quantiles))
  if len(edges) < 2:
    psi = 0.0
  else:
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = np.clip(ref_counts / max(len(ref), 1), eps, None)
    cur_pct = np.clip(cur_counts / max(len(cur), 1), eps, None)
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

  ks_res = stats.ks_2samp(ref, cur)
  return {
      "psi": round(psi, 6),
      "status": classify_psi(psi),
      "ks_statistic": round(float(ks_res.statistic), 6),
      "ks_p_value": round(float(ks_res.pvalue), 6),
      "reference_mean": round(float(np.mean(ref)), 6),
      "current_mean": round(float(np.mean(cur)), 6),
      "reference_std": round(float(np.std(ref)), 6),
      "current_std": round(float(np.std(cur)), 6),
  }


def compute_categorical_psi(
    reference: pl.Series,
    current: pl.Series,
    eps: float = 1e-6,
) -> dict[str, Any]:
  ref_vc = reference.drop_nulls().cast(pl.Utf8).value_counts()
  cur_vc = current.drop_nulls().cast(pl.Utf8).value_counts()

  ref_col = ref_vc.columns[0]
  cur_col = cur_vc.columns[0]
  ref_map = {
      str(r[ref_col]): int(r["count"]) for r in ref_vc.iter_rows(named=True)
  }
  cur_map = {
      str(r[cur_col]): int(r["count"]) for r in cur_vc.iter_rows(named=True)
  }

  ref_total = max(sum(ref_map.values()), 1)
  cur_total = max(sum(cur_map.values()), 1)
  categories = sorted(set(ref_map.keys()) | set(cur_map.keys()))

  if not categories:
    return {"psi": 0.0, "status": "no_drift", "num_categories": 0}

  ref_pct = np.array(
      [max(ref_map.get(c, 0) / ref_total, eps) for c in categories],
      dtype=np.float64,
  )
  cur_pct = np.array(
      [max(cur_map.get(c, 0) / cur_total, eps) for c in categories],
      dtype=np.float64,
  )
  ref_pct = ref_pct / ref_pct.sum()
  cur_pct = cur_pct / cur_pct.sum()

  psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
  return {
      "psi": round(psi, 6),
      "status": classify_psi(psi),
      "num_categories": len(categories),
  }


def run_drift_report(
    paths: PathsConfig | None = None,
    target_split: str = "test",
    output_path: Path | None = None,
) -> dict[str, Any]:
  if target_split not in {"val", "test"}:
    raise ValueError(
        f"Invalid target_split '{target_split}'. Expected 'val' or 'test'."
    )

  paths = paths or PathsConfig()
  bundle = load_processed_datasets(paths.processed_dir)

  ref_inter = bundle.train_interactions
  cur_inter = (
      bundle.val_interactions
      if target_split == "val"
      else bundle.test_interactions
  )
  ref_impr = bundle.train_impressions
  cur_impr = (
      bundle.val_impressions
      if target_split == "val"
      else bundle.test_impressions
  )

  item_meta = bundle.items_enriched.select(
      ["item_id", "price", "log_price", "root_category_id"]
  )
  user_meta = bundle.users_enriched.select(
      ["user_id", "city", "device", "language"]
  )

  ref_impr_enriched = ref_impr.join(item_meta, on="item_id", how="left").join(
      user_meta, on="user_id", how="left"
  )
  cur_impr_enriched = cur_impr.join(item_meta, on="item_id", how="left").join(
      user_meta, on="user_id", how="left"
  )

  numeric_drift: dict[str, dict[str, Any]] = {
      "impression_position": compute_numeric_psi(
          ref_impr_enriched["position"].to_numpy(),
          cur_impr_enriched["position"].to_numpy(),
      ),
      "impressed_item_price": compute_numeric_psi(
          ref_impr_enriched["price"].drop_nulls().to_numpy(),
          cur_impr_enriched["price"].drop_nulls().to_numpy(),
      ),
      "impressed_item_log_price": compute_numeric_psi(
          ref_impr_enriched["log_price"].drop_nulls().to_numpy(),
          cur_impr_enriched["log_price"].drop_nulls().to_numpy(),
      ),
  }

  categorical_drift: dict[str, dict[str, Any]] = {
      "interaction_event_type": compute_categorical_psi(
          ref_inter["event_type"], cur_inter["event_type"]
      ),
      "impression_user_city": compute_categorical_psi(
          ref_impr_enriched["city"], cur_impr_enriched["city"]
      ),
      "impression_user_device": compute_categorical_psi(
          ref_impr_enriched["device"], cur_impr_enriched["device"]
      ),
      "impression_root_category": compute_categorical_psi(
          ref_impr_enriched["root_category_id"],
          cur_impr_enriched["root_category_id"],
      ),
  }

  ref_ctr = float(ref_impr["clicked"].mean() or 0.0)
  cur_ctr = float(cur_impr["clicked"].mean() or 0.0)

  train_users = set(ref_inter["user_id"].unique().to_list())
  train_items = set(ref_inter["item_id"].unique().to_list())
  cur_users = set(cur_inter["user_id"].unique().to_list())
  cur_items = set(cur_inter["item_id"].unique().to_list())

  unseen_user_rate = len(cur_users - train_users) / max(len(cur_users), 1)
  unseen_item_rate = len(cur_items - train_items) / max(len(cur_items), 1)

  all_statuses = [m["status"] for m in numeric_drift.values()] + [
      m["status"] for m in categorical_drift.values()
  ]
  if "significant_drift" in all_statuses:
    overall_status = "alert"
  elif "moderate_drift" in all_statuses:
    overall_status = "warning"
  else:
    overall_status = "healthy"

  report: dict[str, Any] = {
      "generated_at_utc": datetime.now(timezone.utc).isoformat(),
      "reference_split": "train",
      "target_split": target_split,
      "overall_status": overall_status,
      "sample_counts": {
          "reference_interactions": int(ref_inter.height),
          "target_interactions": int(cur_inter.height),
          "reference_impressions": int(ref_impr.height),
          "target_impressions": int(cur_impr.height),
      },
      "target_shift": {
          "reference_ctr": round(ref_ctr, 6),
          "target_ctr": round(cur_ctr, 6),
          "ctr_delta": round(cur_ctr - ref_ctr, 6),
      },
      "cold_start_exposure": {
          "unseen_user_rate": round(unseen_user_rate, 6),
          "unseen_item_rate": round(unseen_item_rate, 6),
      },
      "numeric_drift": numeric_drift,
      "categorical_drift": categorical_drift,
  }

  dest = output_path or (paths.reports_dir / "drift_report.json")
  dest.parent.mkdir(parents=True, exist_ok=True)
  with dest.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
    f.write("\n")

  logger.info(
      "Saved drift report (%s) to %s", overall_status, dest
  )
  return report


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Compute PSI & KS-test data drift report between train and val/test splits."
  )
  parser.add_argument(
      "--split",
      choices=["val", "test"],
      default="test",
      help="Target split to compare against train reference.",
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=None,
      help="Optional path for drift_report.json.",
  )
  return parser.parse_args()


def main() -> None:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s | %(levelname)s | %(message)s",
  )
  args = parse_args()
  report = run_drift_report(target_split=args.split, output_path=args.output)
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
