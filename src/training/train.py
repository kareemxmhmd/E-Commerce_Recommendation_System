import argparse
from contextlib import nullcontext
import json
import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np
import polars as pl

from src.config import KerasNNConfig, MLflowConfig, PathsConfig
from src.data.dataset_builder import DatasetBuilder, DatasetBuilderConfig
from src.data.loader import ProcessedDatasetBundle, load_processed_datasets
from src.models.baseline import PopularityBaseline
from src.models.keras_nn import build_keras_recommender_nn
from src.tracking import (
    create_keras_epoch_logger,
    log_numeric_metrics,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def build_feature_dict_from_impressions(
    impressions_df: pl.DataFrame,
    users_enriched: pl.DataFrame,
    items_enriched: pl.DataFrame,
    max_rows: int | None = None,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:

  df = impressions_df
  if max_rows is not None and df.height > max_rows:
    df = df.sample(n=max_rows, seed=seed)

  user_cols = users_enriched.select(["user_id", "city", "device", "language"])
  item_cols = items_enriched.select(
      [
          "item_id",
          "category_id",
          "root_category_id",
          "seller_id",
          "log_price",
          "train_smoothed_ctr",
          "train_avg_rating",
      ]
  )

  joined = df.join(user_cols, on="user_id", how="inner").join(
      item_cols, on="item_id", how="inner"
  )

  inputs: dict[str, np.ndarray] = {
      "user_id": joined["user_id"].to_numpy().astype(np.int64),
      "city": joined["city"].cast(pl.Utf8).fill_null("").to_numpy().astype(object),
      "device": joined["device"].cast(pl.Utf8).fill_null("").to_numpy().astype(object),
      "language": joined["language"].cast(pl.Utf8).fill_null("").to_numpy().astype(object),
      "item_id": joined["item_id"].to_numpy().astype(np.int64),
      "category_id": joined["category_id"].to_numpy().astype(np.int64),
      "root_category_id": (
          joined["root_category_id"].to_numpy().astype(np.int64)
      ),
      "seller_id": joined["seller_id"].to_numpy().astype(np.int64),
      "log_price": joined["log_price"].to_numpy().astype(np.float32),
      "train_smoothed_ctr": (
          joined["train_smoothed_ctr"].to_numpy().astype(np.float32)
      ),
      "train_avg_rating": (
          joined["train_avg_rating"].to_numpy().astype(np.float32)
      ),
      "position": (
          joined["position"].clip(1, 10).to_numpy().astype(np.int64)
      ),
  }
  targets = joined["clicked"].to_numpy().astype(np.float32)
  sample_weights = (
      joined["ipw_weight"].to_numpy().astype(np.float32)
      if "ipw_weight" in joined.columns
      else np.ones_like(targets, dtype=np.float32)
  )
  return inputs, targets, sample_weights


def export_serving_bundle(
    full_model: keras.Model,
    user_tower: keras.Model,
    item_tower: keras.Model,
    bundle: ProcessedDatasetBundle,
    models_dir: Path,
    config: KerasNNConfig,
    training_history: dict[str, Any],
    mlflow_run_id: str | None = None,
) -> dict[str, Any]:

  models_dir.mkdir(parents=True, exist_ok=True)

  keras_path = models_dir / "keras_recommender.keras"
  full_model.save(keras_path)

  users_df = bundle.users_enriched.sort("user_id")
  items_df = bundle.items_enriched.sort("item_id")

  user_inputs = {
      "user_id": users_df["user_id"].to_numpy().astype(np.int64),
      "city": users_df["city"].cast(pl.Utf8).fill_null("").to_numpy().astype(object),
      "device": users_df["device"].cast(pl.Utf8).fill_null("").to_numpy().astype(object),
      "language": users_df["language"].cast(pl.Utf8).fill_null("").to_numpy().astype(object),
  }
  item_inputs = {
      "item_id": items_df["item_id"].to_numpy().astype(np.int64),
      "category_id": items_df["category_id"].to_numpy().astype(np.int64),
      "root_category_id": (
          items_df["root_category_id"].to_numpy().astype(np.int64)
      ),
      "seller_id": items_df["seller_id"].to_numpy().astype(np.int64),
      "log_price": items_df["log_price"].to_numpy().astype(np.float32),
      "train_smoothed_ctr": (
          items_df["train_smoothed_ctr"].to_numpy().astype(np.float32)
      ),
      "train_avg_rating": (
          items_df["train_avg_rating"].to_numpy().astype(np.float32)
      ),
  }

  user_embeddings = np.asarray(
      user_tower.predict(user_inputs, batch_size=1024, verbose=0),
      dtype=np.float32,
  )
  item_embeddings = np.asarray(
      item_tower.predict(item_inputs, batch_size=1024, verbose=0),
      dtype=np.float32,
  )

  mlp_0_w, mlp_0_b = full_model.get_layer("ranker_mlp_0").get_weights()
  mlp_1_w, mlp_1_b = full_model.get_layer("ranker_mlp_1").get_weights()
  logit_w, logit_b = full_model.get_layer("relevance_logit").get_weights()
  pos_bias_w = full_model.get_layer("position_bias_embedding").get_weights()[0]

  np.savez_compressed(
      models_dir / "serving_bundle.npz",
      user_ids=user_inputs["user_id"],
      user_embeddings=user_embeddings,
      item_ids=item_inputs["item_id"],
      item_embeddings=item_embeddings,
      mlp_0_w=np.asarray(mlp_0_w, dtype=np.float32),
      mlp_0_b=np.asarray(mlp_0_b, dtype=np.float32),
      mlp_1_w=np.asarray(mlp_1_w, dtype=np.float32),
      mlp_1_b=np.asarray(mlp_1_b, dtype=np.float32),
      logit_w=np.asarray(logit_w, dtype=np.float32),
      logit_b=np.asarray(logit_b, dtype=np.float32),
      pos_bias_w=np.asarray(pos_bias_w, dtype=np.float32),
  )

  baseline = PopularityBaseline().fit(
      train_interactions=bundle.train_interactions,
      items_enriched=bundle.items_enriched,
      users_enriched=bundle.users_enriched,
  )
  with (models_dir / "popularity_baseline.json").open(
      "w", encoding="utf-8"
  ) as f:
    json.dump(baseline.to_dict(), f, indent=2)
    f.write("\n")

  meta = {
      "model_name": "keras_recommender_nn",
      "keras_model_path": str(keras_path.name),
      "serving_bundle_path": "serving_bundle.npz",
      "embedding_dim": config.embedding_dim,
      "tower_hidden_units": list(config.tower_hidden_units),
      "ranker_hidden_units": list(config.ranker_hidden_units),
      "num_users": int(users_df.height),
      "num_items": int(items_df.height),
      "default_serving_position": config.default_serving_position,
      "mlflow_run_id": mlflow_run_id,
      "training_history": training_history,
  }
  with (models_dir / "model_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")

  logger.info("Saved Keras model and serving bundle to %s", models_dir)
  return meta


def run_training(
    paths: PathsConfig | None = None,
    nn_config: KerasNNConfig | None = None,
    max_train_samples: int | None = None,
    mlflow_config: MLflowConfig | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:

  paths = paths or PathsConfig()
  nn_config = nn_config or KerasNNConfig()
  mlflow_cfg = mlflow_config or MLflowConfig()
  keras.utils.set_random_seed(nn_config.random_seed)

  if not (paths.processed_dir / "split_metadata.json").exists():
    logger.info(
        "Processed splits not found in %s; running DatasetBuilder...",
        paths.processed_dir,
    )
    builder = DatasetBuilder(
        DatasetBuilderConfig(
            raw_dir=paths.raw_dir, processed_dir=paths.processed_dir
        )
    )
    bundle = builder.build(save=True, validate_integrity=True)
  else:
    bundle = load_processed_datasets(processed_dir=paths.processed_dir)

  train_inputs, train_y, train_w = build_feature_dict_from_impressions(
      bundle.train_impressions,
      bundle.users_enriched,
      bundle.items_enriched,
      max_rows=max_train_samples,
      seed=nn_config.random_seed,
  )
  val_inputs, val_y, _ = build_feature_dict_from_impressions(
      bundle.val_impressions,
      bundle.users_enriched,
      bundle.items_enriched,
      max_rows=(max_train_samples // 4) if max_train_samples else None,
      seed=nn_config.random_seed,
  )

  full_model, user_tower, item_tower = build_keras_recommender_nn(
      users_df=bundle.users_enriched,
      items_df=bundle.items_enriched,
      config=nn_config,
  )

  mlflow_mod = setup_mlflow(mlflow_cfg)
  resolved_run_name = (
      run_name
      or f"train-emb{nn_config.embedding_dim}-lr{nn_config.learning_rate}-ep{nn_config.epochs}"
  )
  run_ctx = (
      mlflow_mod.start_run(run_name=resolved_run_name)
      if mlflow_mod is not None
      else nullcontext()
  )

  with run_ctx as active_run:
    run_id = active_run.info.run_id if active_run is not None else None

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_roc_auc",
            mode="max",
            patience=nn_config.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_roc_auc",
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
        ),
    ]

    if mlflow_mod is not None:
      mlflow_mod.set_tags(
          {
              "stage": "training",
              "model_architecture": "two_tower_debiased_neural_ranker",
              "keras_version": keras.__version__,
              "polars_version": pl.__version__,
          }
      )
      mlflow_mod.log_params(
          {
              "embedding_dim": nn_config.embedding_dim,
              "tower_hidden_units": str(list(nn_config.tower_hidden_units)),
              "ranker_hidden_units": str(list(nn_config.ranker_hidden_units)),
              "dropout_rate": nn_config.dropout_rate,
              "l2_reg": nn_config.l2_reg,
              "learning_rate": nn_config.learning_rate,
              "batch_size": nn_config.batch_size,
              "epochs": nn_config.epochs,
              "early_stopping_patience": nn_config.early_stopping_patience,
              "max_position": nn_config.max_position,
              "default_serving_position": nn_config.default_serving_position,
              "random_seed": nn_config.random_seed,
              "max_train_samples": str(max_train_samples),
              "train_samples": int(len(train_y)),
              "val_samples": int(len(val_y)),
              "num_users": int(bundle.users_enriched.height),
              "num_items": int(bundle.items_enriched.height),
          }
      )
      callbacks.append(create_keras_epoch_logger(mlflow_mod))

    history = full_model.fit(
        x=train_inputs,
        y=train_y,
        sample_weight=train_w,
        validation_data=(val_inputs, val_y),
        batch_size=nn_config.batch_size,
        epochs=nn_config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_dict = {
        k: [round(float(v), 6) for v in vals]
        for k, vals in history.history.items()
    }

    meta = export_serving_bundle(
        full_model=full_model,
        user_tower=user_tower,
        item_tower=item_tower,
        bundle=bundle,
        models_dir=paths.models_dir,
        config=nn_config,
        training_history=history_dict,
        mlflow_run_id=run_id,
    )

    if mlflow_mod is not None:
      val_aucs = history_dict.get("val_roc_auc", [])
      val_pr_aucs = history_dict.get("val_pr_auc", [])
      val_losses = history_dict.get("val_loss", [])
      summary_metrics: dict[str, Any] = {
          "epochs_ran": len(val_aucs) or len(history_dict.get("loss", [])),
      }
      if val_aucs:
        best_idx = int(np.argmax(val_aucs))
        summary_metrics["best_epoch"] = best_idx + 1
        summary_metrics["best_val_roc_auc"] = val_aucs[best_idx]
        if best_idx < len(val_pr_aucs):
          summary_metrics["best_val_pr_auc"] = val_pr_aucs[best_idx]
        if best_idx < len(val_losses):
          summary_metrics["best_val_loss"] = val_losses[best_idx]
      log_numeric_metrics(mlflow_mod, summary_metrics)

      for artifact_name in (
          "keras_recommender.keras",
          "serving_bundle.npz",
          "popularity_baseline.json",
          "model_metadata.json",
      ):
        artifact_file = paths.models_dir / artifact_name
        if artifact_file.exists():
          mlflow_mod.log_artifact(
              str(artifact_file), artifact_path="model_artifacts"
          )

    return meta


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Train the production Keras Neural Network Recommender."
  )
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--embedding-dim", type=int, default=32)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--max-train-samples", type=int, default=None)
  parser.add_argument(
      "--no-mlflow",
      action="store_true",
      help="Disable MLflow experiment tracking for this run.",
  )
  parser.add_argument(
      "--tracking-uri",
      type=str,
      default=None,
      help="Override MLflow tracking URI (defaults to MLFLOW_TRACKING_URI or local SQLite).",
  )
  parser.add_argument(
      "--experiment-name",
      type=str,
      default=None,
      help="Override MLflow experiment name.",
  )
  parser.add_argument(
      "--run-name",
      type=str,
      default=None,
      help="Optional custom MLflow run name.",
  )
  return parser.parse_args()


def main() -> None:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s | %(levelname)s | %(message)s",
  )
  args = parse_args()
  cfg = KerasNNConfig(
      embedding_dim=args.embedding_dim,
      learning_rate=args.learning_rate,
      batch_size=args.batch_size,
      epochs=args.epochs,
  )
  default_mlflow = MLflowConfig()
  mlflow_cfg = MLflowConfig(
      enabled=not args.no_mlflow,
      store_dir=default_mlflow.store_dir,
      tracking_uri=args.tracking_uri or default_mlflow.tracking_uri,
      experiment_name=args.experiment_name or default_mlflow.experiment_name,
      artifact_location=default_mlflow.artifact_location,
  )
  meta = run_training(
      nn_config=cfg,
      max_train_samples=args.max_train_samples,
      mlflow_config=mlflow_cfg,
      run_name=args.run_name,
  )
  print(json.dumps(meta, indent=2))


if __name__ == "__main__":
  main()
