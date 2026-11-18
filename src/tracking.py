import logging
import math
import re
from typing import Any

from src.config import MLflowConfig

logger = logging.getLogger(__name__)

_INVALID_METRIC_CHARS = re.compile(r"[^a-zA-Z0-9_\-\. /:]")


def sanitize_metric_name(name: str) -> str:
  cleaned = name.replace("@", "_at_")
  return _INVALID_METRIC_CHARS.sub("_", cleaned)


def setup_mlflow(config: MLflowConfig | None = None) -> Any | None:
  cfg = config or MLflowConfig()
  if not cfg.enabled:
    return None

  try:
    import mlflow
  except ImportError:
    logger.warning(
        "MLflow is not installed; skipping experiment tracking."
    )
    return None

  if cfg.tracking_uri.startswith("sqlite:"):
    cfg.store_dir.mkdir(parents=True, exist_ok=True)
    (cfg.store_dir / "mlartifacts").mkdir(parents=True, exist_ok=True)

  mlflow.set_tracking_uri(cfg.tracking_uri)

  existing = mlflow.get_experiment_by_name(cfg.experiment_name)
  if existing is None:
    if cfg.tracking_uri.startswith("sqlite:"):
      mlflow.create_experiment(
          cfg.experiment_name, artifact_location=cfg.artifact_location
      )
    else:
      mlflow.create_experiment(cfg.experiment_name)

  mlflow.set_experiment(cfg.experiment_name)
  logger.info(
      "MLflow tracking configured: uri=%s, experiment=%s",
      cfg.tracking_uri,
      cfg.experiment_name,
  )
  return mlflow


def log_numeric_metrics(
    mlflow_mod: Any,
    metrics: dict[str, Any],
    step: int | None = None,
    prefix: str = "",
) -> None:
  payload: dict[str, float] = {}
  for key, val in metrics.items():
    try:
      num = float(val)
    except (TypeError, ValueError):
      continue
    if math.isnan(num) or math.isinf(num):
      continue
    metric_key = sanitize_metric_name(f"{prefix}{key}")
    payload[metric_key] = num

  if payload:
    mlflow_mod.log_metrics(payload, step=step)


def create_keras_epoch_logger(mlflow_mod: Any) -> Any:
  import keras

  class MLflowEpochCallback(keras.callbacks.Callback):

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
      if not logs:
        return
      log_numeric_metrics(mlflow_mod, logs, step=epoch + 1)

  return MLflowEpochCallback()
