from datetime import datetime, timezone
import json
import logging
from typing import Any, Sequence

from fastapi import Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

HTTP_REQUESTS_TOTAL = Counter(
    "recsys_http_requests_total",
    "Total HTTP requests handled by the Recommender API",
    labelnames=("method", "endpoint", "status_code"),
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "recsys_http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "endpoint"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

RECOMMENDATIONS_TOTAL = Counter(
    "recsys_recommendations_total",
    "Total recommendation requests served",
    labelnames=("user_type", "stage"),
)

RECOMMENDATION_SCORES = Histogram(
    "recsys_recommendation_score",
    "Distribution of predicted recommendation scores",
    labelnames=("user_type",),
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

RECOMMENDATION_RETURNED_COUNT = Histogram(
    "recsys_recommendation_returned_count",
    "Number of items returned per recommendation response",
    buckets=(0, 1, 3, 5, 10, 20, 50, 100),
)

CATALOG_IN_STOCK_ITEMS = Gauge(
    "recsys_catalog_in_stock_items",
    "Number of active in-stock items in the serving catalog",
)

CATALOG_KNOWN_USERS = Gauge(
    "recsys_catalog_known_users",
    "Number of known users in the serving catalog",
)

MODEL_LOADED = Gauge(
    "recsys_model_loaded",
    "1 when serving bundle is loaded in memory, 0 otherwise",
)


class _JsonLogFormatter(logging.Formatter):

  def format(self, record: logging.LogRecord) -> str:

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
    }

    extra_fields = getattr(record, "structured_fields", None)
    if isinstance(extra_fields, dict):
      payload.update(extra_fields)
    if record.exc_info:
      payload["exception"] = self.formatException(record.exc_info)
    return json.dumps(payload, ensure_ascii=False)


def get_json_access_logger(
    name: str = "recsys.api.access",
    level: int = logging.INFO,
) -> logging.Logger:
  
  log = logging.getLogger(name)
  log.setLevel(level)
  if not any(
      isinstance(h.formatter, _JsonLogFormatter) for h in log.handlers
  ):
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonLogFormatter())
    log.addHandler(handler)
    log.propagate = False
  return log


def normalize_route_path(request: Request) -> str:
  
  route = request.scope.get("route")
  route_path = getattr(route, "path", None)
  if isinstance(route_path, str) and route_path:
    return route_path
  return request.url.path


def record_http_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_seconds: float,
) -> None:
  
  HTTP_REQUESTS_TOTAL.labels(
      method=method,
      endpoint=endpoint,
      status_code=str(status_code),
  ).inc()

  HTTP_REQUEST_LATENCY_SECONDS.labels(
      method=method,
      endpoint=endpoint,
  ).observe(duration_seconds)


def record_catalog_state(
    in_stock_items: int,
    known_users: int,
    model_loaded: bool,
) -> None:
  
  CATALOG_IN_STOCK_ITEMS.set(max(int(in_stock_items), 0))
  CATALOG_KNOWN_USERS.set(max(int(known_users), 0))
  MODEL_LOADED.set(1.0 if model_loaded else 0.0)


def record_recommendation_event(
    is_cold_start: bool,
    stage: str,
    scores: Sequence[float],
) -> None:
  
  user_type = "cold_start" if is_cold_start else "warm"
  RECOMMENDATIONS_TOTAL.labels(
      user_type=user_type,
      stage=stage,
  ).inc()
  
  RECOMMENDATION_RETURNED_COUNT.observe(len(scores))
  for score in scores:
    RECOMMENDATION_SCORES.labels(user_type=user_type).observe(float(score))


def render_prometheus_metrics() -> tuple[bytes, str]:
  return generate_latest(), CONTENT_TYPE_LATEST
