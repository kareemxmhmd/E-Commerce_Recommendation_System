from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import logging
import time
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from api.monitoring import (
    get_json_access_logger,
    normalize_route_path,
    record_catalog_state,
    record_http_request,
    record_recommendation_event,
    render_prometheus_metrics,
)
from api.schemas import (
    HealthResponse,
    RankCandidatesRequest,
    RecommendedItemResponse,
    RecommendRequest,
    RecommendResponse,
)
from src.config import PathsConfig, RerankConfig
from src.inference.predictor import ProductionRecommender

logger = logging.getLogger(__name__)
access_logger = get_json_access_logger()
paths_config = PathsConfig()

recommender_service = ProductionRecommender(
    paths=paths_config, rerank_config=RerankConfig()
)


def _ensure_loaded() -> ProductionRecommender:
  if recommender_service.user_embeddings is None:
    recommender_service.load()
    record_catalog_state(
        in_stock_items=(
            len(recommender_service.baseline.in_stock_items)
            if recommender_service.baseline
            else 0
        ),
        known_users=len(recommender_service.user_profiles),
        model_loaded=recommender_service.user_embeddings is not None,
    )
  return recommender_service


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
  _ensure_loaded()
  yield


app = FastAPI(
    title="Production E-Commerce Keras Recommender API",
    version="1.0.0",
    description=(
        "Production Two-Tower Retrieval + Position-Debiased Keras Neural Ranking API "
        "with cold-start demographic fallback and catalog diversity reranking."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
  request_id = request.headers.get("X-Request-Id") or str(uuid4())
  start = time.perf_counter()
  status_code = 500
  try:
    response = await call_next(request)
    status_code = response.status_code
  except Exception:
    elapsed_sec = time.perf_counter() - start
    endpoint = normalize_route_path(request)
    record_http_request(request.method, endpoint, status_code, elapsed_sec)
    access_logger.exception(
        "http_request_failed",
        extra={
            "structured_fields": {
                "event": "http_request_failed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "endpoint": endpoint,
                "status_code": status_code,
                "latency_ms": round(elapsed_sec * 1000.0, 2),
            }
        },
    )
    raise

  elapsed_sec = time.perf_counter() - start
  elapsed_ms = elapsed_sec * 1000.0
  endpoint = normalize_route_path(request)
  if endpoint != "/prometheus":
    record_http_request(request.method, endpoint, status_code, elapsed_sec)
    access_logger.info(
        "http_request",
        extra={
            "structured_fields": {
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "endpoint": endpoint,
                "status_code": status_code,
                "latency_ms": round(elapsed_ms, 2),
            }
        },
    )
  response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
  response.headers["X-Request-Id"] = request_id
  return response


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:

  svc = _ensure_loaded()
  warm_count = sum(
      1
      for p in svc.user_profiles.values()
      if not p.get("is_cold_start_user", False)
  )
  return HealthResponse(
      status="healthy",
      model_loaded=svc.user_embeddings is not None,
      catalog_size=len(svc.item_id_to_idx),
      in_stock_items=len(svc.baseline.in_stock_items) if svc.baseline else 0,
      known_users=len(svc.user_profiles),
      warm_users=warm_count,
      artifact_metadata=svc.metadata,
  )


@app.get("/metrics", tags=["System"])
def get_offline_metrics() -> Dict[str, Any]:

  metrics_path = paths_config.reports_dir / "evaluation_metrics.json"
  if not metrics_path.exists():
    raise HTTPException(
        status_code=404,
        detail=(
            "Offline evaluation metrics not found. Run `python -m"
            " src.evaluation.evaluate` first."
        ),
    )
  with open(metrics_path, "r", encoding="utf-8") as f:
    return json.load(f)


@app.get("/prometheus", tags=["Monitoring"])
def prometheus_metrics() -> Response:
  _ensure_loaded()
  payload, content_type = render_prometheus_metrics()
  return Response(content=payload, media_type=content_type)


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend_post(payload: RecommendRequest) -> RecommendResponse:

  svc = _ensure_loaded()
  _, is_cold = svc._get_user_vector(user_id=payload.user_id, city=payload.city)
  recs = svc.recommend(
      user_id=payload.user_id,
      top_k=payload.top_k,
      retrieval_candidates=payload.retrieval_candidates,
      serving_position=payload.debiased_position,
      city=payload.city,
      device=payload.device,
      language=payload.language,
      exclude_seen=payload.exclude_seen,
      filter_out_of_stock=payload.filter_out_of_stock,
      apply_diversity_rerank=payload.apply_diversity_rerank,
  )
  stage_name = recs[0].stage if recs else ("cold_start_empty" if is_cold else "warm_empty")
  record_recommendation_event(
      is_cold_start=is_cold,
      stage=stage_name,
      scores=[r.score for r in recs],
  )
  return RecommendResponse(
      user_id=payload.user_id,
      is_cold_start=is_cold,
      debiased_position=payload.debiased_position,
      count=len(recs),
      recommendations=[RecommendedItemResponse(**r.to_dict()) for r in recs],
  )


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendResponse,
    tags=["Recommendations"],
)
def recommend_get(
    user_id: int,
    top_k: int = Query(10, ge=1, le=100),
    exclude_seen: bool = Query(True),
    apply_diversity_rerank: bool = Query(True),
    debiased_position: int = Query(1, ge=1, le=10),
) -> RecommendResponse:

  req = RecommendRequest(
      user_id=user_id,
      top_k=top_k,
      exclude_seen=exclude_seen,
      apply_diversity_rerank=apply_diversity_rerank,
      debiased_position=debiased_position,
  )
  return recommend_post(req)


@app.post(
    "/rank", response_model=List[RecommendedItemResponse], tags=["Ranking"]
)
def rank_candidates(
    payload: RankCandidatesRequest,
) -> List[RecommendedItemResponse]:

  svc = _ensure_loaded()
  assert svc.reranker is not None
  valid_ids = [
      iid for iid in payload.candidate_item_ids if iid in svc.item_id_to_idx
  ]
  if not valid_ids:
    raise HTTPException(
        status_code=400, detail="None of the candidate_item_ids exist in catalog."
    )
  user_vec, _ = svc._get_user_vector(user_id=payload.user_id)
  cand_indices = np.array(
      [svc.item_id_to_idx[iid] for iid in valid_ids], dtype=np.int64
  )
  scores = svc.score_candidates_mlp(
      user_vec=user_vec,
      candidate_indices=cand_indices,
      position=payload.debiased_position,
  ).tolist()
  ranked = svc.reranker.rerank(
      candidate_ids=valid_ids,
      candidate_scores=scores,
      seen_items=set(),
      top_k=len(valid_ids),
      filter_out_of_stock=False,
      max_items_per_seller=len(valid_ids),
      max_items_per_leaf_category=len(valid_ids),
      stage_label="explicit_candidate_rerank",
  )
  return [RecommendedItemResponse(**r.to_dict()) for r in ranked]


@app.get(
    "/similar-items/{item_id}",
    response_model=List[RecommendedItemResponse],
    tags=["Catalog"],
)
def get_similar_items(
    item_id: int,
    top_k: int = Query(10, ge=1, le=50),
    in_stock_only: bool = Query(True),
) -> List[RecommendedItemResponse]:

  svc = _ensure_loaded()
  if item_id not in svc.item_id_to_idx:
    raise HTTPException(status_code=404, detail=f"Item {item_id} not found.")
  sims = svc.similar_items(
      item_id=item_id, top_k=top_k, filter_out_of_stock=in_stock_only
  )
  return [RecommendedItemResponse(**r.to_dict()) for r in sims]


@app.get("/users/{user_id}/profile", tags=["Users"])
def get_user_profile(user_id: int) -> Dict[str, Any]:

  svc = _ensure_loaded()
  profile = svc.user_profiles.get(int(user_id))
  if profile is None:
    return {
        "user_id": int(user_id),
        "known_in_catalog": False,
        "is_cold_start": True,
        "city": "unknown",
        "device": "unknown",
        "language": "en",
        "num_train_interactions": 0,
    }
  return {
      "user_id": int(user_id),
      "known_in_catalog": True,
      "is_cold_start": bool(profile.get("is_cold_start_user", False)),
      "city": str(profile.get("city", "unknown")),
      "device": str(profile.get("device", "unknown")),
      "language": str(profile.get("language", "en")),
      "num_train_interactions": int(profile.get("train_interaction_count", 0)),
      "train_purchase_count": int(profile.get("train_purchase_count", 0)),
      "train_user_ctr": float(profile.get("train_user_ctr", 0.0)),
  }
