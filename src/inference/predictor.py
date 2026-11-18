from __future__ import annotations

import json
import logging
from typing import Any
import numpy as np

from src.config import PathsConfig, RerankConfig
from src.data.loader import ProcessedDatasetBundle, load_processed_datasets
from src.models.baseline import PopularityBaseline
from src.models.reranker import CatalogReranker, RankedCandidate

logger = logging.getLogger(__name__)


def _relu(x: np.ndarray) -> np.ndarray:
  return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class ProductionRecommender:

  def __init__(
      self,
      paths: PathsConfig | None = None,
      rerank_config: RerankConfig | None = None,
  ) -> None:
    self.paths = paths or PathsConfig()
    self.rerank_config = rerank_config or RerankConfig()
    self.bundle: ProcessedDatasetBundle | None = None
    self.baseline: PopularityBaseline | None = None
    self.reranker: CatalogReranker | None = None

    self.user_id_to_idx: dict[int, int] = {}
    self.item_id_to_idx: dict[int, int] = {}
    self.user_embeddings: np.ndarray | None = None
    self.item_embeddings: np.ndarray | None = None
    self.item_ids: np.ndarray | None = None
    self.in_stock_mask: np.ndarray | None = None

    self.mlp_0_w: np.ndarray | None = None
    self.mlp_0_b: np.ndarray | None = None
    self.mlp_1_w: np.ndarray | None = None
    self.mlp_1_b: np.ndarray | None = None
    self.logit_w: np.ndarray | None = None
    self.logit_b: np.ndarray | None = None
    self.pos_bias_w: np.ndarray | None = None

    self.user_profiles: dict[int, dict[str, Any]] = {}
    self.user_seen_items: dict[int, set[int]] = {}
    self.city_mean_embeddings: dict[str, np.ndarray] = {}
    self.global_mean_user_embedding: np.ndarray | None = None
    self.metadata: dict[str, Any] = {}

  def load(self) -> ProductionRecommender:

    self.bundle = load_processed_datasets(self.paths.processed_dir)
    self.baseline = PopularityBaseline().fit(
        train_interactions=self.bundle.train_interactions,
        items_enriched=self.bundle.items_enriched,
        users_enriched=self.bundle.users_enriched,
    )
    self.reranker = CatalogReranker(
        items_enriched=self.bundle.items_enriched,
        config=self.rerank_config,
    )
    self.user_seen_items = self.baseline.user_seen_items

    for row in self.bundle.users_enriched.iter_rows(named=True):
      self.user_profiles[int(row["user_id"])] = dict(row)

    bundle_path = self.paths.models_dir / "serving_bundle.npz"
    meta_path = self.paths.models_dir / "model_metadata.json"
    if not bundle_path.exists():
      raise FileNotFoundError(
          f"Trained serving bundle not found at {bundle_path}. Run"
          " `python -m src.training.train` first."
      )

    npz = np.load(bundle_path)
    user_ids = npz["user_ids"].astype(np.int64)
    item_ids = npz["item_ids"].astype(np.int64)
    self.user_embeddings = npz["user_embeddings"].astype(np.float32)
    self.item_embeddings = npz["item_embeddings"].astype(np.float32)
    self.item_ids = item_ids

    self.user_id_to_idx = {int(uid): idx for idx, uid in enumerate(user_ids)}
    self.item_id_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}

    self.mlp_0_w = npz["mlp_0_w"].astype(np.float32)
    self.mlp_0_b = npz["mlp_0_b"].astype(np.float32)
    self.mlp_1_w = npz["mlp_1_w"].astype(np.float32)
    self.mlp_1_b = npz["mlp_1_b"].astype(np.float32)
    self.logit_w = npz["logit_w"].astype(np.float32)
    self.logit_b = npz["logit_b"].astype(np.float32)
    self.pos_bias_w = npz["pos_bias_w"].astype(np.float32)

    in_stock_set = self.baseline.in_stock_items
    self.in_stock_mask = np.array(
        [int(iid) in in_stock_set for iid in item_ids], dtype=bool
    )

    self.global_mean_user_embedding = np.mean(self.user_embeddings, axis=0)
    city_buckets: dict[str, list[np.ndarray]] = {}
    for uid, profile in self.user_profiles.items():
      if not profile.get("is_cold_start_user", False) and uid in self.user_id_to_idx:
        c = str(profile.get("city", "unknown"))
        city_buckets.setdefault(c, []).append(
            self.user_embeddings[self.user_id_to_idx[uid]]
        )
    for c, vecs in city_buckets.items():
      mean_v = np.mean(np.stack(vecs, axis=0), axis=0)
      norm = max(float(np.linalg.norm(mean_v)), 1e-8)
      self.city_mean_embeddings[c] = (mean_v / norm).astype(np.float32)

    if meta_path.exists():
      with meta_path.open("r", encoding="utf-8") as f:
        self.metadata = json.load(f)

    logger.info(
        "Loaded ProductionRecommender: %d users, %d items",
        len(self.user_id_to_idx),
        len(self.item_id_to_idx),
    )
    return self

  def _get_user_vector(
      self,
      user_id: int | None,
      city: str | None = None,
  ) -> tuple[np.ndarray, bool]:

    assert self.user_embeddings is not None
    assert self.global_mean_user_embedding is not None

    if user_id is not None and int(user_id) in self.user_id_to_idx:
      profile = self.user_profiles.get(int(user_id), {})
      is_cold = bool(profile.get("is_cold_start_user", False))
      if not is_cold:
        return self.user_embeddings[self.user_id_to_idx[int(user_id)]], False
      city = city or str(profile.get("city", ""))

    if city and city in self.city_mean_embeddings:
      return self.city_mean_embeddings[city], True

    return self.global_mean_user_embedding, True

  def score_candidates_mlp(
      self,
      user_vec: np.ndarray,
      candidate_indices: np.ndarray,
      position: int = 1,
  ) -> np.ndarray:

    assert self.item_embeddings is not None
    assert self.mlp_0_w is not None and self.mlp_0_b is not None
    assert self.mlp_1_w is not None and self.mlp_1_b is not None
    assert self.logit_w is not None and self.logit_b is not None
    assert self.pos_bias_w is not None

    item_vecs = self.item_embeddings[candidate_indices]  # (N, D)
    u_tiled = np.broadcast_to(
        user_vec.reshape(1, -1), item_vecs.shape
    ).astype(np.float32)
    hadamard = u_tiled * item_vecs
    dot_sim = np.sum(hadamard, axis=-1, keepdims=True)

    features = np.concatenate([u_tiled, item_vecs, hadamard, dot_sim], axis=-1)
    h0 = _relu(features @ self.mlp_0_w + self.mlp_0_b)
    h1 = _relu(h0 @ self.mlp_1_w + self.mlp_1_b)
    rel_logit = (h1 @ self.logit_w + self.logit_b).reshape(-1)

    pos_idx = int(np.clip(position, 1, len(self.pos_bias_w) - 1))
    pos_logit = float(self.pos_bias_w[pos_idx, 0])
    return _sigmoid(rel_logit + pos_logit)

  def retrieve_two_tower(
      self,
      user_id: int | None = None,
      top_k: int = 10,
      city: str | None = None,
      exclude_seen: bool = True,
  ) -> list[int]:

    assert self.item_embeddings is not None and self.item_ids is not None
    assert self.in_stock_mask is not None and self.baseline is not None

    user_vec, is_cold = self._get_user_vector(user_id=user_id, city=city)
    if is_cold:
      return self.baseline.recommend(
          user_id=user_id, top_k=top_k, city=city, exclude_seen=exclude_seen
      )

    dots = self.item_embeddings @ user_vec
    dots = np.where(self.in_stock_mask, dots, -1e9)

    seen = self.user_seen_items.get(int(user_id), set()) if (exclude_seen and user_id is not None) else set()
    top_indices = np.argsort(dots)[::-1]
    results: list[int] = []
    for idx in top_indices:
      iid = int(self.item_ids[idx])
      if iid in seen:
        continue
      results.append(iid)
      if len(results) >= top_k:
        break
    return results

  def recommend(
      self,
      user_id: int | None = None,
      top_k: int = 10,
      retrieval_candidates: int = 100,
      serving_position: int = 1,
      city: str | None = None,
      device: str | None = None,
      language: str | None = None,
      exclude_seen: bool = True,
      filter_out_of_stock: bool = True,
      apply_diversity_rerank: bool = True,
  ) -> list[RankedCandidate]:

    assert self.item_embeddings is not None and self.item_ids is not None
    assert self.baseline is not None and self.reranker is not None

    user_vec, is_cold = self._get_user_vector(user_id=user_id, city=city)
    seen = (
        self.user_seen_items.get(int(user_id), set())
        if (exclude_seen and user_id is not None)
        else set()
    )

    dots = self.item_embeddings @ user_vec
    if filter_out_of_stock and self.in_stock_mask is not None:
      dots = np.where(self.in_stock_mask, dots, -1e9)

    n_cand = max(retrieval_candidates, top_k * 4)
    top_cand_indices = np.argsort(dots)[::-1][:n_cand]

    if is_cold:
      pop_ids = self.baseline.recommend(
          user_id=user_id,
          top_k=n_cand,
          city=city,
          language=language,
          exclude_seen=exclude_seen,
      )
      pop_indices = np.array(
          [self.item_id_to_idx[iid] for iid in pop_ids if iid in self.item_id_to_idx],
          dtype=np.int64,
      )
      if len(pop_indices) > 0:
        top_cand_indices = np.unique(
            np.concatenate([pop_indices[: n_cand // 2], top_cand_indices[: n_cand // 2]])
        )

    mlp_scores = self.score_candidates_mlp(
        user_vec=user_vec,
        candidate_indices=top_cand_indices,
        position=serving_position,
    )
    cand_item_ids = [int(self.item_ids[idx]) for idx in top_cand_indices]

    if is_cold:
      pop_scores = np.array(
          [self.baseline.item_scores.get(iid, 0.0) for iid in cand_item_ids],
          dtype=np.float32,
      )
      final_scores = (0.55 * mlp_scores + 0.45 * pop_scores).tolist()
      stage_name = "cold_start_hybrid_nn"
    else:
      final_scores = mlp_scores.tolist()
      stage_name = (
          "keras_nn_debiased"
          if serving_position == 1
          else f"keras_nn_pos_{serving_position}"
      )

    if apply_diversity_rerank:
      return self.reranker.rerank(
          candidate_ids=cand_item_ids,
          candidate_scores=final_scores,
          seen_items=seen,
          top_k=top_k,
          filter_out_of_stock=filter_out_of_stock,
          stage_label=stage_name,
      )

    return self.reranker.rerank(
        candidate_ids=cand_item_ids,
        candidate_scores=final_scores,
        seen_items=seen,
        top_k=top_k,
        filter_out_of_stock=filter_out_of_stock,
        max_items_per_seller=top_k,
        max_items_per_leaf_category=top_k,
        stage_label=stage_name,
    )

  def similar_items(
      self, item_id: int, top_k: int = 10, filter_out_of_stock: bool = True
  ) -> list[RankedCandidate]:

    assert self.item_embeddings is not None and self.item_ids is not None
    assert self.reranker is not None

    if int(item_id) not in self.item_id_to_idx:
      raise KeyError(f"Item ID {item_id} not found in catalog.")

    target_idx = self.item_id_to_idx[int(item_id)]
    target_vec = self.item_embeddings[target_idx]
    sims = (self.item_embeddings @ target_vec).astype(np.float32)
    sims[target_idx] = -1e9
    if filter_out_of_stock and self.in_stock_mask is not None:
      sims = np.where(self.in_stock_mask, sims, -1e9)

    best_idx = np.argsort(sims)[::-1][: top_k * 3]
    cand_ids = [int(self.item_ids[i]) for i in best_idx]
    cand_scores = [float(sims[i]) for i in best_idx]

    return self.reranker.rerank(
        candidate_ids=cand_ids,
        candidate_scores=cand_scores,
        seen_items={int(item_id)},
        top_k=top_k,
        filter_out_of_stock=filter_out_of_stock,
        stage_label="keras_item_to_item",
    )
