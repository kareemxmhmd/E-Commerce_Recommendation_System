from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):

  user_id: Optional[int] = Field(
      None, description="Target user ID (omit or pass unseen ID for cold-start)"
  )
  top_k: int = Field(10, ge=1, le=100, description="Number of items to return")
  retrieval_candidates: int = Field(
      100, ge=10, le=500, description="Number of Stage-1 candidates to retrieve"
  )
  exclude_seen: bool = Field(
      True,
      description="Exclude items the user already interacted with in training",
  )
  filter_out_of_stock: bool = Field(
      True, description="Filter out items with stock <= 0"
  )
  apply_diversity_rerank: bool = Field(
      True,
      description="Apply seller and leaf-category diversity caps during reranking",
  )
  debiased_position: int = Field(
      1,
      ge=1,
      le=10,
      description="Fixed display position index used at serving for position debiasing (default=1)",
  )
  city: Optional[str] = Field(
      None, description="Optional city override for cold-start demographic fallback"
  )
  device: Optional[str] = Field(
      None, description="Optional device override for cold-start fallback"
  )
  language: Optional[str] = Field(
      None, description="Optional language override for cold-start fallback"
  )


class RecommendedItemResponse(BaseModel):

  item_id: int
  title: str
  category_id: int
  category_name: str
  root_category_id: int
  root_category_name: str
  brand: str
  price: float
  seller_id: int
  seller_name: str
  seller_rating: float
  stock: int
  score: float
  stage: str


class RecommendResponse(BaseModel):

  user_id: Optional[int]
  is_cold_start: bool
  debiased_position: int
  count: int
  recommendations: List[RecommendedItemResponse]


class RankCandidatesRequest(BaseModel):

  user_id: int = Field(..., ge=1, description="Target user ID")
  candidate_item_ids: List[int] = Field(
      ..., min_items=1, description="Candidate item IDs to score and rank"
  )
  debiased_position: int = Field(
      1, ge=1, le=10, description="Position index for debiased scoring"
  )


class HealthResponse(BaseModel):

  status: str
  model_loaded: bool
  catalog_size: int
  in_stock_items: int
  known_users: int
  warm_users: int
  artifact_metadata: Dict[str, Any]
