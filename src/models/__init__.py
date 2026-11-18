from typing import Any

from src.models.baseline import PopularityBaseline
from src.models.reranker import CatalogReranker, RankedCandidate

_KERAS_EXPORTS = {
    "L2Normalize",
    "build_item_tower",
    "build_keras_recommender_nn",
    "build_user_tower",
}

__all__ = [
    "CatalogReranker",
    "L2Normalize",
    "PopularityBaseline",
    "RankedCandidate",
    "build_item_tower",
    "build_keras_recommender_nn",
    "build_user_tower",
]


def __getattr__(name: str) -> Any:
  if name in _KERAS_EXPORTS:
    from src.models import keras_nn

    return getattr(keras_nn, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
