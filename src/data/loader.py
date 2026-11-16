import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


EXPECTED_RAW_SCHEMAS: dict[str, set[str]] = {
    "items": {
        "item_id",
        "title",
        "category_id",
        "brand",
        "price",
        "seller_id",
        "created_at",
        "stock",
    },
    "users": {
        "user_id",
        "signup_date",
        "city",
        "device",
        "language",
    },
    "interactions": {
        "user_id",
        "item_id",
        "event_type",
        "timestamp",
        "session_id",
        "rating",
    },
    "impressions": {
        "user_id",
        "item_id",
        "session_id",
        "position",
        "clicked",
        "timestamp",
    },
    "categories": {
        "category_id",
        "name",
        "parent_id",
    },
    "sellers": {
        "seller_id",
        "name",
        "rating",
        "joined_at",
    },
}


@dataclass(frozen=True)
class RawDatasetBundle:
    
    items: pl.DataFrame
    users: pl.DataFrame
    interactions: pl.DataFrame
    impressions: pl.DataFrame
    categories: pl.DataFrame
    sellers: pl.DataFrame

    def as_dict(self) -> dict[str, pl.DataFrame]:
        return {
            "items": self.items,
            "users": self.users,
            "interactions": self.interactions,
            "impressions": self.impressions,
            "categories": self.categories,
            "sellers": self.sellers,
        }