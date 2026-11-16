import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(os.getenv(
     "RECSYS_PROJECT_ROOT",
     Path(__file__).resolve().parent[1],
))


@dataclass(frozen=True)
class PathsConfig:

     root_dir: Path = PROJECT_ROOT
     raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
     processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
     models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "models")
     reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "reports")
     ground_truth_path: Path = field(default_factory=lambda: (PROJECT_ROOT / "data" / "raw" / "ground_truth" / "user_leaf_pref.csv"))


@dataclass(frozen=True)
class KerasNNConfig:
    
    embedding_dim: int = 32
    tower_hidden_units: tuple[int, ...] = (64, 32)
    ranker_hidden_units: tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.20
    l2_reg: float = 1e-5
    learning_rate: float = 1e-3
    batch_size: int = 1024
    epochs: int = 10
    early_stopping_patience: int = 3
    max_position: int = 10
    default_serving_position: int = 1
    random_seed: int = 42


@dataclass(frozen=True)
class RerankConfig:

     top_k: int = 10
     retrieval_candidates: int = 100
     filter_out_of_stock: bool = True
     exclude_seen_items: bool = True
     max_items_per_seller: int = 2
     max_items_per_leaf_category: int = 3
     diversity_penalty: float = 0.08
