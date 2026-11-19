# E-Commerce Neural Recommendation System

A production-ready two-stage recommendation system that combines Two-Tower retrieval, position-debiased Keras neural ranking, cold-start demographic fallback, and marketplace diversity reranking.

**Live App:** [E-Commerce Recommendation System (Streamlit)](https://e-commercerecommendationsystem-kd7.streamlit.app)

---

## 1. Business Focus

- **Problem:** Naive popularity recommenders concentrate exposure on a tiny fraction of top-selling products, suffer from presentation position bias in logged click data, fail on new cold-start users, and risk surfacing out-of-stock items or single-seller spam.
- **Marketplace & Ranking Constraints:**
  - **Position Bias Debiasing:** Logged impression clicks heavily favor top display slots; training uses Inverse Propensity Weighting (IPW) and an additive position-bias tower, then serves at a fixed debiased slot (`position = 1`).
  - **Inventory & Diversity Guardrails:** Hard exclusion of out-of-stock products (`324` inactive items filtered from the `4,000`-item catalog), seen-item filtering, and reranking caps (`max 2` items/seller, `max 3` items/leaf category, diversity penalty `0.08`).
  - **Strict Cold-Start Coverage:** Hybrid demographic-embedding + segment-popularity fallback (`city | language`) for unseen users (`825` cold-start users in the `8,000`-user catalog).
- **Impact & Results (Held-Out Test Set — `1,000` Users, `3,676` Active In-Stock Items):**
  - The position-debiased neural ranker achieves **`Val ROC-AUC = 0.708`** (`Val PR-AUC = 0.288`, `Val Loss = 0.375`) and learns a clean monotonic position-bias decay from **`+1.024`** (Slot 1) to **`-0.997`** (Slot 10).
  - **Catalog Coverage@10** increases **3.8x** over the Demographic Popularity Baseline — from **10.0%** to **38.5%** — while **Novelty@10** improves from **9.35** to **11.03**, with **0%** out-of-stock recommendations.
  - On strict cold-start users, the hybrid neural ranker lifts **Cold Recall@10** from **7.62%** to **9.75%** (**+27.9% relative lift**) and **Cold NDCG@10** from **0.0531** to **0.0652** (**+22.8% relative lift**), while Stage-2 neural ranking improves overall **Recall@10** by **+62.5%** over raw Two-Tower retrieval (`0.0072` to `0.0116`).

---

## 2. Software & ML Engineering Execution

- **Leakage-Free Temporal Data Pipeline:** Built a Polars-based ingestion and chronological split pipeline (`DataLoader`, `DatasetBuilder` in `src/data/loader.py`, `src/data/dataset_builder.py`) across `278,841` interactions and `718,850` impressions (`70/15/15` temporal split with IPW propensities and train-only catalog aggregates), paired with custom serializable Keras 3 layers (`UserPreprocessor`, `ItemPreprocessor`) in `src/data/preprocessing.py`.
- **Two-Stage Neural Architecture & Reranking:** Implemented L2-normalized User and Item Towers with Hadamard + dot-product interaction MLP and an additive position-bias branch (`src/models/keras_nn.py`), backed by a demographic `PopularityBaseline` (`src/models/baseline.py`) and constraint-enforcing `CatalogReranker` (`src/models/reranker.py`).
- **Experiment Tracking & Offline Evaluation:** Integrated MLflow with a local SQLite backend (`src/tracking.py`, `src/training/train.py`, `mlflow_store/mlflow.db`) to log hyperparameters, per-epoch curves, and model artifacts (`keras_recommender.keras`, `serving_bundle.npz`), alongside multi-variant offline evaluation (`src/evaluation/evaluate.py`, `src/evaluation/metrics.py`) tracking `Recall@K`, `NDCG@K`, `MAP@K`, `MRR@K`, `Coverage`, `Novelty`, `Category KL`, and warm vs. cold-start breakdowns.
- **Low-Latency Serving, Observability & UI:** Decoupled inference from TensorFlow via a NumPy/Polars `ProductionRecommender` (`src/inference/predictor.py`, ~5 ms/request) served through a Pydantic-validated FastAPI service (`api/main.py` supporting `/recommend`, `/rank`, `/similar-items/{item_id}`, `/users/{user_id}/profile`, `/health`, `/metrics`, and Prometheus `/prometheus` + structured JSON access logs in `api/monitoring.py`) and an interactive Streamlit studio (`app/streamlit_app.py`), containerized with `Dockerfile`, a lean TensorFlow-free `Dockerfile.serving`, and `docker-compose.yml` (API, Streamlit, and MLflow UI).
- **Quality Assurance, Drift Monitoring & CI/CD:** Wrote 11 unit and integration tests (`tests/`) covering ranking metrics, zero temporal leakage, preprocessors, diversity/stock caps, inference, and FastAPI endpoints (`unittest` / `pytest`), automated syntax checks and testing in GitHub Actions (`.github/workflows/ci.yml`), and built PSI + KS-test feature, CTR, and cold-start drift monitoring in `src/monitoring/drift.py` (`artifacts/reports/drift_report.json`).
