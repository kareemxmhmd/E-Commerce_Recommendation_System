import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PathsConfig, RerankConfig
from src.inference.predictor import ProductionRecommender

st.set_page_config(
    page_title="Production Keras Recommender System",
    page_icon="🛍️",
    layout="wide",
)


@st.cache_resource
def load_recommender() -> ProductionRecommender:
  svc = ProductionRecommender(paths=PathsConfig(), rerank_config=RerankConfig())
  svc.load()
  return svc


def main() -> None:
  st.title("E-Commerce Neural Recommender & Position-Debiased Ranking Studio")

  svc = load_recommender()
  paths = PathsConfig()

  with st.sidebar:
    st.header("Inference Controls")
    user_id = st.number_input(
        "Select User ID (1 - 8000, or >8000 for new cold-start user)",
        min_value=1,
        max_value=9000,
        value=42,
        step=1,
    )
    top_k = st.slider("Top-K Recommendations", min_value=5, max_value=30, value=10)
    exclude_seen = st.checkbox("Exclude Previously Interacted Items", value=True)
    apply_diversity = st.checkbox(
        "Apply Seller & Category Diversity Caps", value=True
    )
    debiased_pos = st.slider(
        "Serving Position Feature (1 = Debiased Top Slot)",
        min_value=1,
        max_value=10,
        value=1,
    )
    st.divider()
    st.subheader("Cold-Start Override (Optional)")
    city_override = st.selectbox(
        "City Override",
        [
            "(Use Profile)",
            "Cairo",
            "London",
            "Berlin",
            "New York",
            "Dubai",
            "Paris",
            "Singapore",
        ],
    )
    lang_override = st.selectbox(
        "Language Override", ["(Use Profile)", "en", "ar", "de", "fr", "es"]
    )

  tab_rec, tab_sim, tab_metrics = st.tabs(
      [
          "Personalized Recommendations",
          "Item-to-Item Embedding Similarity",
          "Offline Benchmark Scorecard",
      ]
  )

  with tab_rec:
    profile = svc.user_profiles.get(int(user_id), {})
    is_cold = bool(profile.get("is_cold_start_user", True))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("User ID", int(user_id))
    col2.metric(
        "User Status",
        "Cold-Start Fallback" if is_cold else "Warm (Trained)",
    )
    col3.metric(
        "Demographics",
        f"{profile.get('city', 'N/A')} | {profile.get('language', 'N/A')}",
    )
    col4.metric(
        "Train Interactions", int(profile.get("train_interaction_count", 0))
    )

    recs = svc.recommend(
        user_id=int(user_id),
        top_k=int(top_k),
        serving_position=int(debiased_pos),
        city=None if city_override == "(Use Profile)" else city_override,
        language=None if lang_override == "(Use Profile)" else lang_override,
        exclude_seen=exclude_seen,
        apply_diversity_rerank=apply_diversity,
    )

    st.subheader(f"Top-{len(recs)} Ranked Recommendations (Position={debiased_pos})")
    if recs:
      st.dataframe(
          pd.DataFrame([r.to_dict() for r in recs]), use_container_width=True
      )

  with tab_sim:
    st.subheader("Nearest Neighbors in Keras Item Tower Embedding Space")
    seed_item_id = st.number_input(
        "Seed Item ID (1 - 4000)", min_value=1, max_value=4000, value=101, step=1
    )
    if svc.reranker and int(seed_item_id) in svc.reranker.catalog_map:
      seed_meta = svc.reranker.catalog_map[int(seed_item_id)]
      st.info(
          f"**Seed Item #{seed_item_id}**: {seed_meta['title']} | Brand:"
          f" `{seed_meta['brand']}` | Category: `{seed_meta['category_name']}`"
          f" | Price: `${seed_meta['price']:.2f}` | Stock:"
          f" `{seed_meta['stock']}`"
      )
      sim_items = svc.similar_items(int(seed_item_id), top_k=10)
      st.dataframe(
          pd.DataFrame([r.to_dict() for r in sim_items]),
          use_container_width=True,
      )

  with tab_metrics:
    st.subheader("Offline Evaluation Metrics (Test Split)")
    metrics_file = paths.reports_dir / "evaluation_metrics.json"
    report_md = paths.reports_dir / "metrics_report.md"
    if report_md.exists():
      st.markdown(report_md.read_text(encoding="utf-8"))
    if metrics_file.exists():
      with st.expander("Raw JSON Evaluation Metrics"):
        st.json(json.loads(metrics_file.read_text(encoding="utf-8")))


if __name__ == "__main__":
  main()
