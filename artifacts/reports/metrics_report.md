# Offline Ranking & Position-Debiasing Evaluation Report

- **Evaluated Test Users**: `1000` (`953` warm users, `47` strict cold-start users)
- **Active In-Stock Catalog Size**: `3676` items (`324` out-of-stock items excluded)

## Primary Ranking & Marketplace Health Scorecard (`@10`)

| Model Variant | Recall@10 | NDCG@10 | MAP@10 | HitRate@10 | MRR@10 | Coverage@10 | Novelty@10 | Category KL@10 (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Demographic Popularity Baseline** | 0.0198 | 0.0141 | 0.0090 | 0.0460 | 0.0195 | 0.1004 | 9.346 | 6.9287 |
| **Keras Two-Tower Retrieval** | 0.0072 | 0.0051 | 0.0031 | 0.0180 | 0.0077 | 0.4391 | 11.755 | 7.2548 |
| **Keras Neural Ranker (Noisy Pos=8)** | 0.0116 | 0.0065 | 0.0036 | 0.0250 | 0.0075 | 0.3817 | 11.047 | 7.0216 |
| **Keras Debiased Neural Ranker (Pos=1 + Diversity)** | 0.0116 | 0.0065 | 0.0036 | 0.0250 | 0.0075 | 0.3852 | 11.035 | 6.9962 |

## Warm vs. Strict Cold-Start Breakdown (`@10`)

| Model Variant | Warm Recall@10 | Warm NDCG@10 | Cold Recall@10 | Cold NDCG@10 |
| :--- | :---: | :---: | :---: | :---: |
| **Demographic Popularity Baseline** | 0.0171 | 0.0122 | 0.0762 | 0.0531 |
| **Keras Two-Tower Retrieval** | 0.0037 | 0.0028 | 0.0762 | 0.0531 |
| **Keras Neural Ranker (Noisy Pos=8)** | 0.0074 | 0.0036 | 0.0975 | 0.0649 |
| **Keras Debiased Neural Ranker (Pos=1 + Diversity)** | 0.0074 | 0.0036 | 0.0975 | 0.0651 |
