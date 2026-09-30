# geneval2 · 4 档 router · Qwen3-emb-8b + Ridge · train 400 vs 2400

4 档：qwen-10 / qwen-50 / flux9b-10 / flux9b-50

成本：10 / 50 / 23.1 / 115.5，预算 B=50，seeds=0/1/2

Backbone: Qwen3-Embedding-8B (mean-pool)，Predictor: 4 个独立 Ridge head

测试集：geneval2 raw 800


| train pool | 10-qwen | 50-qwen | 10-flux9b | 50-flux9b | oracle@50 | LP | OC | Q3+Ridge@50 |
|---|---|---|---|---|---|---|---|---|
| 400 | 24.21 | 24.82 | 31.63 | 33.21 | 44.42 | 32.08 (31.8%) | 28.77 (24.9%) | 33.00 (32.2%) |
| 2400 | 24.21 | 24.82 | 31.63 | 33.21 | 44.42 | 32.08 (31.8%) | 28.78 (25.0%) | 33.67 (32.4%) |

## 4 档 Ridge head val R²（正数越大越好）

| train pool | qwen-10 | qwen-50 | flux9b-10 | flux9b-50 |
|---|---|---|---|---|
| 400 | +0.225 | +0.234 | +0.340 | +0.352 |
| 2400 | +0.270 | +0.281 | +0.431 | +0.430 |
