# geneval2 · 4 档(含 qwen-tts) · Qwen3-emb-8b + Ridge · train 400 vs 2400

4 档：qwen-10 / qwen-50 / flux9b-10 / qwen-tts

成本：10 / 50 / 23.1 / 100，预算 B=50，seeds=0/1/2

Backbone: Qwen3-Embedding-8B (mean-pool)，Predictor: 4 个独立 Ridge head

测试集：geneval2 raw 800


| train pool | 10-qwen | 50-qwen | 10-flux9b | tts-qwen | oracle@50 | Q3+Ridge@50 |
|---|---|---|---|---|---|---|
| 400 | 24.21 | 24.82 | 31.63 | 44.82 | 50.30 | 37.76 (35.3%) |
| 2400 | 24.21 | 24.82 | 31.63 | 44.82 | 50.30 | 38.90 (38.5%) |

## 4 档 head val R²

| train pool | qwen-10 | qwen-50 | flux9b-10 | qwen-tts |
|---|---|---|---|---|
| 400 | +0.225 | +0.234 | +0.340 | +0.333 |
| 2400 | +0.270 | +0.281 | +0.431 | +0.475 |
