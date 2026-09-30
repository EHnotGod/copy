# geneval2 · 4-tier router · train 400 vs 2400

4 档：qwen-10 / qwen-50 / flux9b-10 / flux9b-50

成本：10 / 50 / 23.1 / 115.5，预算 B=50，seeds=0/1/2

测试集：geneval2 raw 800（qwen+flux 对齐）


| train pool | 10-qwen | 50-qwen | 10-flux9b | 50-flux9b | oracle@50 | LP | OC | KNN@50 | TFMLP@50 |
|---|---|---|---|---|---|---|---|---|---|
| 400 | 24.21 | 24.82 | 31.63 | 33.21 | 44.42 | 32.08 (31.8%) | 28.77 (24.9%) | 32.65 (31.9%) | 31.77 (30.6%) |
| 2400 | 24.21 | 24.82 | 31.63 | 33.21 | 44.42 | 32.08 (31.8%) | 28.78 (25.0%) | 32.69 (32.9%) | 32.60 (31.5%) |

## 附加方法：Qwen3-Embedding-8B + Ridge（每档一个 Ridge head）

| train pool | oracle@50 | LP | OC | Q3+Ridge@50 |
|---|---|---|---|---|
| 400 | 44.42 | 32.08 (31.8%) | 28.77 (24.9%) | **33.00 (32.2%)** |
| 2400 | 44.42 | 32.08 (31.8%) | 28.78 (25.0%) | **33.67 (32.4%)** |

Ridge head val R²（越大越准）：

| train pool | qwen-10 | qwen-50 | flux9b-10 | flux9b-50 |
|---|---|---|---|---|
| 400 | +0.225 | +0.234 | +0.340 | +0.352 |
| 2400 | +0.270 | +0.281 | +0.431 | +0.430 |

跟 CLIP-KNN / TFMLP 对比：
- Q3+Ridge 在 400 pool 上 33.00 vs KNN 32.65 / TFMLP 31.77，涨 ~0.35–1.2
- 2400 pool 上 33.67 vs KNN 32.69 / TFMLP 32.60，涨 ~1.0
- Q3+Ridge 也是唯一一个从 400→2400 有明显涨幅的 predictor（33.00→33.67，+0.67），跟 exp1 里 v3→v4 加数据涨 R² 的观察一致
- 但仍然低于 fixed flux9b-50 = 33.21（400 pool 时）——4 档下 predictor router 拿不到明显收益的结论没变

脚本：`run_catimage_4tier_qwen3.py`，图：`catimage_plots_4tier_qwen3/`

