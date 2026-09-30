# dp 路由：AM vs GM 对照（3-tier, train=2400）

## 设置

跟 `results_catimage_3tier_geneval2_ALL.md`（GM 版）**完全同**：
- 3 档 (10, 50, tts)
- qwen cost=[10, 50, 100] B=50；flux cost=[23.1, 115.5, 231] B=115.5
- Predictor：KNN / TFMLP / LP-train / OC-train
- train pool = 2400（原 400 + 新扩 2000）
- seeds = {0, 1, 2}
- test = raw 800

**唯一差异**：evaluator 端 sub-question 聚合方式 **GM ↔ AM**。图像本身没变、predictor pipeline 没变。

## 决策收益（quality @ budget，0-1）

**qwen（B=50, GM oracle@B=0.4476, AM oracle@B=0.8136）**

| mode | metric | GM | AM | Δ |
|---|---|---:|---:|---:|
| **KNN(delta)** | quality | 0.3560 | 0.7760 | +0.4200 |
| KNN(delta) | %oracle | 79.6% | 95.4% | +15.8% |
| KNN(delta) | acc | 52.3% | 51.7% | −0.6% |
| **KNN(score)** | quality | 0.3560 | 0.7760 | +0.4200 |
| KNN(score) | %oracle | 79.6% | 95.4% | +15.8% |
| KNN(score) | acc | 52.3% | 51.7% | −0.6% |
| **TFMLP(delta)** | quality | 0.3705 | **0.7783** | +0.4078 |
| TFMLP(delta) | %oracle | 82.8% | **95.7%** | +12.9% |
| TFMLP(delta) | acc | 56.6% | 54.2% | −2.4% |
| **TFMLP(score)** | quality | 0.3698 | **0.7784** | +0.4086 |
| TFMLP(score) | %oracle | 82.6% | **95.7%** | +13.1% |
| TFMLP(score) | acc | 57.1% | 55.1% | −2.0% |
| LP-train | quality | 0.3336 | 0.7642 | +0.4306 |
| LP-train | %oracle | 74.5% | 93.9% | +19.4% |
| OC-train | quality | 0.3299 | 0.7629 | +0.4330 |
| OC-train | %oracle | 73.7% | 93.8% | +20.1% |

**flux（B=115.5, GM oracle@B=0.4331, AM oracle@B=0.8074）**

| mode | metric | GM | AM | Δ |
|---|---|---:|---:|---:|
| **KNN(delta)** | quality | 0.3437 | 0.7663 | +0.4226 |
| KNN(delta) | %oracle | 79.4% | 94.9% | +15.5% |
| KNN(delta) | acc | 36.7% | 34.7% | −2.0% |
| **KNN(score)** | quality | 0.3437 | 0.7662 | +0.4225 |
| KNN(score) | %oracle | 79.4% | 94.9% | +15.5% |
| KNN(score) | acc | 36.7% | 34.7% | −2.0% |
| **TFMLP(delta)** | quality | 0.3456 | 0.7657 | +0.4201 |
| TFMLP(delta) | %oracle | 79.8% | 94.8% | +15.0% |
| TFMLP(delta) | acc | 39.0% | 34.2% | −4.8% |
| **TFMLP(score)** | quality | 0.3487 | **0.7660** | +0.4173 |
| TFMLP(score) | %oracle | 80.5% | 94.9% | +14.4% |
| TFMLP(score) | acc | 41.9% | 35.0% | −6.9% |
| LP-train | quality | 0.3406 | 0.7626 | +0.4220 |
| LP-train | %oracle | 78.7% | 94.4% | +15.7% |
| OC-train | quality | 0.3395 | 0.7638 | +0.4243 |
| OC-train | %oracle | 78.4% | 94.6% | +16.2% |

## 关键观察

1. **AM 版 quality 全线大涨 +0.42 左右**——但这是**评分尺度整体上移的假象**：oracle@B 也涨了 0.37（qwen 0.45→0.81）、fixed baseline 涨了 0.4+，predictor 只是跟着"水涨船高"。**不是 predictor 变强了**。

2. **%oracle（用 oracle 归一化后的相对指标）AM 版反而更高**：qwen KNN 79.6% → 95.4%，flux TFMLP(score) 80.5% → 94.9%。但这也不完全是"predictor 更好"——AM 分数分布更集中（[0.5, 1.0]），**oracle 和随便乱走的差距本身就变小了**，%oracle 天然被压缩到 95% 附近。

3. **argmax accuracy 反而下降**：qwen −0.6% 到 −2.4%，flux −2.0% 到 −6.9%。argmax acc 是"predictor 挑对最佳档的比例"，这个指标不受 quality scale 影响，AM 上更差意味着**predictor 的细粒度决策能力其实变弱了**（跟 exp1.1_am 的 flux Sp ρ 下降一致）。

4. **KNN delta / score 数字**完全相同（跟 GM 版一样）——KNN 平均出的绝对分数和 delta 只差常数偏移，argmax 不受影响。

5. **AM 让 predictor "看起来好"，但真信号变弱**：
   - quality 绝对分数：AM 高（+0.42）——尺度效应
   - %oracle 相对分数：AM 高（+15%）——oracle-baseline gap 缩小
   - argmax accuracy：AM **低**（−1% ~ −7%）——决策真质量下降
   - Sp d1（step-wise 相对增益的排序）：qwen 上从 GM 0.01 → AM −0.01/−0.02（依旧近零）；flux 上 GM 0.08 → AM 0.04（也变弱）

6. **换 AM 评分并不真正提升 dp 决策质量**。看似 %oracle 涨了 15%，实际 argmax acc 下降 2-7%，说明 AM 上 predictor 更容易被"猜均值"骗过（这个跟你之前的直觉一致：AM y_true 在 [0.5, 0.9]，随便预测个 0.7 也很难错太多）。

## 一句话结论

**dp 分数从 0.34 → 0.78 是 evaluator 尺度上移，不是 predictor 变好**。用不受 scale 影响的 argmax accuracy 看，AM 版比 GM 版反而**下降 1–7%**——**AM 评分对 dp 决策没有实质帮助**，甚至可能更差。

## 产物

- 脚本：`run_catimage_3tier_delta_geneval2_am.py`（复制自 GM 版，仅换数据源到 `edit_am/`）
- 图：`catimage_plots_3tier_am/deferral_geneval2_{model}_{mode}_train2400.png`（4 张）
- 每样本预测：`catimage_preds_3tier_am/preds_geneval2_{model}_{mode}_train2400.json`
- log：`logs_3tier_am/{model}.log`
- 单模型 md：`results_md/results_3tier_am_{qwen,flux}.md`
- 汇总（本文档）：`results_md/results_3tier_am_vs_gm.md`
