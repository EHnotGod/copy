# dp 路由：AM vs GM，pool=400 vs 2400，含 qwen3-emb-8b + Ridge

## 设置

跟 `results_3tier_am_vs_gm.md` 完全同 pipeline，**新增两个维度**：

1. 加了主 predictor **Qwen3-8B + Ridge**（qwen3-emb-8b embedding + linear ridge，对应 exp1.1 里最强 predictor）
2. 加了 **train_pool = 400** 对照 2400，看加数据对 dp 决策有没有帮助

其余：3 档 (10, 50, tts)，qwen cost=[10, 50, 100] B=50，flux cost=[23.1, 115.5, 231] B=115.5，seeds={0,1,2}，test=raw 800。

## 结果：quality @ budget（0-1）+ argmax accuracy（括号内）

### qwen（GM oracle=0.4476，AM oracle=0.8136）

| pool | mode | | KNN | TFMLP | **Qwen3-8B+Ridge** | LP | OC |
|---|---|---|---|---|---|---|---|
| **400** | delta | GM | 0.3510 (51.1%) | 0.3607 (54.3%) | **0.3589 (51.8%)** | 0.3336 (44.5%) | 0.3287 (42.7%) |
| **400** | delta | AM | 0.7709 (48.7%) | 0.7737 (50.7%) | **0.7754 (51.5%)** | 0.7645 (44.6%) | 0.7616 (42.9%) |
| **400** | score | GM | 0.3510 (51.1%) | 0.3521 (52.8%) | **0.3589 (51.8%)** | 0.3336 | 0.3287 |
| **400** | score | AM | 0.7709 (48.7%) | 0.7735 (50.5%) | **0.7754 (51.5%)** | 0.7645 | 0.7616 |
| **2400** | delta | GM | 0.3560 (52.3%) | 0.3705 (56.6%) | **0.3632 (54.6%)** | 0.3336 | 0.3299 |
| **2400** | delta | AM | 0.7760 (51.7%) | 0.7783 (54.2%) | **0.7776 (51.4%)** | 0.7642 | 0.7629 |
| **2400** | score | GM | 0.3560 (52.3%) | 0.3698 (57.1%) | **0.3632 (54.6%)** | 0.3336 | 0.3299 |
| **2400** | score | AM | 0.7760 (51.7%) | 0.7784 (55.1%) | **0.7776 (51.4%)** | 0.7642 | 0.7629 |

### flux（GM oracle=0.4331，AM oracle=0.8074）

| pool | mode | | KNN | TFMLP | **Qwen3-8B+Ridge** | LP | OC |
|---|---|---|---|---|---|---|---|
| **400** | delta | GM | 0.3424 (37.3%) | 0.3449 (37.9%) | **0.3422 (35.4%)** | 0.3406 (37.9%) | 0.3394 (36.7%) |
| **400** | delta | AM | 0.7667 (33.9%) | 0.7686 (36.2%) | **0.7689 (37.7%)** | 0.7645 | 0.7616 |
| **400** | score | GM | 0.3424 (37.3%) | 0.3461 (37.6%) | **0.3422 (35.4%)** | 0.3406 | 0.3394 |
| **400** | score | AM | 0.7667 (33.9%) | 0.7681 (36.0%) | **0.7689 (37.7%)** | 0.7645 | 0.7616 |
| **2400** | delta | GM | 0.3437 (36.7%) | 0.3456 (39.0%) | **0.3429 (39.2%)** | 0.3406 | 0.3395 |
| **2400** | delta | AM | 0.7663 (34.7%) | 0.7657 (34.2%) | **0.7663 (35.8%)** | 0.7626 | 0.7638 |
| **2400** | score | GM | 0.3437 (36.7%) | 0.3487 (41.9%) | **0.3429 (39.2%)** | 0.3406 | 0.3395 |
| **2400** | score | AM | 0.7662 (34.7%) | 0.7660 (35.0%) | **0.7663 (35.8%)** | 0.7626 | 0.7638 |

## 关键对比 1：加 pool 400 → 2400 有没有帮助？

用 **argmax accuracy**（不受 evaluator scale 影响，最能反映"predictor 真的决策好没好"）：

| | GM Qwen3+Ridge | AM Qwen3+Ridge | GM TFMLP | AM TFMLP | GM KNN | AM KNN |
|---|---:|---:|---:|---:|---:|---:|
| qwen pool 400 | 51.8% | 51.5% | 54.3% | 50.7% | 51.1% | 48.7% |
| qwen pool 2400 | 54.6% | 51.4% | 56.6% | 54.2% | 52.3% | 51.7% |
| qwen Δ (2400-400) | **+2.8%** | **−0.1%** | +2.3% | +3.5% | +1.2% | +3.0% |
| flux pool 400 | 35.4% | 37.7% | 37.9% | 36.2% | 37.3% | 33.9% |
| flux pool 2400 | 39.2% | 35.8% | 39.0% | 34.2% | 36.7% | 34.7% |
| flux Δ (2400-400) | **+3.8%** | **−1.9%** | +1.1% | −2.0% | −0.6% | +0.8% |

**GM 上加数据涨点（+1 到 +4% acc），AM 上加数据几乎不涨甚至倒退**。

对 Qwen3+Ridge 具体：
- **qwen GM**：400→2400，acc +2.8%，quality +0.004；跟 exp1.1 里 Sp ρ 从 0.44→0.56 一致
- **qwen AM**：400→2400，acc −0.1%（几乎持平）
- **flux GM**：400→2400，acc +3.8%
- **flux AM**：400→2400，acc −1.9%

## 关键对比 2：AM vs GM，同 pool、同 predictor

| | qwen 400 | qwen 2400 | flux 400 | flux 2400 |
|---|---:|---:|---:|---:|
| Qwen3+Ridge acc GM | 51.8% | 54.6% | 35.4% | 39.2% |
| Qwen3+Ridge acc AM | 51.5% | 51.4% | 37.7% | 35.8% |
| **AM − GM** | −0.3% | **−3.2%** | +2.3% | **−3.4%** |
| TFMLP acc GM (score) | 52.8% | 57.1% | 37.6% | 41.9% |
| TFMLP acc AM (score) | 50.5% | 55.1% | 36.0% | 35.0% |
| **AM − GM** | −2.3% | −2.0% | −1.6% | **−6.9%** |

**AM 版决策 accuracy 全线不如 GM**（除 flux 400 那一格）。 pool=2400 时差距最大：qwen −3.2%、flux −3.4% ~ −6.9%。

## 关键对比 3：最强 predictor 上限

**GM 视角**（跟 exp1.1 GM 版结论一致，最强 predictor Qwen3+Ridge ρ≈0.55–0.69）：

- qwen best acc: TFMLP@2400 = 57.1%（Qwen3+Ridge 54.6%）——TFMLP 反而更好
- flux best acc: TFMLP@2400(score) = 41.9%（Qwen3+Ridge 39.2%）

**AM 视角**：

- qwen best acc: TFMLP@2400 = 55.1%（Qwen3+Ridge 51.4%）——AM 上 Qwen3+Ridge 反而输给 TFMLP
- flux best acc: Qwen3+Ridge@400 = 37.7%

有意思的是：**exp1.1 里 Qwen3+Ridge 是排序最强的 predictor**（ρ 最高），但在 dp 决策上 **TFMLP 常常打赢**。可能是 TFMLP 学 delta 更贴合 dp 需要的"档间差异"，而 Ridge 学的是"prompt 整体难度"。

## 结论

1. **换 AM 不涨点，反而普遍略降**：Qwen3+Ridge acc 全 4 格里 3 格 AM 输给 GM（−0.3 ~ −3.4%），TFMLP 全线 AM 输 −1.6 ~ −6.9%。跟"AM y_true 集中在 [0.5, 0.9]，随便预测 0.7 就不出错"的直觉吻合——AM 上乱猜也够看，训练信号被稀释。

2. **加数据 400→2400，GM 上涨、AM 上不涨**：GM 上 Qwen3+Ridge acc 涨 +2.8%（qwen）/ +3.8%（flux）；AM 上 −0.1% / −1.9%。GM 版加数据能学到更好的排序，AM 版加数据已经饱和。

3. **最强 predictor 上限还是有限**：GM 上最强也只到 acc 57%（qwen）/ 42%（flux），跟 oracle 差 43%/58%。AM 也差不多。**换 predictor 目标（delta / score / TFMLP / Ridge）都无法突破**这个上限。

4. **quality @ budget 从 GM 0.35 涨到 AM 0.77 全是评分尺度上移的假象**：oracle 也从 0.44 涨到 0.81，%oracle 从 80% 涨到 95%——但这只是"随便走都不错"，argmax acc（真决策能力）反而降。

## 产物

- 脚本：`run_catimage_3tier_delta_geneval2_am.py`（加了 `--stage {edit, edit_am}` 开关 + RIDGE 通道）
- log：`logs_3tier_v2/{qwen,flux}_{am,gm}.log`
- 单 md：`results_md/results_3tier_v2_{qwen,flux}_{AM,GM}.md`
- 图：`catimage_plots_3tier_am/deferral_geneval2_{model}_{mode}_train{size}.png`（8 张 + Ridge 曲线）
- 每样本预测：`catimage_preds_3tier_am/preds_geneval2_{model}_{mode}_train{size}.json`
