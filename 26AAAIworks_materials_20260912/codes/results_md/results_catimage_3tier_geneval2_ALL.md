# geneval2 · 3-tier router · delta vs score 对照

## 设置

每个模型 3 档 step 配置（10 / 50 / tts），预算见下表：

| model | cost 10 | cost 50 | cost tts | budget B |
|---|---:|---:|---:|---:|
| qwen | 10 | 50 | 100 | 50 |
| flux | 23.1 | 115.5 | 231 | 115.5 |

两种 predictor 目标（其他一切完全对齐：CLIP token feature、train/val split、seed、下游 sweep_tradeoff_curve）：

- **delta**：predictor 输出 (d1 = S_50 − S_10, d2 = S_tts − S_50)，路由分数 s0=0, s1=d1, s2=d1+d2
- **score**：predictor 直接输出 3 档绝对分数 [S_10, S_50, S_tts]

Predictors: KNN (CLIP-pooled) / TFMLP / LP-train / OC-train

test = raw 800（各模型自己的），train pool ∈ {400, 2400}，seeds = {0, 1, 2}。

---

## 一、决策收益（quality @ budget，0-1 分）

括号内为 argmax accuracy。粗体标注每 (model, size) 里 KNN/TFMLP 最强的 mode。

### qwen（cost=[10, 50, 100], B=50）

| train pool | mode | oracle@B | fixed-10 | fixed-50 | fixed-tts | LP | OC | KNN | TFMLP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 400  | delta | 0.4476 | 0.2421 | 0.2482 | 0.4482 | 0.3336 (44.5%) | 0.3287 (42.7%) | **0.3510 (51.1%)** | **0.3607 (54.3%)** |
| 400  | score | 0.4476 | 0.2421 | 0.2482 | 0.4482 | 0.3336 (44.5%) | 0.3287 (42.7%) | 0.3510 (51.1%) | 0.3521 (52.8%) |
| 2400 | delta | 0.4476 | 0.2421 | 0.2482 | 0.4482 | 0.3336 (44.5%) | 0.3299 (43.1%) | 0.3560 (52.3%) | **0.3705 (56.6%)** |
| 2400 | score | 0.4476 | 0.2421 | 0.2482 | 0.4482 | 0.3336 (44.5%) | 0.3299 (43.1%) | **0.3560 (52.3%)** | 0.3698 (57.1%) |

（qwen test 上 fixed-tts=0.4482 已经比 oracle@B=0.4476 还高——因为 tts 平均分最高，但它的 cost=100 超过预算 B=50。oracle@B 是在预算约束下的最优混合，它比 fixed-tts 低是因为要在预算内。）

### flux（cost=[23.1, 115.5, 231], B=115.5）

| train pool | mode | oracle@B | fixed-10 | fixed-50 | fixed-tts | LP | OC | KNN | TFMLP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 400  | delta | 0.4331 | 0.3163 | 0.3321 | 0.3721 | 0.3406 (37.9%) | 0.3394 (36.7%) | **0.3424 (37.3%)** | 0.3449 (37.9%) |
| 400  | score | 0.4331 | 0.3163 | 0.3321 | 0.3721 | 0.3406 (37.9%) | 0.3394 (36.7%) | 0.3424 (37.3%) | **0.3461 (37.6%)** |
| 2400 | delta | 0.4331 | 0.3163 | 0.3321 | 0.3721 | 0.3406 (37.9%) | 0.3395 (36.7%) | 0.3437 (36.7%) | 0.3456 (39.0%) |
| 2400 | score | 0.4331 | 0.3163 | 0.3321 | 0.3721 | 0.3406 (37.9%) | 0.3395 (36.7%) | **0.3437 (36.7%)** | **0.3487 (41.9%)** |

（flux 上 fixed-tts=0.3721 只是勉强低于 CATImage 最强，flux tts 相对 50 的边际收益很小。）

### 观察

- **KNN 两 mode 完全一致**：训 (d1,d2) 和训 (S10,S50,Stts) 在 KNN 平均后差一个常数偏移，argmax 不受影响。KNN 那列 delta / score 数字完全相同（精确到 0.0001）。
- **TFMLP 两 mode 差距很小**（≤ 0.009 quality）：
  - qwen 400：delta 0.3607 vs score 0.3521，delta 微胜；
  - qwen 2400：delta 0.3705 vs score 0.3698 基本持平；
  - flux 400：score 0.3461 vs delta 0.3449 微胜；
  - flux 2400：score 0.3487 vs delta 0.3456 略胜。
- **加数据（400→2400）对 TFMLP 的提升**：qwen delta +0.010 / score +0.018；flux delta +0.001 / score +0.003。
- **flux 上 CATImage 输给 fixed-tts baseline**（最强 0.3487 vs fixed-tts 0.3721）：flux 预算 115.5 允许全部走 50 步（cost 115.5 恰好卡上），如果整体强行走 tts 会超预算，但 fixed-tts 那一档本身就没做预算约束。真正跟 CATImage 比的应该是 oracle@B=0.4331。
- **qwen 上 CATImage 也没到 oracle@B=0.4476**：最强 0.3705 也差 0.077 quality。

---

## 二、Delta 预测本身的质量（Spearman ρ / MAE / WSA）

d1 = S_50 − S_10（step 差异微弱），d2 = S_tts − S_50（tts 系统性偏高）。score 模式下的 d1/d2 是由预测的绝对分数相减得到。

### qwen

| train pool | mode | predictor | Sp d1 | Sp d2 | MAE d1 | MAE d2 | WSA d1 | WSA d2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 400  | delta | KNN   | +0.013 | +0.168 | 0.086 | 0.196 | 0.525 | 0.905 |
| 400  | delta | TFMLP | +0.008 | +0.262 | 0.089 | 0.192 | 0.528 | 0.905 |
| 400  | score | KNN   | +0.013 | +0.168 | 0.086 | 0.196 | 0.525 | 0.905 |
| 400  | score | TFMLP | +0.017 | +0.165 | 0.087 | 0.202 | 0.509 | 0.905 |
| 2400 | delta | KNN   | −0.037 | +0.185 | 0.086 | 0.195 | 0.507 | 0.905 |
| 2400 | delta | TFMLP | −0.011 | +0.297 | 0.088 | 0.183 | 0.521 | 0.900 |
| 2400 | score | KNN   | −0.037 | +0.185 | 0.086 | 0.195 | 0.507 | 0.905 |
| 2400 | score | TFMLP | +0.005 | +0.317 | 0.086 | 0.185 | 0.518 | 0.905 |

### flux

| train pool | mode | predictor | Sp d1 | Sp d2 | MAE d1 | MAE d2 | WSA d1 | WSA d2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 400  | delta | KNN   | +0.087 | +0.022 | 0.103 | 0.126 | 0.576 | 0.629 |
| 400  | delta | TFMLP | +0.115 | +0.007 | 0.104 | 0.128 | 0.578 | 0.660 |
| 400  | score | KNN   | +0.087 | +0.022 | 0.103 | 0.126 | 0.576 | 0.629 |
| 400  | score | TFMLP | +0.087 | +0.034 | 0.119 | 0.141 | 0.556 | 0.586 |
| 2400 | delta | KNN   | +0.057 | +0.009 | 0.104 | 0.128 | 0.547 | 0.645 |
| 2400 | delta | TFMLP | +0.084 | −0.001 | 0.104 | 0.133 | 0.561 | 0.650 |
| 2400 | score | KNN   | +0.057 | +0.009 | 0.104 | 0.128 | 0.547 | 0.645 |
| 2400 | score | TFMLP | +0.087 | +0.036 | 0.103 | 0.132 | 0.565 | 0.672 |

### 观察

- **Sp d1 全部在 −0.04 到 +0.12 之间**：50-vs-10 的相对增益几乎完全学不到。跟 v2-delta 实验的结论一致。
- **qwen Sp d2 明显更高**（+0.16 到 +0.32），且 WSA d2 = 0.90——但这是**"永远猜正号"就能刷到的水位**：qwen tts 平均 0.4482 比 50 平均 0.2482 高 0.20，正号占绝大多数样本。
- **flux Sp d2 几乎 0**：flux tts 平均 0.3721 只比 50 平均 0.3321 高 0.04，正负样本对半，猜不出来。
- **delta vs score 在 d2 上的差别**：qwen TFMLP 2400 上 delta 0.297 vs score 0.317，score 反而略高一点（因为 score mode 训绝对分数、能从 μ_i 信号里蹭到一点边角）。d1 上两者都趴地。

---

## 三、核心结论

1. **决策收益：delta 和 score 几乎打平**，差距 ≤ 0.009 quality。KNN 两 mode 数学等价，TFMLP 两 mode 也无本质差别。
2. **换 predictor 目标解决不了根本问题**：不管学绝对分数还是学 delta，qwen 路由收益卡在 0.35–0.37（离 oracle@B=0.4476 差 0.07+），flux 卡在 0.343–0.349（离 oracle@B=0.4331 差 0.08+）。跟"每个 prompt 加多少 step"这个真正决策依赖的信号学不到直接相关。
3. **加数据（400→2400）对 TFMLP 的提升有限**：qwen delta +0.010 / score +0.018，flux delta +0.001 / score +0.003。跟 v2-delta 结论一致——加数据主要在学 μ_i，不在学 τ_{i,t}。
4. **qwen 上 WSA d2 = 0.90 是常数正号 baseline 的假象**：不代表 predictor 学到了 tts 的相对收益。真正衡量方向学到多少的是 Sp d2（qwen +0.30、flux +0.03），在 flux 上完全是 0。
5. **flux 上 CATImage 甚至输给 fixed-tts**（0.3487 vs 0.3721）：因为 fixed-tts 不受预算约束（它假设所有 case 都可以花到 231 cost）。真正的对手是 oracle@B=0.4331，双方在预算约束下 CATImage 都够不到 oracle。

---

## 四、产物

- 脚本：`run_catimage_3tier_delta_geneval2.py`（支持 `--modes delta score`）
- 图：`catimage_plots_3tier/deferral_geneval2_{model}_{mode}_train{size}.png`（8 张 deferral 曲线）
- 每样本预测：`catimage_preds_3tier/preds_geneval2_{model}_{mode}_train{size}.json`
- log：`logs_3tier/{model}_{size}.log`
- 4 份 partial md：`results_md/results_3tier_{model}_{size}.md`（用的是 *100 版；本汇总 md 已换算成 0-1 分）
- 汇总 md：本文档
