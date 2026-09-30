# §23 9-action vs 10-action（加入 flux4b 蒸馏动作）

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

Ridge 部分已在 CPU 重跑；A9/A10 同 prompt 划分，路由选择不读取测试标签。

| 数据集 | train / val / test | A9 Ridge@10 | A10 Ridge@10 | 结果 |
|---|---|---:|---:|---|
| GM | 2200 / 200 / 800 | 0.379654 | 0.378813 | [JSON](results/prompt_disjoint_v1/exp23_10action_distill_gm.json) |
| AM | 2200 / 200 / 800 | 0.801569 | 0.801576 | [JSON](results/prompt_disjoint_v1/exp23_10action_distill_am.json) |
| UniGen | 1800 / 200 / 600 | 0.853054 | 0.852317 | [JSON](results/prompt_disjoint_v1/exp23_10action_distill_unigen.json) |

旧 TFMLP/KNN 未混入新结果；`legacy_exp18_audit` 仅保留旧指标的诊断，不作为修复后路由收益。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 设置

- A9：qwen / flux / boogu × {10steps, 50steps, TTS}。
- A10：A9 + flux4b_4steps（cost=0.455）。
- train/val 固定为 2200/200（seed=42）；GM/AM test=800，UniGen test=600。
- Predictor：固定使用 Ridge；A9/A10 使用完全相同的 prompt、划分和特征。
- 本实验研究动作池变化，不再重复比较 KNN/TFMLP。
- 所有混合基线与路由曲线均按 E[c]≤B 计算；LP 用 train action mean 选择、test 评估。

## budget=10 主结果

| dataset | pool | Oracle | LP | Ridge | Ridge−LP | distill share (Ridge) |
|---|---|---:|---:|---:|---:|---:|
| gm | A9 | 0.5489 | 0.3755 | 0.3823 | +0.0068 | 0.0% |
| gm | A10 | 0.5684 | 0.3755 | 0.3806 | +0.0052 | 9.9% |
| am | A9 | 0.8583 | 0.7919 | 0.8017 | +0.0098 | 0.0% |
| am | A10 | 0.8665 | 0.7919 | 0.8017 | +0.0098 | 15.8% |
| unigen | A9 | 0.9626 | 0.8618 | 0.9104 | +0.0486 | 0.0% |
| unigen | A10 | 0.9693 | 0.8618 | 0.9158 | +0.0541 | 17.4% |

## 蒸馏动作增量分解（A10−A9）

其中 learned gain = ΔRouter − ΔLP；正值表示新增动作增加了超越静态 LP 的实例级收益。

| dataset | budget | ΔOracle | ΔLP | ΔRidge | learned Ridge | distill share (Ridge) |
|---|---:|---:|---:|---:|---:|---:|
| gm | 2 | +0.2069 | +0.0515 | +0.0661 | +0.0146 | 44.8% |
| gm | 5 | +0.0300 | +0.0000 | -0.0035 | -0.0035 | 15.2% |
| gm | 10 | +0.0195 | +0.0000 | -0.0016 | -0.0016 | 9.9% |
| gm | 20 | +0.0103 | +0.0000 | -0.0008 | -0.0008 | 4.8% |
| gm | 30 | +0.0058 | +0.0000 | -0.0020 | -0.0020 | 2.7% |
| gm | 45 | +0.0058 | +0.0000 | -0.0021 | -0.0021 | 2.0% |
| gm | 65 | +0.0058 | +0.0000 | -0.0021 | -0.0021 | 2.0% |
| am | 2 | +0.1028 | +0.0324 | +0.0496 | +0.0172 | 46.9% |
| am | 5 | +0.0131 | +0.0000 | +0.0004 | +0.0004 | 20.8% |
| am | 10 | +0.0082 | +0.0000 | -0.0000 | -0.0000 | 15.8% |
| am | 20 | +0.0042 | +0.0000 | -0.0012 | -0.0012 | 9.4% |
| am | 30 | +0.0023 | +0.0000 | -0.0026 | -0.0026 | 5.7% |
| am | 45 | +0.0023 | +0.0000 | -0.0025 | -0.0025 | 5.4% |
| am | 65 | +0.0023 | +0.0000 | -0.0025 | -0.0025 | 5.4% |
| unigen | 2 | +0.1702 | +0.0585 | +0.1165 | +0.0580 | 52.6% |
| unigen | 5 | +0.0169 | +0.0000 | +0.0113 | +0.0113 | 25.0% |
| unigen | 10 | +0.0067 | +0.0000 | +0.0055 | +0.0055 | 17.4% |
| unigen | 20 | +0.0013 | +0.0000 | +0.0006 | +0.0006 | 8.9% |
| unigen | 30 | +0.0013 | +0.0000 | +0.0003 | +0.0003 | 5.4% |
| unigen | 45 | +0.0013 | +0.0000 | -0.0001 | -0.0001 | 5.0% |
| unigen | 65 | +0.0013 | +0.0000 | -0.0001 | -0.0001 | 5.0% |

## 新增蒸馏动作的可预测性

| dataset | Ridge R²(all/mid) |
|---|---:|
| gm | +0.357/-0.025 |
| am | +0.311/+0.108 |
| unigen | +0.518/+0.090 |

## 结论

1. **蒸馏动作首先改变的是低预算可行域。** budget=2 时，A9 只能使用最低成本为 2.0 的 qwen_10steps；A10 可用 cost=0.455 的 flux4b，并把剩余预算与更贵动作混合。因此三数据集 Oracle 分别增加 +0.2069 / +0.1028 / +0.1702，Ridge 分别增加 +0.0661 / +0.0496 / +0.1165。
2. **在主预算 budget=10，系统机会增加，但学习性收益不普遍增加。** GM / AM / UniGen 的 ΔOracle 分别为 +0.0195 / +0.0082 / +0.0067；对应 ΔRidge 为 −0.0016 / −0.0000 / +0.0055。GM 反而轻微退步，AM 持平，只有 UniGen 学到了新增动作的大部分收益。
3. **budget≥5 时 ΔLP 全部为 0。** flux4b 的全局平均质量没有进入 train-mean LP 前沿；它带来的 Oracle 增益来自 prompt-specific 互补性，而不是一个全局更优的固定动作。但 GM/AM 的 Ridge 无法利用这种互补性。
4. **Ridge 系统性低估了蒸馏动作的适用范围。** budget=10 时 Oracle 对 flux4b 的分配比例为 16.9% / 19.8% / 43.3%，Ridge 只有 9.9% / 15.8% / 17.4%；UniGen 的可利用空间尤其大。
5. **新增动作的 R²(all) 仍受分布两端贡献。** Ridge R²(all) 为 0.311~0.518，但 R²(mid) 只有 −0.025~0.108，与前面绝对预测审计结论一致。

**总判断：加入蒸馏动作显著增强系统的低预算 Pareto 前沿，但不会普遍增强路由的可学习性；学习性收益具有明显的数据集依赖，在 UniGen 成立，在 GM/AM 基本不成立。**

## 旧 §18 复现审计

旧 §18 的 Ridge embedding 直接按原文件行序切片，而 action label 使用排序后的 instruction；本实验按 instruction 显式对齐。下表用旧 §18 路由插值口径比较 budget=10。

| dataset | method | old §18 | aligned rerun | difference |
|---|---|---:|---:|---:|
| gm | oracle | 0.5489 | 0.5489 | +0.0000 |
| gm | ridge | 0.3507 | 0.3797 | +0.0289 |
| am | oracle | 0.8583 | 0.8583 | +0.0000 |
| am | ridge | 0.7806 | 0.8016 | +0.0210 |
| unigen | oracle | 0.9626 | 0.9626 | +0.0000 |
| unigen | ridge | 0.8300 | 0.9103 | +0.0804 |

## 产物

- 合并结果：[results/exp23_10action_distill.json](results/exp23_10action_distill.json)
- 分数据集结果与预测缓存：`results/exp23_10action_distill_{dataset}.json` / `exp23_10action_distill_preds_{dataset}.npz`
- 图：[figs/exp23_10action_distill.png](figs/exp23_10action_distill.png)

</details>
