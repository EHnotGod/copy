# §35–36 动作库选择的跨 Benchmark 共性

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

§35 A10 与 §36 A7 均已按清洗后的 UniGen pool2000/raw600 重跑。下面分别报告验证集选出的每域 one-SE 库和跨域共同 one-SE 库，所有差值相对本实验完整动作库。GM/AM 共享 prompt 来源。

### §35 A10

共同 one-SE 库：`qwen_tts, flux_10steps, boogu_10steps, boogu_50steps, flux4b_4steps`。[JSON](results/prompt_disjoint_v1/exp35_action_subset_crossbench.json)


| 评分集 | 完整库质量 | 每域 one-SE Δ (95% CI) | 共同 one-SE Δ (95% CI) |
|---|---:|---|---|
| GM | 0.381551 | +0.010069 [+0.001318, +0.018815] | +0.005639 [-0.000623, +0.011912] |
| AM | 0.799947 | -0.000457 [-0.004577, +0.003588] | +0.001616 [-0.000778, +0.004126] |
| UniGen | 0.858962 | +0.011797 [+0.002343, +0.021291] | +0.001242 [-0.003861, +0.006330] |

### §36 A7

共同 one-SE 库：`qwen_tts, flux_10steps, flux4b_4steps`。[JSON](results/prompt_disjoint_v1/exp36_common_a7_crossbench.json)

| 评分集 | 完整库质量 | 每域 one-SE Δ (95% CI) | 共同 one-SE Δ (95% CI) |
|---|---:|---|---|
| GM | 0.354778 | -0.000089 [-0.012776, +0.011831] | -0.003314 [-0.011893, +0.005565] |
| AM | 0.780616 | -0.003081 [-0.007210, +0.000936] | -0.003081 [-0.007096, +0.000752] |
| UniGen | 0.837171 | -0.012794 [-0.024169, -0.001901] | +0.000763 [-0.007334, +0.008403] |
| CoreBench | 0.540348 | -0.001269 [-0.006369, +0.003567] | +0.003543 [+0.000523, +0.006504] |
| WISE | 0.794788 | +0.000588 [-0.009807, +0.011054] | +0.001025 [-0.006783, +0.009072] |

修复后，UniGen 的 A10 每域 one-SE 有显著正收益，A7 每域 one-SE 则显著下降。旧版把两种动作空间一起概括成“UniGen 选库下降”已不适用。

A7 共同库在 CoreBench 显著提高；其余四项区间跨零。未检出显著损失不能替代非劣检验，因此仍不能宣称存在跨域无损的通用小库。该共同库使用各已知域 validation 选择，不属于未见域 LOBO。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 实验范围

本实验回答两个问题：各 benchmark 单独选择动作时会选什么，以及是否存在一个跨 benchmark 通用的小动作库。

- A10：GM、AM、UniGen，动作是 Qwen/Flux/Boogu × `{10,50,TTS}` + F4B4。
- 共同 A7：GM、AM、UniGen、CoreBench、WISE。WISE/CoreBench 没有 Boogu 输出，因此公平交集只能使用 Qwen/Flux × `{10,50,TTS}` + F4B4。
- GM/AM/UniGen：2400 pool 拆为 2000 Ridge train / 200 alpha-val / 200 library-val。
- CoreBench/WISE：400 pool 拆为 300 Ridge train / 50 alpha-val / 50 library-val，并额外用 seeds 40–49 做 10-split 稳定性复核。
- 所有非空可行子集全部枚举；动作库只根据 library-val 选择，held-out test 标签不参与选库。
- 主选择规则为 one-standard-error：在 validation 最优点的一个标准误以内选动作数最少的库。Router 始终是同一套 Ridge + budget=10 路由。

## A10：GM / AM / UniGen

| Benchmark | validation one-SE 库 | Test quality | 完整 A10 | Δ vs A10 | 95% CI |
|---|---|---:|---:|---:|---|
| GM | `QTTS+F10+B10` | **0.3916** | 0.3816 | **+0.0101** | `[+0.0013,+0.0188]` |
| AM | `F10+B10+B50+F4B4` | 0.7995 | 0.7999 | −0.0005 | `[−0.0046,+0.0036]` |
| UniGen | `B10+B50+F4B4` | 0.8999 | **0.9118** | **−0.0118** | `[−0.0198,−0.0044]` |

三者 validation 平均的 one-SE 库仍是 `QTTS+F10+B10`。它在 GM 提升 +0.0101、AM +0.0013，但在 UniGen 下降 −0.0222。并且不存在一个 A10 子集能同时落入 GM、AM、UniGen 各自的 one-SE 区间。

这说明 GM 的三动作库是真实的 benchmark-specific 结果，不能直接升级为全局默认库。UniGen 的动作分数较高、Prompt Ridge R² 很低，200 条 library-val 仍会把验证集上的偶然排序当成动作冗余。

## 五 Benchmark 的共同 A7

### Validation 选择

| Benchmark | one-SE 动作库 |
|---|---|
| GM | `QTTS+F10` |
| AM | `QTTS+F10+F4B4` |
| UniGen | `QTTS+F4B4` |
| CoreBench | `QTTS+F10` |
| WISE | `QTTS+F10` |
| 五 benchmark 平均 | `QTTS+F10` |
| 同时满足五者 one-SE | `QTTS+F10+F4B4` |

### Held-out test

| Benchmark | 完整 A7 | dataset one-SE | Δ | 五库稳健 `QTTS+F10+F4B4` | Δ vs A7 |
|---|---:|---:|---:|---:|---:|
| GM | 0.3548 | 0.3547 | −0.0001 | 0.3515 | −0.0033 |
| AM | 0.7806 | 0.7775 | −0.0031 | 0.7775 | −0.0031 |
| UniGen | **0.8846** | 0.8624 | **−0.0221** | 0.8763 | **−0.0083** |
| CoreBench | 0.5403 | 0.5391 | −0.0013 | **0.5439** | **+0.0035** |
| WISE | 0.7948 | 0.7954 | +0.0006 | 0.7958 | +0.0010 |

稳健三动作库只在 CoreBench 上显著超过完整 A7（CI `[+0.0005,+0.0065]`），在 WISE/GM/AM 与 A7 打平，在 UniGen 显著下降。因此 validation 上“同时在 one-SE 内”不等于 held-out 上真正无损；跨未知 benchmark 部署时，完整 A7 仍是更安全的默认值。

## 动作级共性信号

下表的 “one-SE 入选” 是五个 dataset-specific 库中的次数；Router LOO 为完整 A7 减去该动作后的 validation 质量变化，正数表示当前 Router 能兑现该动作。

| 动作 | one-SE 入选 | Router LOO 为正 | 平均 Router LOO | 结论 |
|---|---:|---:|---:|---|
| Q10 | 0/5 | 4/5 | +0.0023 | 有小边际，但与更有效的低成本动作联合冗余 |
| Q50 | 0/5 | 1/5 | −0.0002 | 稳定冗余 |
| **QTTS** | **5/5** | **5/5** | **+0.0171** | 最稳定的高质量 booster |
| **F10** | **4/5** | **5/5** | **+0.0076** | 最稳定的可负担 anchor |
| F50 | 0/5 | 0/5 | −0.0003 | 稳定冗余 |
| FTTS | 0/5 | 3/5 | −0.0007 | 成本太高，整体不能兑现 |
| F4B4 | 2/5 | 4/5 | +0.0021 | 有 Oracle 机会，但是否可学/是否入库依赖 benchmark 和 Router 信息 |

在完整 A10 的三个 benchmark 中，`B10` 是更强的共性动作：3/3 one-SE 入选、3/3 Router LOO 为正，平均 LOO +0.0100。`B50` 只在 AM/UniGen 有用，在 GM 会制造误选；`BTTS` 基本无用。

## WISE / CoreBench 小池稳定性

400 条 pool 下，单个 split 只有 50 条 library-val。10 个 split 的结果显示：

| 选择方式 | CoreBench Δ vs A7 | WISE Δ vs A7 |
|---|---:|---:|
| 每个 split 单独选 0.002 容差库 | **−0.0045** `[−0.0070,−0.0021]` | **−0.0054** `[−0.0102,−0.0008]` |
| 10 split 聚合 one-SE：`QTTS+F10` | −0.0024 `[−0.0070,+0.0022]` | +0.0000 `[−0.0097,+0.0097]` |

逐 split 精确选库显著过拟合；重复 split 聚合 + one-SE 能把损失缩到统计不可区分，但也没有证明小库优于完整库。WISE 的动作 R² 很低甚至为负，`QTTS+F10` 的价值更多来自稳定的全局成本—质量混合，不应解释成 Router 已学会了可靠的 prompt 级分类。

## 共性规律与建议

1. **动作库的骨架不是“每个模型留一个 50-step”，而是“便宜 anchor + 高质量 booster”。** 当前最稳的共享骨架是 `F10 + QTTS`；有 Boogu 时再加入 `B10`。
2. **同模型中间档最容易冗余。** Q50/F50 在五 benchmark 从未进入 one-SE 库，虽然 Oracle LOO 通常为正，但当前 Prompt Router 无法稳定兑现；动作相关性高、性价比又夹在 10-step 与 TTS 之间。
3. **Oracle 机会和 Router 可学习性必须分开。** F4B4 在五个 benchmark 的 Oracle LOO 全为正，但 Prompt Router 的实际价值不稳定。它适合 result-aware Router，而不是仅凭平均分直接入库。
4. **选库本身会过拟合。** 尤其 50–200 条 library-val、动作预测 R² 低时，“validation 最优子集”可能在 test 显著下降。应使用重复 split、one-SE、动作入选频率，并保留完整库 safety comparison。
5. **不存在当前证据支持的全局最优小库。** 已知 benchmark 可以使用各自验证过的库；未知/跨 benchmark 部署优先保留完整可用库，再用逐动作 gate 或 result-aware 信息抑制误选，而不是永久删除所有非核心动作。

实用决策：

- GenEval2 GM Prompt-only：`QTTS+F10+B10`。
- 有 Flux probe 图的 GenEval2：`QTTS+F10+B10+F4B4`，交给 Qwen3.5-2B result-aware Router。
- WISE：`QTTS+F10` 与完整 A7 持平，但只能称为压缩候选，不能称为提升。
- 跨未知 benchmark：保留完整共享 A7；`QTTS/F10` 作为必留核心，Q50/F50 是最优先移除候选，F4B4 由 result-aware gate 决定。

## 产物与数值审计

- A10 数值：[results/exp35_action_subset_crossbench.json](results/exp35_action_subset_crossbench.json)。
- 五 benchmark A7 与 WISE/CoreBench 10-split：[results/exp36_common_a7_crossbench.json](results/exp36_common_a7_crossbench.json)。
- 所有可行策略平均成本均不超过 10；所有 allocation 和为 1；JSON、embedding、prediction cache 均无 NaN/Inf。
- CoreBench/WISE 的 qwen/flux/F4B4 instruction 逐条取交集并与 Qwen3-Embedding 缓存重新对齐，pool/test 分别为 400/1080 与 400/1000。

</details>
