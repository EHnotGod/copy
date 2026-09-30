# §37 跨 Benchmark 新动作快速入库判断

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

§37 已完成 CPU 重算，当前结果为 [JSON](results/prompt_disjoint_v1/exp37_action_admission_crossbench.json)。UniGen 使用 pool2000/raw600，所有新预测、基线与汇总来自同一数据版本。历史正文中的旧 UniGen 分数不再作为当前结论。GM/AM 是同一 prompt 来源的两种评分。

沿用主规则：单次 k=50、`router_mean`（OOF Router 边际增益大于零），评价后续充分训练的 investment 目标。三个 split 作为重复试验分别决策。


| 评分集 | 接受精度 | 正收益召回 | 平均决策 regret |
|---|---:|---:|---:|
| GM | 0.888 | 0.588 | 0.002806 |
| AM | 0.897 | 0.573 | 0.001871 |
| UniGen | 0.801 | 0.516 | 0.003614 |
| CoreBench | 0.891 | 0.790 | 0.000785 |
| WISE | 0.773 | 0.653 | 0.001729 |

精度的分母是被接受的候选，召回的分母是真实正收益候选；regret 是相对知道真实边际收益的最优接受/拒绝决策损失。动作价值依赖当前库。这里不是从空库发现最优动作库，也不把三次筛选平均当作只花一次 k 条标签。CoreBench/WISE 的严格分离筛选最大为 50 条。

该表判断的是“值得补齐标签后训练”，不代表只训练50条就能达到同样的部署效果；直接低标签量部署对应 JSON 的 `lowshot_delta`，两种目标需分别引用。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 研究问题

本实验不寻找一个跨 benchmark 固定动作库，而是验证一套可迁移的 **action-admission
procedure**：给定目标 benchmark、当前可行动作库 `S` 和候选动作 `a`，只生成少量候选
动作结果，能否判断 `S → S∪{a}` 是否值得？

动作价值始终是相对当前库的条件边际量，不把动作压缩成脱离上下文的全局分数。

## 严格协议

- Benchmark：GenEval2 GM、GenEval2 AM、UniGen、CoreBench、WISE。
- 公共动作空间 A7：Qwen/Flux × `{10,50,TTS}` + `F4B4`；budget=10。
- 枚举所有可行的“当前库 + 候选动作”组合，共 **404 个条件上下文**；每个设置使用
  split seeds `42/43/44` 重复。
- 候选标签数：GM/AM/UniGen 为 `k∈{50,100,200}`；CoreBench/WISE 为
  `k∈{25,50}`，后两者严格留出的 library-val 只有 50 条。
- 旧动作 heads 在 screening block 外训练并冻结；候选 head 只用 k 条标签，screening
  预测为五折 nested-OOF Ridge，候选样本不会被训练后原位评价。
- test 标签只在 admission 策略冻结后使用。
- 主结果的决策单位是 **一个 k-label screening block**；三个 split 是重复实验，不先取均值
  给单次决策。三 screen ensemble 只作为更高成本诊断。

必须区分两个目标：

1. **Investment screening**：k 条结果能否判断“补齐正常训练标签并正式训练 candidate head
   后”是否值得入库。
2. **Low-shot deployment**：candidate head 只用这 k 条结果训练，能否立即安全入库。

## 单个 k=50 screen 的主结果

Admission 使用最简单的条件规则：screen 上 candidate 的 cost-aware OOF Router 边际增益
`ΔR>0`。Oracle opportunity 只作诊断；由于“加入动作”的 Oracle 增益按构造几乎总是非负，
`Oracle Δ>0` 本身不是有效筛选器。

### Low-shot deployment

| Benchmark | screen↔test Spearman | Precision | Recall | harmful admission | Oracle decision value recovered |
|---|---:|---:|---:|---:|---:|
| GM | 0.715 | 0.822 | 0.708 | 15.7% | 0.830 |
| AM | 0.462 | 0.588 | 0.665 | 37.3% | 0.686 |
| UniGen | 0.645 | 0.614 | 0.857 | 32.0% | 0.653 |
| CoreBench | 0.818 | 0.816 | 0.853 | 20.8% | 0.931 |
| WISE | 0.541 | 0.748 | 0.635 | 29.9% | 0.827 |

其中 harmful admission 是在真实 `ΔR_test < -0.001` 的上下文中被误收的比例。50 条已经
提供显著排序信号，但在 AM/UniGen/WISE 上还不足以作为稳定的二元硬门槛。

### Investment screening 的关键反例

若“值得”只定义成完整训练后的质量增益 `>0`，不计新增动作的数据生成、维护和系统复杂度，
直接全部入库已经能恢复 GM/AM/UniGen/CoreBench/WISE 最优 admission value 的
**93.6% / 97.6% / 100.0% / 96.6% / 85.8%**。零阈值 screen 反而只有
81.9% / 76.2% / 77.8% / 93.8% / 83.8%。

这不是 admission 方法失败，而是说明：若忽略已有的动作接入成本，“值不值得”在数学上
接近“只要有任何微小正增益就加入”，预筛选没有存在理由。下面直接使用实验中已有的动作
生成成本和质量收益计算回本点，不再人为引入最小有效收益 `ε`。

## 计入已有接入成本后的回本实验

对 candidate 的完整离线数据生成成本记为
`C_offline = N_prompt × 5 image seeds × action cost`。若预期部署调用量为 `H`，将这笔一次性
成本摊到每次调用后，新增动作库的可用在线预算为
`B_eff = 10 − C_offline/H`；然后比较 `S∪{a}` 在 `B_eff` 下的 test quality 与当前库 `S`
在 budget=10 下的 quality。这样，“值得”就是净增益 `>0`，不存在任意阈值。

在所有最终可回本的 `(S,a)` 上，部署调用量的中位回本点为：

| Benchmark | GM | AM | UniGen | CoreBench | WISE |
|---|---:|---:|---:|---:|---:|
| 中位回本调用量 | 23.2k | 36.4k | 35.0k | 5.6k | 7.8k |

不同部署规模下，404 个条件上下文中净收益为正的比例为：

| Benchmark | H=10k | H=30k | H=100k | H=1M |
|---|---:|---:|---:|---:|
| GM | 18.1% | 34.7% | 53.0% | 59.9% |
| AM | 23.5% | 36.6% | 59.7% | 72.8% |
| UniGen | 30.9% | 46.3% | 67.8% | 91.8% |
| CoreBench | 41.6% | 56.7% | 59.2% | 61.9% |
| WISE | 32.9% | 52.5% | 53.7% | 54.2% |

因此预筛选在计入接入成本后有明确价值：例如 `H=100k` 时仍有 32.2%–47.0% 的候选上下文
不值得接入，而 `H=10k` 时多数候选都不能回本。便宜动作 F4B4 在有收益时通常只需约
0.3k–1.8k 次调用回本；QTTS 约需 12k–76k；F50/FTTS 在大数据 benchmark 上常需
80k–280k 以上，且部分上下文永远不能回本。

### k=50 预筛选能否预测真实回本（H=100k）

下表以单个独立 k=50 screening block 为决策单位。`mean>0` 偏均衡；`LCB-1SE>0` 是控制
误收的安全规则。

| Benchmark | 值得比例 | mean>0 P/R | LCB-1SE>0 P/R | LCB-1.96SE>0 P/R |
|---|---:|---:|---:|---:|
| GM | 53.0% | .813/.670 | .903/.391 | .977/.196 |
| AM | 59.7% | .732/.602 | .844/.358 | .963/.145 |
| UniGen | 67.8% | .783/.589 | .806/.313 | .788/.100 |
| CoreBench | 59.2% | .835/.833 | .944/.589 | .996/.363 |
| WISE | 53.7% | .749/.670 | .879/.436 | 1.000/.195 |

结论：若优先避免错误接入，k=50 的 `LCB-1SE>0` 在五个 benchmark 上达到
**0.806–0.944 precision**；代价是 recall 只有 0.313–0.589。若希望覆盖更多好动作，使用
`mean>0`，precision/recall 分别为 0.732–0.835 / 0.589–0.833。

这里尚未把 evaluator 与长期维护成本数值化，因此回本调用量是 generation-only 的乐观下界；
加入维护成本只会提高回本点，不会推翻“成本感知预筛选有效”的结论。budget=10 的曲线增益与
Exp37 原始 investment 增益最大误差为 `1.01e-7`。

## 样本量曲线

下表报告 low-shot deployment 的 `Precision / Oracle value recovered`：

| Benchmark | k=25 | k=50 | k=100 | k=200 |
|---|---:|---:|---:|---:|
| GM | — | 0.822 / 0.830 | 0.856 / 0.865 | **0.880 / 0.907** |
| AM | — | 0.588 / 0.686 | 0.851 / 0.890 | **0.901 / 0.911** |
| UniGen | — | 0.614 / 0.653 | 0.711 / 0.789 | **0.766 / 0.856** |
| CoreBench | 0.771 / 0.849 | **0.816 / 0.931** | — | — |
| WISE | 0.669 / 0.648 | **0.748 / 0.827** | — | — |

结论是 50 条适合首轮 screen，但不是所有 benchmark 的最终硬判断规模；GM/AM/UniGen 在
100–200 条时明显更稳。WISE 仍是最困难的压力测试。

## 推荐的通用顺序方法

给定 `(benchmark, current library S, candidate a, budget B)`：

1. 先在 50 条代表性 prompt 上生成并评价候选动作；小数据 benchmark 可从 25 条开始。
2. 旧动作 predictor 冻结；候选动作做 nested-OOF head。
3. 在同一批 prompt 上分别冻结 `S` 与 `S∪{a}` 的预算路由，计算
   `ΔR_hat = quality(S∪{a}) − quality(S)` 及 paired standard error。
4. 若已知预期部署量 `H`，直接扣除 `C_offline/H` 并以净收益 0 为门槛：
   - `net ΔR_hat − SE > 0`：接受；
   - `net ΔR_hat + SE < 0`：拒绝；
   - 其余：追加下一批标签，最多到 100/200。
   若 `H` 未知，则输出 candidate 的回本调用量，不人为指定 `ε`。
5. 接受后用全部已收集 candidate 标签 refit head；动作价值只对当前 `S` 有效，当前库改变后
   应重新测试边际量。

### 质量阈值敏感性：顺序策略结果（τ=0.002，1-SE）

| Benchmark | 平均候选标签 | Precision | Recall | harmful / all contexts | strict staged-oracle value recovered |
|---|---:|---:|---:|---:|---:|
| GM | 103.0 | 0.910 | 0.628 | 3.1% | 0.766 |
| AM | 102.4 | 0.848 | 0.580 | 4.6% | 0.726 |
| UniGen | 96.5 | 0.802 | 0.812 | 7.3% | 0.795 |
| CoreBench | 35.4 | 0.859 | 0.695 | 5.8% | 0.830 |
| WISE | 37.8 | 0.805 | 0.573 | 6.8% | 0.746 |

这里的 value recovered 使用更严格上界：oracle 可在所有阶段中选择真实 low-shot 增益最高的
阶段，因此低于固定 k 表中的同名数值。该顺序策略优先控制误收；若目标更看重 recall，可将
`τ` 降到 0 或使用 0.5-SE。

## 跨 Benchmark 阈值迁移

以单个 k=50 screen 为单位，在四个 benchmark 上选择阈值、留一 benchmark 测试：

- low-shot 的 leave-one-benchmark-out 阈值落在约 `0.0015–0.0034`（quality-only）或
  `0.0031–0.0047`（计入 `ε=0.001` 最小有效收益）范围，均为稳定的正阈值。
- investment target 的最优阈值却为负，意味着仅按最终质量评价时应接近“多数都收”。这再次
  证明 investment admission 必须把全量标签生成/维护成本显式写入效用，不能只看质量正负。

## 结论

1. **可以总结一套通用方法，但通用的是条件 admission procedure，不是动作名单。**
2. **最有效的统计量是 cost-aware nested-OOF Router marginal gain。** Oracle `Δ>0` 按构造
   接近恒真，只能说明机会大小，不能单独决定入库。
3. **50 条适合作为第一阶段，不适合作为所有 benchmark 的最终二元判断。** 100–200 条能
   明显改善难 benchmark；顺序 accept/reject/defer 比固定样本量更符合实际。
4. **计入已有离线生成成本后，预筛选值得做。** 决策应由部署量 `H` 下的净收益或回本调用量
   驱动，不需要再造一个任意 `ε`；k=50 的 1-SE 规则适合安全接入。
5. 当前结论来自已有 A7 动作的 retrospective pseudo-new-action 回放。论文若要声称 prospective
   validity，最后还应选一个真正未参与协议设计的新动作，按冻结规则做一次前瞻验证。

## 产物与数值审计

- 脚本：`run_exp37_action_admission_crossbench.py`。
- 汇总：[results/exp37_action_admission_crossbench.json](results/exp37_action_admission_crossbench.json)。
- 成本摊销脚本：`analyze_exp37_amortized_cost.py`；汇总：
  [results/exp37_amortized_cost.json](results/exp37_amortized_cost.json)。
- 39 组候选 low-shot 预测缓存与 39 组 scenario 断点缓存均完整。
- 总计 15,756 个 single-screen 条件试验；全部 JSON/预测/聚合数值有限。
- screen 与旧动作 train/alpha 集严格不相交；investment target 在不同 k 下完全一致。
- 所有冻结策略平均成本均不超过 budget=10；最大值 `10.000000000000002` 为浮点误差。

</details>
