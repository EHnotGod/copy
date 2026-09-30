# §40 Nested-LOBO Router 混合网格

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

§40 已完成 CPU 重算，当前结果为 [JSON](results/prompt_disjoint_v1/exp40_nested_router_mix_grid.json)。UniGen 使用 pool2000/raw600，所有新预测、基线与汇总来自同一数据版本。历史正文中的旧 UniGen 分数不再作为当前结论。GM/AM 是同一 prompt 来源的两种评分。


| source-only 选参规则 | 平均 Router−对应静态策略 | 最差评分集差值 | 显著正 / 负 |
|---|---:|---:|---:|
| exact | -0.002054 | -0.006948 | 0 / 1 |
| one_se_safe | -0.001922 | -0.004810 | 0 / 1 |

嵌套选择缩小了相对 §39 的平均损失，但两种规则仍未获得显著正收益。回退或弱化 Router 可以减少损失，不能据此写成已解决未见域泛化。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 研究问题与防泄漏协议

本实验研究能否通过选择 source 数据规模、source 权重和 Router residual 强度，使 Router 泛化
到完全未见的 benchmark。

- 外层依次完整留出 GM、AM、UniGen、CoreBench、WISE；外层标签不参与任何网格选择。
- 在剩余四个 benchmark 内做 inner leave-one-domain-out，共4个 inner targets × 5 seeds =
  20个调参试验。
- 每个 source 独立训练 two-way-centered residual expert：先去掉 source 动作先验，再去掉每个
  prompt 的公共难度，只学习 prompt×action 交互。
- source 混合仅使用无标签 target prompt embedding 的域中心余弦相似度。
- 最终预测为 `mixed normalized action prior + γ × mixed residual`；`γ=0` 精确退化为 matched
  static LP，是预先允许的安全 abstention。
- 内层用50条 meta-validation prompt 和253个粗 λ 点选参；外层冻结配置后使用 §25 的完整
  λ 网格和 held-out test labels 评价。

网格共70组：

- 每 source 样本数：`{100,300}`；
- source 权重：top-`{1,2,3}`、uniform-all、similarity softmax temperature
  `{0.01,0.05,0.2}`；
- residual 强度：`γ∈{0,0.25,0.5,1,2}`。

同时报告两种严格 source-only 选择：exact 取 inner learned gain 均值最高；one-SE safe 在最优
一个标准误内优先更小 γ 和更平滑的 source 混合。

## 选择与外层结果

主指标是相对该配置自身 matched static LP 的 learned gain。

| Outer target | inner exact 配置 | inner gain | outer exact gain | 95% CI | one-SE safe outer gain |
|---|---|---:|---:|---|---:|
| GM | n100 / uniform / γ=0 | .0000 | .0000 | [.0000,.0000] | .0000 |
| AM | n100 / uniform / γ=0 | .0000 | .0000 | [.0000,.0000] | .0000 |
| UniGen | n100 / softmax-.01 / γ=.25 | +.0041 | −.0048 | [−.0108,+.0012] | −.0048 |
| CoreBench | n300 / softmax-.05 / γ=.25 | +.0064 | −.0048 | [−.0091,−.0005] | −.0032 |
| WISE | n100 / softmax-.01 / γ=.25 | +.0017 | +.0006 | [−.0040,+.0055] | .0000 |

Exact 策略的五目标 macro learned gain 为 `−0.0018`，1/5点估计为正、0/5显著为正、1/5
显著为负。one-SE safe 为 `−0.0016`，0/5正、0/5显著负；GM、AM、WISE 均退回 γ=0。

与 §39 的均匀 pooled normalized-advantage Router 相比：

| 方法 | macro learned gain | 显著负目标 |
|---|---:|---:|
| §39 naive pooled | −.0082 | 3/5 |
| §40 nested exact | −.0018 | 1/5 |
| §40 nested one-SE safe | **−.0016** | **0/5** |

因此结构化混合消除了约78%–80%的平均负迁移，并能通过小 γ/γ=0 控制严重误路由；但没有把
零样本 learned gain 变成正值。

## 网格透露出的规律

1. **大 γ 从未被选中。** 五个 outer target 的 exact 只选择 γ=0 或 .25，说明 source
   residual 只能被弱信任。
2. **更多 source 标签没有稳定帮助。** exact 仅 CoreBench 选择 n300，其余目标选择 n100；
   当前瓶颈是域冲突，不是 source 样本不足。
3. **embedding 相似度权重有合理结构但不等于动作规律相似。** 例如 UniGen 会主要选择
   CoreBench/WISE，WISE 会主要选择 UniGen；这降低了部分极端错误，却没有产生显著正收益。
4. **inner domain 数太少，exact 仍会过拟合。** CoreBench 的 inner gain 为 +.0064，outer
   却显著为 −.0048；one-SE 将其减轻到 −.0032 且 CI 跨0。
5. **GM/AM 暴露了无标签 gate 的边界。** 两者 prompt 完全相同但评分口径不同，仅凭 prompt
   embedding 无法识别 metric shift；inner CV 最终选择 γ=0。

## 结论

1. **网格实验值得做，并显著改善了安全性；但当前五 benchmark 下，它学到的是“何时不要迁移
   Router”，还不是正的零样本泛化。**
2. 最稳的零样本默认仍是 matched static LP；source residual 只能在跨域证据充分时以小 γ
   叠加。
3. 继续扩大纯 source 网格的边际价值已经较低。下一步应给目标域25或50条 calibration 标签，
   只用于选择 source 权重/γ或校准 residual，而不是立即重训完整 Router；这能直接检验极少量
   目标监督能否把当前接近0的安全迁移转成稳定正收益。

## 产物与审计

- 70组配置、5个 outer targets、每个20个 inner target-seed 试验完成。
- 50个 residual expert（5 sources × 5 seeds × 2 sample sizes）缓存完整。
- 外层 target labels 不参与网格选择；target embeddings 仅用于无标签 source 权重。
- 最大外层策略成本10.0，全部满足预算；所有数值有限。
- 脚本：`run_exp40_nested_router_mix_grid.py`。
- 结果：[results/exp40_nested_router_mix_grid.json](results/exp40_nested_router_mix_grid.json)。

</details>
