# §39 混合多 Benchmark 数据能否产生可迁移 Router

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

§39 已完成 CPU 重算，当前结果为 [JSON](results/prompt_disjoint_v1/exp39_pooled_router_lobo.json)。UniGen 使用 pool2000/raw600，所有新预测、基线与汇总来自同一数据版本。历史正文中的旧 UniGen 分数不再作为当前结论。GM/AM 是同一 prompt 来源的两种评分。


| 标签变换 | 平均 Router−Pooled LP | 显著正 / 负的评分集数 |
|---|---:|---:|
| raw_score | -0.008261 | 0 / 3 |
| q50_advantage | -0.008263 | 0 / 3 |
| normalized_q50_advantage | -0.008200 | 0 / 2 |

三种变换下平均收益均为负，没有显著正收益的评分集。此处基线是同 source 标签训练的 Pooled LP；目标侧 LP、同域 Router 是另外的对照，不混用差值。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 严格问题

每次完整留出一个目标 benchmark，只用其余四个 benchmark 混合训练一个 Router，再在目标上
零样本部署。目标 benchmark 的训练标签也完全不可见；因此这是 unseen-benchmark transfer，
而不是把目标数据混入后的普通多域监督训练。

五个目标依次为 GM、AM、UniGen、CoreBench、WISE，统一公共 A7、budget=10、split seeds
42–46。为避免大数据集淹没 CoreBench/WISE，每个 source 固定抽300条 Ridge train 和50条
alpha-val；每次 pooled fit 共1200 train + 200 alpha-val。target 侧只用预测和成本冻结策略，
冻结后才读取 test labels；每个目标使用2000次 paired bootstrap。

测试三种混合标签：

1. `raw_score`：直接混合各 benchmark 原始质量分数。
2. `q50_advantage`：预测每个动作相对同 prompt Qwen50 的增益，消除 prompt 公共基线。
3. `normalized_q50_advantage`：再用各 source train-only 的增益标准差做尺度归一化。

主基线是与每种方法使用相同四源标签的静态 pooled LP。只有
`Pooled-Ridge − Pooled-LP > 0` 才能证明实例级 Router 规律发生了迁移。

## 总体结果

| 混合方式 | 五目标 macro learned gain | 正收益目标 | 显著正 | 显著负 | 超过目标标签 oracle 选出的最佳单源 Router |
|---|---:|---:|---:|---:|---:|
| raw score | −.0088 | 1/5 | 0/5 | 4/5 | 1/5 |
| Q50 advantage | −.0088 | 1/5 | 0/5 | 4/5 | 1/5 |
| normalized Q50 advantage | **−.0082** | 1/5 | 0/5 | 3/5 | 1/5 |

尺度归一化略微减轻负迁移，但没有改变结论。只减去 Q50 与 raw score 的部署结果几乎相同，
说明去掉 prompt 公共偏置本身不足以得到跨域动作规律。

## 最好混合方式逐 Benchmark 结果

下表使用 `normalized_q50_advantage`：

| Held-out target | Pooled Ridge | Matched pooled LP | learned gain | 95% CI | 同域 Ridge |
|---|---:|---:|---:|---|---:|
| GM | .3345 | .3434 | −.0089 | [−.0205,+.0028] | .3583 |
| AM | .7642 | .7610 | +.0032 | [−.0014,+.0081] | .7808 |
| UniGen | .8142 | .8294 | −.0152 | [−.0243,−.0056] | .8893 |
| CoreBench | .5038 | .5132 | −.0093 | [−.0143,−.0042] | .5403 |
| WISE | .7718 | .7824 | −.0106 | [−.0187,−.0026] | .7957 |

只有 AM 为小幅正值，但不显著。UniGen、CoreBench、WISE 明确负迁移；GM 的 CI 跨0。

混合训练相对四个单源 Router 的平均值在 GM/AM/UniGen/CoreBench 上有所改善，说明它能降低
单一 source 的方差与极端失败；但它只在 UniGen 上超过了“用 target test label 事后挑出的最佳
单源 Router”约 +.0057，而 UniGen 上仍显著低于静态 pooled LP −.0152。因此这是稳健化，
不是成功的 prompt-level transfer。

所有 pooled Router 仍高于固定 Qwen50（约 +.034到+.086），但同样主要来自四源共享的静态
动作组合；不能把这部分记为 Router 泛化收益。

## 结论与下一步

1. **把其他 benchmark 的学习数据直接混合，不能让当前 Prompt Ridge Router 零样本迁移到
   未见 benchmark。** 最佳归一化方法仍为 −.0082 macro learned gain，0/5显著为正。
2. **混合有一定稳健化作用，但无法替代目标域监督。** 它通常好于平均单源迁移，却仍低于
   matched static LP 和同域 Router。
3. 若目标是“一个模型服务已知五个 benchmark”，可以把目标域样本也纳入训练；但那是多域
   supervised Router，不是 unseen-benchmark 泛化。
4. 当前最值得继续验证的是 few-shot adaptation：向四源 pooled training 加入目标域
   `k∈{25,50,100,200}` 标签，并与 target-only k-shot 比较。需要证明混合 source 数据减少达到
   同一目标域收益所需的标签量，才算迁移真正有价值。

## 审计与产物

- 5个 held-out targets × 5 seeds × 3种混合方式全部完成。
- 最大部署成本10.0，全部满足 budget；所有数值有限。
- target labels 不参与 scaler、Ridge、alpha、LP 或部署策略选择。
- 脚本：`run_exp39_pooled_router_lobo.py`。
- 结果：[results/exp39_pooled_router_lobo.json](results/exp39_pooled_router_lobo.json)。
- 25份 target×seed 预测缓存：`results/exp39_pooled_predictions_*.npz`。

</details>
