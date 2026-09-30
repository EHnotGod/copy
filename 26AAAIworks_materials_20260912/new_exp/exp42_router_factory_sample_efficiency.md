# §42 Router Factory：样本效率、构建成本与安全停止规则

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

完整曲线已在 CPU 重跑，当前分数与 gate 以 [JSON](results/prompt_disjoint_v1/exp42_router_factory_sample_efficiency.json) 为准。使用同版本 §38 全量参照和 §40 source300 预测，重新计算目标模型、LP、source 校准与置信区间。

下表为 dense / target_only 的末端点；k 包含 safety 标签。


| 评分集 | k | Router−LP | 95% CI | gate通过率 |
|---|---:|---:|---|---:|
| GM | 400 | -0.004949 | [-0.014154, +0.004111] | 0% |
| AM | 400 | +0.008915 | [+0.004779, +0.013296] | 20% |
| UniGen | 400 | -0.003375 | [-0.010748, +0.004147] | 0% |
| CoreBench | 400 | +0.029441 | [+0.021966, +0.037466] | 80% |
| WISE | 400 | +0.008445 | [-0.002526, +0.019309] | 0% |
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 结论先行

1. **可以低成本开始，但不能假设400条目标域数据就足以稳定建成 Router。** 在公共 A7 上，
   400条 dense 数据加少量 source reuse 的平均 `Router−LP` 为 `+.0133`，只恢复 full-data
   Router 平均 `+.0270` 的49%；GM 仍不稳定。
2. **按动作成本稀疏标注适合预研，不适合直接替代最终 dense 数据。** `k=400` 时它将生成
   成本从231,310降到74,377 units（−67.8%），但平均 raw learned gain 从 `+.0133` 降到
   `+.0085`，且 GM 为负。
3. **复用 source Router 值得，但只应作为小权重 prior。** 固定25% calibrated source prior
   在所有12个 `(dense/sparse)×k` 宏观比较中都提高 raw gain，改善约 `+.0010` 到
   `+.0024`；它没有消除目标域监督的必要性。
4. **真正的瓶颈是稳定验证，不只是训练。** 最多50条 safety prompt 时，400条 dense 数据
   只有24%的 split 能通过严格部署 gate；五个 benchmark 中只有 CoreBench 达到80%的通过率。
   因此正确的系统行为是：先上线静态 LP，Router 未被证实时自动回退，而不是为了上线 Router
   放宽统计标准。
5. **推荐构建方式是 staged acquisition。** 先用便宜骨架和稀疏标签做100条级别的探索；只有
   raw signal 和摊销价值都为正，才补 expensive-action 标签并扩大独立 safety block。停止条件
   应由验证下界触发，不使用统一固定样本数。

## 实验协议

- Benchmark：GM、AM、UniGen、CoreBench、WISE。
- 公共动作库 A7：Qwen/Flux × `{10,50,TTS}` + Flux4B-4step。
- 5个 construction split seeds：42–46；测试集始终独立。
- 总目标域 prompt 数 `k={25,50,100,200,300,400}`；其中最多50条是从未参与训练或
  alpha选择的 dense safety prompts。
- 每个动作一个独立 Ridge head；alpha 在该动作 fit labels 的内部20%上选择，随后用全部 fit
  labels 固定 alpha 重训。
- dense：每个 prompt 生成所有 A7 动作。
- cost-stratified：safety prompts 保持 dense；训练标签按
  `max(20, round(fit_cap × min(1, 4.7/action_cost)))` 分配。
- source reuse：留出目标 benchmark，均匀混合其余四个 `n=300` residual experts；用目标标签
  做逐动作 location/slope calibration，再以固定25%权重和 target-only prediction 混合。
- 安全部署：至少50条 safety prompts，且 paired bootstrap 95% CI 的
  `Router−matched target LP` 下界大于0；否则部署结果精确回退 LP。
- 生成成本按已有标签的5个 image seeds 计：
  `C_gen = 5 × Σ_a n_a c_a`。source expert 的历史训练成本视为可复用 sunk cost。

共训练300个 target Router，完成600个 method runs；最大策略成本为10.000000000000002，预算
检查通过，所有数值有限。

## 五 benchmark 宏观曲线

下表使用更好的 `source25_calibrated`。`raw` 是不经过 safety gate 的 Router 相对自身 matched
LP 的测试收益；`safe` 是 gate 后的实际部署收益。

| Layout | k | 生成成本 | macro raw | macro safe | 平均通过率 | raw负目标数 |
|---|---:|---:|---:|---:|---:|---:|
| dense | 25 | 14,457 | +.0030 | .0000 | 0% | 1/5 |
| dense | 50 | 28,914 | +.0063 | .0000 | 0% | 1/5 |
| dense | 100 | 57,828 | +.0088 | .0000 | 0% | 1/5 |
| dense | 200 | 115,655 | +.0085 | .0000 | 0% | 1/5 |
| dense | 300 | 173,482 | +.0109 | +.0040 | 16% | 1/5 |
| dense | 400 | 231,310 | **+.0133** | **+.0058** | 24% | **0/5** |
| sparse | 25 | 14,457 | +.0030 | .0000 | 0% | 1/5 |
| sparse | 50 | 18,064 | +.0034 | .0000 | 0% | 2/5 |
| sparse | 100 | 26,178 | **+.0080** | .0000 | 0% | 2/5 |
| sparse | 200 | 44,719 | +.0049 | .0000 | 0% | 1/5 |
| sparse | 300 | 61,432 | +.0053 | +.0013 | 8% | 1/5 |
| sparse | 400 | 74,377 | **+.0085** | **+.0042** | 16% | 1/5 |

三个直接的成本判断：

- sparse `k=100` 用26,178 units 得到 `+.0080` raw gain；它是很好的 cheap probe，但 GM 和
  UniGen 尚未转正，不能无 gate 部署。
- sparse `k=400` 和 dense `k=200` 的 raw gain 都约 `+.0085`；前者少花35.7%生成成本。
- dense `k=400` 才把五个目标的点估计全部拉到非负，但仍只恢复 full-data macro learned gain
  的49.3%，说明400不是普适的“训练完成”阈值。

曲线不是严格单调的，主要来自低样本高维 Ridge、expensive-action 标签过稀和 construction
split 方差。这本身就是不采用固定 `k` 停止的证据。

## `k=400` 的逐 benchmark 结果

| Target | full-data gain | dense raw | raw 95% CI | 恢复率 | safety通过率 | 预计检出所需 safety n |
|---|---:|---:|---|---:|---:|---:|
| GM | +.0149 | +.0007 | [−.0081,+.0096] | 5% | 0% | 123,964* |
| AM | +.0198 | +.0110 | [+.0069,+.0151] | 56% | 20% | 120 |
| UniGen | +.0599 | +.0148 | [+.0084,+.0214] | 25% | 20% | 110 |
| CoreBench | +.0271 | +.0290 | [+.0210,+.0377] | 107% | 80% | 81 |
| WISE | +.0133 | +.0112 | [+.0002,+.0224] | 84% | 0% | 936 |

`*` GM 在当前400条配置下的均值接近0，因此该规划值没有实际采购意义；应继续训练、换特征，
或停在 LP，而不是购买十万条 safety labels。

“预计检出所需 safety n”是用 held-out paired prompt 方差和观测 effect size 计算的规划量
`ceil((1.96σ/g)^2)`，不是部署证书。它说明统一50条 safety 数据不合理：AM/UniGen 大约需要
110–120，CoreBench 约80，WISE 因 prompt-level 方差大可能接近千条。

## Source reuse 的价值

`source25_calibrated − target_only` 的 macro raw gain 增量：

| Layout | k=25 | 50 | 100 | 200 | 300 | 400 |
|---|---:|---:|---:|---:|---:|---:|
| dense | +.0010 | +.0016 | +.0022 | +.0021 | +.0013 | +.0021 |
| sparse | +.0010 | +.0013 | +.0021 | +.0016 | +.0021 | +.0024 |

结论是“低权重复用有小而一致的宏观价值”，不是“source Router 可以迁移”。它只占25%，先做
目标域尺度校准，且最终仍经过目标域 safety gate；这与 §38–40 中 zero-shot Router 迁移失败
并不矛盾。

## 推荐的 Router Factory

### 1. 冷启动与动作库

- 生产默认立即上线 matched static LP。
- 初始最小骨架用 `Flux-10 + Qwen-TTS + Flux4B-4step`；Qwen/Flux-50 不作为默认冷启动
  动作。
- 新动作只生成自己的标签并训练独立 head；按 §37 的 `k=50→100→200` admission 判断
  边际收益是否覆盖生成和维护成本。
- 50条 dense safety prompts 同时复用于动作库验证、Router gate 和之后的新动作 admission，
  但绝不进入训练。

按本实验 sparse 分配规则，A7 `k=100` 的生成成本为26,178 units；若冷启动仅用上述 A3，
对应成本为约8,678 units。A3 的这项数字只是成本核算；本实验的 Router 质量曲线使用 A7，
不能把 A7 的质量收益直接归给 A3。

### 2. Router 数据获取

建议使用三段式，而不是一次购买完整 action matrix：

1. `probe`：便宜动作约100个 fit labels；昂贵动作保底20个；先判断 prompt×action signal。
2. `expand`：仅对 admission 已通过、head R²/路由敏感度高的动作补到100/200/300。
3. `certify`：根据 paired variance 扩大独立 safety block；不再用固定50覆盖所有 benchmark。

source experts 已存在时，默认加入“逐动作目标校准后的25% source prior”；不要使用 source-only
Router，也不要扩大该权重，除非新的目标 safety evidence 支持。

### 3. 自动停止规则

每轮 acquisition 后按以下顺序决策：

1. 冻结 action heads、动作库和预算10下的 Router policy。
2. 在独立 safety block 上计算 paired `Δ=Router−LP`。
3. 若 `CI95_lower(Δ)>0`，再把新增离线成本代入 horizon-aware utility；两者均为正才部署
   Router。
4. 若 CI 跨0：线上继续 LP；用当前 `σ/g` 估计下一轮 safety/fit 样本数，仅当预计调用量能摊销
   下一轮成本时继续买数据。
5. 若 point estimate 为负或连续两轮不改善：停止该 Router 路线，保留 LP；检查动作库、特征或
   benchmark metric，而不是盲目扩样本。

所以 Router Factory 的“稳定”来自可验证回退和按证据扩数，而不是期望一个统一的400条配方
在所有 benchmark 上都成功。

## 数据限制

未运行统一的1/3/5 image-seed 曲线。UniGen pool 和 CoreBench Flux4B pool 缺少覆盖 A7 的统一
逐 image-seed 分解；强行比较会改变 benchmark/action 数据支持，不再是严格五 benchmark paired
实验。当前成本统一按已有标签的5 seeds 核算。

## 产物

- 脚本：`run_exp42_router_factory_sample_efficiency.py`
- 结果：[results/exp42_router_factory_sample_efficiency.json](results/exp42_router_factory_sample_efficiency.json)
- 300个 target Router prediction caches 已保存，可在不重训的情况下重算 gate 和成本口径。

## 长尾补充

GM、AM、UniGen 的 `k=400→2400` 曲线已在 §43 补齐。三个 benchmark 均到2400才首次恢复
90% full-data gain；AM首次稳定上线点为1800，UniGen为1200，GM在2400时仍需要更多
deployment-like safety 数据。详见 `exp43_largebench_router_tail.md`。

</details>
