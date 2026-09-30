# §38 Router 五 Benchmark 零样本泛化

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

§38 已完成 CPU 重算，当前结果为 [JSON](results/prompt_disjoint_v1/exp38_router_crossbench_generalization.json)。UniGen 使用 pool2000/raw600，所有新预测、基线与汇总来自同一数据版本。历史正文中的旧 UniGen 分数不再作为当前结论。GM/AM 是同一 prompt 来源的两种评分。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 问题与严格协议

本实验检验的是 **prompt-level Router 本身**能否跨 benchmark 零样本迁移，不是跨 benchmark
选择固定动作库。五个 benchmark 为 GM、AM、UniGen、CoreBench、WISE；由于 CoreBench/WISE
没有 Boogu 动作，统一使用五者完整共有的 A7：Qwen/Flux × `{10,50,TTS}` + F4B4。

- 枚举完整 `5×5=25` 个 source→target 方向，其中20个为跨 benchmark。
- 5个 split seeds：42–46；budget=10；每个方向使用相同 Ridge Router。
- GM/AM/UniGen：2200条 source Ridge train + 200条 source alpha-val。
- CoreBench/WISE：300条 source Ridge train + 50条 source alpha-val；保留的50条 library-val
  不参与本实验。
- source scaler、Ridge heads、逐动作 alpha 均只读取 source 标签。
- target 侧只用预测值与动作成本冻结 λ/混合策略；冻结后才读取 target test 标签。
- 主比较为 `Cross-Ridge − Source-LP`。Source-LP 使用同一份 source supervision，但只迁移
  静态动作比例，因此差值才是跨域 prompt-level learned routing 的净贡献。
- 每个方向对5个 split 的逐 prompt 差值取均值后做2000次 paired bootstrap。

## 完整 5×5 结果

表中数值为 `Cross-Ridge − Source-LP`；正值表示迁移 Ridge 比仅迁移动作先验更好。对角线是
同域 sanity check，并非跨域结果。

| train \ test | GM | AM | UniGen | CoreBench | WISE |
|---|---:|---:|---:|---:|---:|
| GM | **+.0149** | +.0105 | −.0352 | −.0169 | −.0054 |
| AM | +.0106 | **+.0198** | −.0288 | −.0109 | +.0024 |
| UniGen | −.0129 | −.0005 | **+.0599** | −.0017 | −.0018 |
| CoreBench | −.0403 | −.0113 | −.0316 | **+.0271** | +.0017 |
| WISE | −.0597 | −.0202 | −.0225 | −.0141 | **+.0133** |

### 总体统计

- 五个同域对角线全部为正且95% CI 全部高于0：Router 在每个 benchmark 内都确实可学。
- 18个真正的 prompt-domain shift 方向，平均 learned gain 为 **−.0172**，中位数
  **−.0135**；只有2/18点估计为正，**0/18 显著为正，11/18 显著为负**。
- GM↔AM 是相同 prompt、不同评分聚合的 metric shift：GM→AM 为 +.0105，95% CI
  `[+.0039,+.0169]`；AM→GM 为 +.0106，但 CI `[-.0027,+.0236]` 跨0。评分口径变化
  比真正 prompt-domain shift 温和，但仍不完全稳定。

按 target 看，跨其他 benchmark 训练得到的 Router 相对 Source-LP 的平均 learned gain 为：

| Target | GM | AM | UniGen | CoreBench | WISE |
|---|---:|---:|---:|---:|---:|
| 其他4个 source 平均 | −.0256 | −.0054 | −.0295 | −.0109 | −.0008 |

UniGen 最明确：四个外域 source→UniGen 全部显著为负。CoreBench 的四个方向均为负，其中
三条显著。WISE 的外域差值接近0且均不显著，表示迁移 Ridge 没有稳定优于或劣于静态 LP，
不能解释成学到了可复用的 prompt-level 规律。

## 结论

1. **当前 Prompt Ridge Router 不具备可靠的跨 benchmark 零样本泛化。** 真正 domain shift
   下，直接迁移 Router 通常比只迁移 source 动作先验更差。
2. **失败不是因为 Router 完全学不会。** 五个同域结果全部显著为正；问题在于 prompt→动作
   质量的映射依赖 benchmark 分布和评分定义。
3. **跨域 Cross-Ridge 仍经常高于固定 Qwen50，但这主要来自 A7 的静态成本—质量组合，不能
   归功于实例级 routing。** 因而必须以 Source-LP 而不是单个固定动作作为泛化基线。
4. 对新 benchmark，当前安全方案是先迁移静态动作先验，再收集少量目标域标签校准 Router；
   不应直接部署其他 benchmark 的 Ridge heads。

## 数值审计与产物

- 25/25方向、125个 seed-direction 运行完成；最大策略成本10.0，全部满足 budget。
- 所有聚合数值有限；target test labels 不参与策略选择。
- 脚本：`run_exp38_router_crossbench_generalization.py`。
- 结果：[results/exp38_router_crossbench_generalization.json](results/exp38_router_crossbench_generalization.json)。
- 25份 source×seed 跨域预测缓存：`results/exp38_cross_a7_predictions_*.npz`。

</details>
