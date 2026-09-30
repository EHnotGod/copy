# §51 固定Flux三动作的五Benchmark单一Joint Router

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

完成全部 5 个 split seeds（42–46）的 CPU 重训。固定 Flux3、每项评分 300 train + 50 alpha-val；UniGen 已改为独立的 pool2000/raw600。GM/AM 是同一 prompt 来源的两种评分，五项评分对应四个 prompt 来源。

| 评分集 | Target LP | Joint | Specialist | Joint−LP (95% CI) |
|---|---:|---:|---:|---|
| GM | 0.323623 | 0.323890 | 0.322760 | +0.000268 [-0.002943, +0.003432] |
| AM | 0.752023 | 0.752461 | 0.753175 | +0.000438 [-0.000886, +0.001726] |
| UniGen | 0.821896 | 0.826042 | 0.826840 | +0.004146 [+0.000233, +0.008384] |
| CoreBench | 0.507022 | 0.524999 | 0.524027 | +0.017977 [+0.012538, +0.023608] |
| WISE | 0.772644 | 0.784468 | 0.784069 | +0.011824 [+0.006787, +0.017102] |

五项评分等权平均：Joint **0.642372**，specialist **0.642174**，差值 **+0.000198**。该均值没有给出跨独立数据域的显著性检验，不能据此声称统一模型显著优于 specialist。

[JSON](results/prompt_disjoint_v1/exp51_joint_supervised_router_flux3.json) · ![修复后的 Joint](results/prompt_disjoint_v1/exp51_joint_supervised_router_flux3.png)

A7 同协议重训：Joint **0.650617**，specialist **0.653899**，差值 **-0.003283**。[JSON](results/prompt_disjoint_v1/exp50_joint_supervised_router.json)。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 直接问题

固定完全相同的动作空间 `Flux-9B {10 steps, 50 steps, TTS/edit}`，能否把五个已知
benchmark的数据放在一起，只训练一个Router，并在五个benchmark上分别做动作决策？

这里不存在唯一的逐prompt“分类正确标签”：动作决策受全局budget约束，并允许边界策略混合。
因此主指标是固定budget=10下的实际Router绝对平均质量，而不是无预算的top-1 accuracy。

## 协议

- Benchmark：GM、AM、UniGen、CoreBench、WISE。
- 每个benchmark平衡抽取300条Ridge train与50条alpha validation；joint模型合计
  1500 train + 250 validation。
- 所有benchmark共用一个Qwen3 prompt embedding scaler和三个Ridge action heads。
- 5个construction split seeds：42–46；所有测试集与construction独立。
- 对比：每个benchmark自己的target static LP，以及使用同样300/50标签预算的五个specialist。
- `Joint + benchmark ID` 只额外加入五维domain one-hot截距特征，仍是单模型；用于检查简单
  domain conditioning是否必要。

![Joint Flux3 Router](results/exp51_joint_supervised_router_flux3.png)

## 结果

| Benchmark | Target LP | 一个Joint Router | Joint+ID | Specialist | Oracle | Joint−LP (95% CI) | Joint−Specialist (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GM | 0.323623 | 0.326862 | 0.326855 | 0.322760 | 0.383843 | +0.003240 [-0.000256,+0.006798] | +0.004102 [+0.001271,+0.006930] |
| AM | 0.752023 | 0.753568 | 0.753712 | 0.753175 | 0.777977 | +0.001545 [+0.000225,+0.002873] | +0.000393 [-0.000820,+0.001629] |
| UniGen | 0.821896 | 0.828963 | 0.828764 | 0.829798 | 0.871333 | +0.007067 [+0.003633,+0.010766] | -0.000836 [-0.004411,+0.002805] |
| CoreBench | 0.507022 | 0.525653 | 0.525659 | 0.524027 | 0.552545 | +0.018631 [+0.013240,+0.024118] | +0.001626 [-0.000956,+0.004214] |
| WISE | 0.772644 | 0.785477 | 0.785391 | 0.784069 | 0.840504 | +0.012833 [+0.007702,+0.018370] | +0.001408 [-0.002841,+0.005836] |

五benchmark等权macro：

- 一个Joint Router：`0.644105`；
- Joint + benchmark ID：`0.644076`；
- 五个specialist：`0.642766`；
- Joint相对target LP平均：`+0.008663`；
- Joint相对specialist平均：`+0.001339`。

## 结论

1. **可以做成一个固定动作空间的统一Router。** Joint Router在五个benchmark上的点估计均高于
   各自target LP；AM、UniGen、CoreBench、WISE的paired CI显著高于0，GM为正但CI略跨0。
2. **单一Joint Router没有牺牲整体质量。** macro质量比五个同预算specialist高0.00134；逐项
   比较中，只有UniGen为小幅负值且不显著，其余四项均为正点估计。
3. **简单benchmark ID没有价值。** `0.644076`与plain joint的`0.644105`近乎相同，说明当前
   三动作下共享prompt→动作规律本身已经足够，不需要先分类benchmark再选head。
4. 该正结论适用于“所有已知benchmark都参与joint监督训练”。它不推翻§38–39的结论：若一个
   新benchmark完全没有标签，zero-shot跨域仍不可靠。

## A7敏感性补充

动作空间扩大为公共A7（Qwen/Flux × `{10,50,TTS}` + Flux4B-4step）时，同协议的joint
macro为`0.653918`，specialist为`0.655826`，joint低`0.001908`；GM不超过LP，只有AM与
CoreBench显著超过LP。因此“单一Joint Router”在简单固定Flux三动作下成立得更干净，动作空间
变复杂后仍接近specialist，但不再能保证每个benchmark都获得明确routing收益。

## 产物

- [results/exp51_joint_supervised_router_flux3.json](results/exp51_joint_supervised_router_flux3.json)
- [results/exp51_joint_supervised_router_flux3.png](results/exp51_joint_supervised_router_flux3.png)
- `run_exp50_joint_supervised_router.py --action-space flux3`
- A7结果：[results/exp50_joint_supervised_router.json](results/exp50_joint_supervised_router.json)

</details>
