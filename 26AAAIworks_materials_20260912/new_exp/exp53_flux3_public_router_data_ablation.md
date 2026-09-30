# §53 固定Flux3动作库的公共Router训练数据条件消融

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

§53 已完成 CPU 重算，当前结果为 [JSON](results/prompt_disjoint_v1/exp53_flux3_public_router_data_ablation.json)。UniGen 使用 pool2000/raw600，所有新预测、基线与汇总来自同一数据版本。历史正文中的旧 UniGen 分数不再作为当前结论。GM/AM 是同一 prompt 来源的两种评分。

数据量消融 §54 也已完成：[JSON](results/prompt_disjoint_v1/exp54_flux3_pooled_router_data_curve.json)。

![修复后的数据消融](results/prompt_disjoint_v1/exp53_flux3_public_router_data_ablation.png)

按测试表现排列的配置仅用于诊断，不能把排序首项当成独立验证过的部署选择规则。

固定 n300、uniform_all，仅改变 residual 强度：


| gamma | 平均 Router−Target LP | 显著正 / 负 |
|---:|---:|---:|
| 0 | +0.000000 | 0 / 0 |
| 0.25 | +0.000785 | 0 / 0 |
| 1 | +0.000231 | 0 / 0 |

gamma=0 是静态策略；极小浮点差不计作学习收益。弱 residual 接近 LP，不等于已经证明实例级路由有效。

§54 固定 Flux3 的数据曲线：

| 每源训练数 | 平均 Router | 平均 Router−Target LP | 显著正 / 负 |
|---:|---:|---:|---:|
| 100 | 0.634466 | -0.000976 | 0 / 0 |
| 200 | 0.635472 | +0.000030 | 0 / 0 |
| 300 | 0.636261 | +0.000819 | 0 / 0 |

100→300 的点估计改善，但每个训练量都没有显著超过 Target LP 的评分集；该范围不足以定位通用饱和点。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 问题与口径

目标是让完全未见目标benchmark标签的公共Router，在固定动作库
`Flux-9B {10 steps, 50 steps, TTS/edit}`、budget=10下，达到或接近使用目标训练标签构造的
`Target Static LP`。

- 外层依次留出GM、AM、UniGen、CoreBench、WISE，5个split seeds（42–46）。
- Target Static LP见过目标侧300条训练标签；它只作为严格评价基线。
- 单一pooled Router平衡混合其余四个source，每源使用`n={100,200,300}`条Ridge train和
  50条alpha validation；目标标签不进入拟合或策略选择。
- 结构化消融复用source-only two-way residual experts，比较每源100/300条、top-1/2/3/
  uniform/natural-size source权重、residual强度`γ={0,.25,.5,1}`。source相似度只读取无标签
  目标prompt embedding。
- 网格使用253点粗lambda统一筛查；文中关键配置再用2701点完整lambda精确复算。

![Flux3 public Router data ablation](results/exp53_flux3_public_router_data_ablation.png)

## 消融一：公共静态动作先验已经足以复现Target LP

五个benchmark自己的Target Static LP在全部seed上都选择了完全相同的预算前沿：

| Flux3动作 | 静态比例 |
|---|---:|
| Flux-10 | 0.868159 |
| Flux-50 | 0.000000 |
| Flux-TTS | 0.131841 |

因此只使用source标签估计的公共动作先验，在`γ=0`时就能得到macro绝对质量`0.635442`，与
Target Static LP的`0.635442`逐目标相同。每源100条且至少混合两个source时，五个split中已经
稳定复现该前沿；单独使用一个100条source仍偶尔选错，macro为`-0.000112`。

这说明“接近Target Static LP”的第一必要条件不是prompt Router很强，而是**不同benchmark的
动作均值/成本前沿一致，并且source数据足以稳定识别该前沿**。

## 消融二：一个pooled Router的数据量曲线

| 每source训练量 | 总train / alpha-val | Router绝对macro | Target LP macro | Router−LP | 正目标 | 显著正/负 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 400 / 200 | 0.634983 | 0.635442 | -0.000458 | 2/5 | 0/0 |
| 200 | 800 / 200 | 0.635758 | 0.635442 | +0.000317 | 3/5 | 0/0 |
| 300 | 1200 / 200 | 0.636843 | 0.635442 | +0.001401 | 4/5 | 1/0 |

100条/source已足以学习静态动作前沿，但不足以让完整prompt Router稳定迁移；200条处于打平区，
300条才出现较清楚的正向macro趋势。300条配置仍只有WISE显著超过Target LP，不能把300解释成
普适理论阈值，而应表述为当前五benchmark、Qwen3 embedding和Ridge容量下的经验下限。

## 消融三：source覆盖、平衡与弱residual

下表均为完整lambda精确复算；`γ=.25`表示公共动作先验上只叠加25%的跨域prompt residual。

| 公共Router条件 | 绝对macro | 相对Target LP | 正目标 | 显著正/负 |
|---|---:|---:|---:|---:|
| nearest source，100条/source，γ=.25 | 0.635339 | -0.000103 | 2/5 | 0/0 |
| natural-size权重，300条/source，γ=.25 | 0.635867 | +0.000426 | 4/5 | 0/0 |
| 四源等权，300条/source，γ=.25 | 0.636636 | +0.001195 | 4/5 | 1/0 |
| embedding最近source，300条/source，γ=.25 | **0.637280** | **+0.001839** | **5/5** | 0/0 |

最近source在300条/source时由无标签embedding稳定选择：GM↔AM，UniGen←WISE，
CoreBench←UniGen，WISE←UniGen。它五项目标点估计均为正，但各目标CI仍跨0；uniform配置的WISE
是唯一显著正项。最近source的收益说明**训练语料需要覆盖目标prompt语义邻域**，但当前证据还
不足以保证对任意新benchmark都成立。

natural-size权重相对四源等权少`0.000769` macro，说明让GM/AM/UniGen这类大域按原始数据量
淹没小域不是好策略。训练batch或loss应按benchmark平衡，而不是按原始样本数自然混合。

## 训练数据需要满足的条件

1. **共享静态动作前沿。** 各source应对“单位成本下哪个动作值得增加”给出一致结论。本实验
   五域都选择`86.82% Flux-10 + 13.18% Flux-TTS`，这是能零样本复现Target LP的核心原因。
2. **每域平衡而非样本自然混合。** 当前经验配置为四个source各300 train + 50 alpha-val；
   domain-balanced优于natural-size weighting。
3. **目标语义邻域必须在source集合里有覆盖。** 多样性本身不够；至少要有一个prompt分布接近、
   且动作收益规律相容的source。只按embedding近邻无法识别GM/AM这类同prompt但评分口径变化，
   因而还需要source-only LOBO检查动作规律的一致性。
4. **跨域实例级规律只能弱注入。** `γ≈.25`比完全信任source residual更稳；公共先验应是主干，
   prompt residual是小修正。source-only nested LOBO不支持正迁移时必须允许`γ=0`回退。
5. **300条/source是当前observed threshold，不是充分保证。** 它让单一pooled Router从100条的
   `-0.000458`提升到`+0.001401`，但除WISE外置信区间仍跨0。要形成跨新benchmark保证，还需要
   更多彼此独立的source benchmark，而不是只在现有五域继续堆同分布样本。

## 可执行建议

公共Router训练集默认采用“每benchmark 300条dense三动作标签 + 50条alpha-val、benchmark等权”。
部署时以公共static prior为安全基线，只在source-only nested LOBO验证非负时加入约25%的
prompt residual；新目标找不到可靠近邻时直接退回公共prior。若允许25–50条目标校准数据，应优先
用于校准动作均值和选择`γ`，而不是立即重训完整Router。

## 产物

- [results/exp53_flux3_public_router_data_ablation.json](results/exp53_flux3_public_router_data_ablation.json)
- [results/exp54_flux3_pooled_router_data_curve.json](results/exp54_flux3_pooled_router_data_curve.json)
- [results/exp53_flux3_public_router_data_ablation.png](results/exp53_flux3_public_router_data_ablation.png)
- `run_exp53_flux3_public_router_data_ablation.py`
- `run_exp54_flux3_pooled_router_data_curve.py`

</details>
