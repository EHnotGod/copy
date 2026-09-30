# 26AAAIworks：项目与实验记录导航

本项目研究：从 prompt 预测生成质量，能否在给定计算预算下做出有效的生成动作选择？现有工作覆盖预测能力审计、动作库选择、结果图辅助路由、跨 benchmark 泛化，以及 Router 的数据需求。

整理日期：2026-09-11。本页负责导航和阅读口径；具体分数以对应实验记录及其 JSON 为准。本次已逐项核对旧日志主张、数据交集、行序、AM重聚合与正式v2产物，并将24份相关记录合为9份。核对与合并详情见 [核对记录](修改日志.md)。

## 先从哪里看

| 想了解什么 | 推荐入口 | 阅读范围 |
|---|---|---|
| 最近的prompt增强、置信度与集成思路如何串起来 | [研究思路整理](writing/prompt_confidence_ensemble_research_summary.md) | 自包含动机与算法、17行同预算质量对照、置信度进展与边界；外置skill和后续条件集成的位置 |
| 测试时能否更准地定位可能选错动作的prompt | [置信度增强实验](new_exp/results/qwen_action_confidence_20260917/README.md) | 19种风险方法、两种固定Router、完整嵌套隔离；高风险定位未确认稳定增强，低风险端改善；附接口和候选CSV |
| 集成、skill、历史检索是否改善实际路由 | [首轮实验](new_exp/results/qwen_history_retrieval_20260916/README.md) · [原方案](writing/qwen_ensemble_skill_retrieval_plan.md) | 同预算收益检索/折外残差/粗skill及随机对照；收益融合在more1600有增益证据，raw800尚未确认；不代表全面超过已有强基线 |
| 低置信度是否真不准，回退静态LP能否涨点 | [四层拆解](new_exp/results/qwen_router_lp_fallback_20260916/confidence_chain.md) · [回退实验](new_exp/results/qwen_router_lp_fallback_20260916/README.md) | 分开检查收益预测误差、动作失误、LP优势与最终同预算收益；保守5%回退未确认稳定增益 |
| Router哪些case好／差，能否在看标签前识别 | [GM逐case与风险识别](new_exp/results/qwen_router_failure_detection_20260916/README.md) | 两种Router、两套测试；坏case特点与定位、森林200棵树的投票熵和一致性、置信度输出；尚未应用skill修正 |
| 新 Router 涨点来自特征、模型、融合还是求解器 | [Qwen 组件消融](new_exp/results/qwen_router_gain_ablation_20260916/README.md) | GM两套测试及UniGen/CoreBench/WISE；固定组件、打乱/常数/格式、仅embedding融合与同预测λ对照 |
| 聚类与prompt特征策略在其他benchmark上是否有效 | [Qwen三动作跨benchmark验证](new_exp/results/crossbench_qwen_feature_lp_20260916/README.md) | UniGen/CoreBench/WISE，5组三折CV、同预算静态LP对照；目标侧重学与固定GM模型直接迁移分开报告 |
| Prompt分析接LP后能否提高实际系统质量 | [结构特征＋LP收益实验](new_exp/results/geneval2_gm_qwen_feature_lp_20260916/README.md) | 训练侧CV选参数，比较静态/聚类/结构/embedding与融合；raw800和more1600、相同预算与条件CI |
| 当前聚焦：GenEval2 GM与Qwen三动作的相对路由分工 | [Oracle分配与prompt特征](new_exp/results/geneval2_gm_qwen_oracle_groups_20260916/README.md) | 按预算Oracle分组，比较10簇、27组结构的动作分配率与富集，含15条例子及预算/seed敏感性 |
| 全部后续实验、编号与脚本怎么对应 | [new_exp 实验目录](new_exp/README.md) | §1–56，包括没有独立 MD、合并记录和编号复用的情况 |
| 当前公共 Router 主线 | [§51 Joint](new_exp/exp51_joint_supervised_router_flux3.md) → [§52 LOBO](new_exp/exp52_flux3_pooled_lobo.md) → [§53–54 数据条件](new_exp/exp53_flux3_public_router_data_ablation.md) → [§55 Flux4](new_exp/exp55_56_flux4_pooled_router.md#corrected-results) → [§56 每源 1k](new_exp/exp55_56_flux4_pooled_router.md#corrected-results) | 区分已知域联合监督与未见域迁移 |
| Router迁移的设置、对照表与共同结论 | [Router迁移专题](writing/router_transfer_analysis.md) | 25个单源方向、15行多源策略，含LP、专用Router、Oracle及扩容和扩量对照 |
| Router 到底需要多少数据 | [§42](new_exp/exp42_router_factory_sample_efficiency.md) → [§43](new_exp/exp43_largebench_router_tail.md) → [§44](new_exp/exp44_router_stability_surface.md) → [§45 GM](new_exp/exp45_geneval2_4000_training_curve.md) / [§46 AM](new_exp/exp46_geneval2_am_4000_training_curve.md) | 区分平均质量平台与稳定验证通过 |
| 新增数据集做到哪一步 | [construction-v2 记录](new_exp/construction_v2_record.md) | 正式状态与设计已合并，旧阶段流水集中归档 |
| 现有基础模型、图片和标签分布 | [五 benchmark 数据盘点](new_exp/base_model_data_inventory_5bench.md) | 历史物理图片存量；AM可恢复标签与UniGen划分已修正 |
| 当前写作主线与结论边界 | [写作入口](writing/README.md) | 主张对应实验来源，区分当前证据与历史草稿 |
| 本地校验与 CPU 复现 | [操作说明](docs/local_reproduction.md) | 结果哈希、冻结划分、文档路径、回归检查与续跑命令 |
| 论文最初问题与分析思路 | [总实验要求](总实验要求.md)、[writing](writing/) | 研究设计和写作材料，不能直接当作最新实验状态 |

> **修复结果入口**：已完成的 CPU 重跑使用独立目录 `new_exp/results/prompt_disjoint_v1/`，实验主记录页首展示修正结果，旧结果折叠保留。UniGen 使用 pool2000/raw600，旧 Ridge 已按 prompt 对齐。标签补齐与 Hope 已按用户 2026-09-12 决定暂缓；最新完成范围见 [修复清单](修改日志.md#corrected-results)。

## 按研究问题梳理

| 研究线 | 现有证据如何阅读 | 主要记录 |
|---|---|---|
| 绝对分能否预测、R² 是否反映路由价值 | 全样本拟合、中段拟合和实际预算收益要分别看；早期现象分析已较完整 | [早期综合分析](exp1_absolute_predict/exp1_gm_vs_am.md)、[§1–8 历史汇总](new_exp/ALL_EXPERIMENTS.md)、[不含 TTS 的 7 档实验](exp_815/exp_815.md) |
| Delta 是否优于直接预测分数 | 不能写成普遍提升；已有 Ridge 对照几乎相同，具体 predictor/目标设置需逐实验区分 | [Delta 对照](exp1_absolute_predict/exp_delta_routing.md)、[统一预算对照](new_exp/exp11/summary.md) |
| 扩大动作库是否值得 | Oracle 空间增加不代表 Router 能利用；动作筛选、成本与跨域适用性共同决定结果 | [§23–24](new_exp/exp23_10action_distill.md)、[§31–32](new_exp/exp31_action_subset_selection.md)、[§35–37](new_exp/exp37_action_admission_crossbench.md)、[§55–56](new_exp/exp55_56_flux4_pooled_router.md#corrected-results) |
| 先看便宜结果图能否改善路由 | 图像信息增益和完整计费净收益不同；§41 在 GM 的条件预览有正证据，跨域结论仍需后续实验 | [§26 设计](new_exp/exp26_probe_image_router.md#exp26-design)、[§30 2B](new_exp/exp30_qwen35_2b_probe_router.md)、[§41 条件预览](new_exp/exp41_conditional_2b_preview.md) |
| 一个 Router 能否服务多个 benchmark | Flux3 Joint 接近 specialist；修复后的 raw-score LOBO 未有显著超过 Target LP 的评分集，Flux4 扩容后仍存在负迁移；旧 UniGen 重合已排除，当前引用页首修复结果 | [§38–40 泛化](new_exp/exp38_router_crossbench_generalization.md)、[§51 Joint](new_exp/exp51_joint_supervised_router_flux3.md)、[§52 LOBO](new_exp/exp52_flux3_pooled_lobo.md)、[§56](new_exp/exp55_56_flux4_pooled_router.md#corrected-results) |
| 数据是否足够、是否达到平台 | 没有跨所有设置通用的“1600 条饱和”；GM/AM 后期绝对质量趋平，不等同于 GM 已通过稳定部署标准 | [GM 统一质量曲线](new_exp/exp45_geneval2_4000_training_curve.md#exp45-quality)、[GM 稳定性](new_exp/exp45_geneval2_4000_training_curve.md)、[AM 曲线](new_exp/exp46_geneval2_am_4000_training_curve.md) |
| CATImage 现象能否复现 | 有独立数据、模型池、指标和成本单位；保留为单独证据线 | [CATImage 摘要](exp_catimage818/catimage_results_summary.md)、[完整记录](exp_catimage818/exp_catimage818.md) |

## 新数据的准确入口

| 数据/产物 | 可核对的本地入口 | 使用时注意 |
|---|---|---|
| CoreBench v2 正式文本 | [manifest](new_exp/results/construction_v2/final/corebench_v2_manifest.json)、[数据目录](codes/corebench/generate_json_v2/) | 已有正式导出；数量与来源按 manifest |
| WISE v2 正式文本 | [manifest](new_exp/results/construction_v2/final/wise_v2_manifest.json)、[正式 JSON](codes/wise/wise_generate_v2.json) | 第一轮候选量、语义筛查量和正式入选量是不同概念 |
| CoreBench v2 图像与评分 | [4B](new_exp/results/construction_v2/image_stage/corebench_full_s0/flux4b/)、[9B](new_exp/results/construction_v2/image_stage/corebench_full_s0_flux9b/flux9b/)、[对应说明](new_exp/construction_v2_record.md#flux-full-inference-authorization-2026-09-06) | 核对生成步数与历史 Gemini Checklist 评分口径 |
| WISE v2 正式图像与评分 | [正式输出](new_exp/results/construction_v2/image_stage/wise_formal_full_s0/flux4b/)、[评分汇总](new_exp/results/construction_v2/image_stage/wise_formal_full_s0/flux4b/eval/summary.json) | 正式集与 provisional 集分开；当前为 seed0 单动作，不能视作多动作、多 seed Router 标签齐备 |

WISE 的正式冻结与最终增量出图见 [construction 记录末节](new_exp/construction_v2_record.md#wise-formal-freeze-and-final-flux-delta-2026-09-06)。[WISE 计划](new_exp/construction_v2_record.md#wise-design)保留设计与执行过程，不再用其中的旧进度判断完成状态。

## 目录分工与历史资料

整理前共检出 **164 份项目 Markdown**：根目录 3、writing 5、new_exp 63、exp1_absolute_predict 23、exp_815 15、exp_catimage818 12、codes 39、logs_v2 4。统计排除了虚拟环境、site-packages、`codes/Infinity` 和 `codes/geneval/geneval-main`；`new_exp` 顶层原有 57 份 MD。本轮将24份相关记录合为9份，净减少15份活动MD；原稿存于 [集中归档](docs/records_before_merge_20260911.tar.gz)。

| 位置 | 定位与入口 |
|---|---|
| `new_exp/` | 后续实验主目录；以 [本目录 README](new_exp/README.md) 查记录与结构化结果 |
| `exp1_absolute_predict/` | 早期绝对分审计；[主记录](exp1_absolute_predict/exp1_gm_vs_am.md)、[past 版本](exp1_absolute_predict/exp1_gm_vs_am.md#version-audit)、[分节表格](exp1_absolute_predict/tables_new/)；past版本已合并，三行差异有JSON核对记录 |
| `exp_815/` | 独立的无 TTS 7 档复核；[协议](exp_815/README.md)、[完整汇总](exp_815/exp_815.md)、[分实验表格与备份](exp_815/results/) |
| `exp_catimage818/` | COCO CATImage 复刻；[摘要](exp_catimage818/catimage_results_summary.md)、[分实验表格与备份](exp_catimage818/results/) |
| `codes/results_md/` | 更早的 3/4 档、direct/mixed、AM/GM、CATImage predictor 对照；按 [文件目录](codes/results_md/)追溯 |
| `codes/cross_bench_results/` | 早期跨 benchmark / RF / CATImage 结果；[方法说明](codes/cross_bench_results/method_details.md)、[汇总](codes/cross_bench_results/results_summary.md)，与后期严格 LOBO 区分 |
| `logs_v2/` | 早期 MLP 训练日志与阶段摘要；[阶段总结](logs_v2/summary_next.md)，保留作运行诊断 |
| `writing/`、`总实验要求.md` | 论文分析与设计讨论；引用数字时回到具体实验 |
| `data/`、`new_exp/results/`、`new_exp/figs/` | 标签、结构化结果、缓存与图表；由实验文档链接，不直接迁移 |
| 根目录其他文件 | [计算分配表](compute_allocation_tables.md)是早期专题表；[核对记录](修改日志.md)列出旧日志各项结论、证据与合并映射 |

`backup_*` 是历史协议或修正前结果，不能与父目录现行表格混用。当前项目 `Results/` 主要是旧 COCO 子目录，不是这些新实验的完整结果入口。

## 核对与合并结果

- 旧日志的动作名称、RF实验是否LOBO、数据规模、饱和点表已完成核对；错误项已纠正，无来源的比例停止引用。
- §18–22的旧Ridge行序问题与UniGen旧pool/test重合已确认，相关主张已标明限制；没有用原数字冒充重跑结果。
- 24份文档合为9份，重复摘要、设计/结果分离、准备日志过期等问题已处理；9个自动写文档入口改为按章节更新。
- 详细证据：[实验记录核对与合并](修改日志.md)。

## 后续记录规则

1. 一个实验以完整文件名标识，记录只保留一个主入口；设计、准备日志、图表分别链接回主记录。
2. 新编号使用 `exp57_<主题>` 起，创建前同时查 `run_exp*`、MD 与结果文件，避免只看 MD 导致重号。
3. 每个实验主记录写清：研究问题、数据版本、动作库与成本单位、划分和 seeds、主比较、结论与限制、结果 JSON/脚本/图表路径。
4. 记录状态区分：设计、准备、有结果、待核对、历史；“脚本存在”不等于完成，“没有 MD”不等于没跑。
5. 分数表保留在实验主记录和其结果 JSON；README 只维护主题导航和关键口径，不再复制一份全量大表。
6. 新增合并先保存可恢复归档与映射，再更新脚本写入位置和链接；结果JSON、图片与缓存不随文档迁移。
