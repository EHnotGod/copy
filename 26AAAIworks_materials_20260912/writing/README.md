# 写作入口：当前证据与结论边界

2026-09-12。正文引用优先使用实验主记录页首的修复结果及其 JSON。这里整理文章结构和论证边界，不另维护一套分数表。[项目导航](../项目导航.md) · [实验目录](../new_exp/README.md) · [核对记录](../修改日志.md)。

## 当前适合的主线

研究问题是：给定计算预算，prompt 预测器在什么条件下能把生成动作之间的质量差异转化为实际路由收益？依次讨论预测指标、动作库选择、已知域联合监督、未见域迁移和数据需求。

目前证据支持“路由收益依赖动作空间和监督条件”。不要把“预测器都学不会”“收益都来自 benchmark 过拟合”或“delta 普遍更好”当作文章前提。特别是 Joint 监督与 LOBO 留出目标任务回答的是不同问题。

## 主张与证据对应

| 拟写内容 | 当前可写到什么程度 | 实验入口 |
|---|---|---|
| 分数预测与路由收益 | R²、mid-R²、固定预算质量需要分别评估；中段 R² 下降本身不能证明路由无效或确定其原因 | [早期绝对分审计](../exp1_absolute_predict/exp1_gm_vs_am.md)、[修复后的 Ridge](../new_exp/exp18_22_9cfg.md#corrected-results) |
| Delta 目标 | 已有 Ridge 对照差异很小，不能宣称普遍提升；区分预测目标变化与预算分配变化 | [Delta 对照](../exp1_absolute_predict/exp_delta_routing.md)、[原主张核对](../修改日志.md) |
| 动作库规模 | Oracle 空间扩大不保证当前 Router 收益扩大；固定加入顺序的增量不是动作固有价值 | [§24](../new_exp/exp24_action_growth.md#corrected-results)、[§31](../new_exp/exp31_action_subset_selection.md#corrected-results) |
| 选小动作库 | GM A10 有正证据；清洗后的 UniGen 在 A10 与 A7 上方向不同。共同库未检出损失不等于通过非劣检验 | [§35–36](../new_exp/exp36_action_selection_crossbench.md#corrected-results) |
| 低标签量动作入库 | 决策针对当前库与目标指标；单次筛选和三次筛选集成的标签成本分别计算 | [§37](../new_exp/exp37_action_admission_crossbench.md#corrected-results) |
| 已知域 Joint | Flux3 单模型接近各域 specialist；相对 LP 的正证据集中于 UniGen/CoreBench/WISE，不能把均值略高写成全面显著优势 | [§51](../new_exp/exp51_joint_supervised_router_flux3.md#corrected-results) |
| 未见任务迁移 | Flux3 raw-score LOBO 当前主表没有显著超过 Target LP 的评分集；A7 pooled/nested 仍存在损失 | [§52](../new_exp/exp52_flux3_pooled_lobo.md#corrected-results)、[§39](../new_exp/exp39_pooled_router_lobo.md#corrected-results)、[§40](../new_exp/exp40_nested_router_mix_grid.md#corrected-results) |
| 训练量增加 | Flux3 小幅改善不等于显著收益；固定支持的 Flux4 曲线到每源1000仍低于 LP。没有通用的1600条饱和阈值 | [§53–54](../new_exp/exp53_flux3_public_router_data_ablation.md#corrected-results)、[§55–56](../new_exp/exp55_56_flux4_pooled_router.md#corrected-results)、[GM扩量](../new_exp/exp45_geneval2_4000_training_curve.md)、[AM扩量](../new_exp/exp46_geneval2_am_4000_training_curve.md) |
| 图像辅助路由 | 现有 GM 结果可作为受限探索；需要区分图像信息增益与完整计费净收益，跨域验证暂缓 | [§30](../new_exp/exp30_qwen35_2b_probe_router.md)、[§41](../new_exp/exp41_conditional_2b_preview.md) |
| v2 新数据 | 两套正式文本与部分单配置图像评测已完成；尚无完整多动作、多seed Router 标签，不作为新 Router 验证 | [construction-v2](../new_exp/construction_v2_record.md) |

## 正文与附录建议

1. 正文先固定动作库、成本、LP 所见标签与测试支持，再给预测和路由对照，避免把不同协议的质量数字连成一条趋势。
2. 动作选择用 §31、§35–37 说明收益与库、评分目标的关系；预先固定选择规则，测试集用于最后评价。
3. 将 Joint 与 LOBO 分开呈现。GM/AM 共享同一 prompt 来源；当前“五项评分”只有四个 prompt 来源。GM/AM 的留出不能表述为两个独立新域验证。
4. 数据量主图使用 §56 固定 test 与 Target LP1000 的四点曲线；§55 的原协议另列。GM/AM 的质量平台与稳定 gate 单独讨论。
5. 早期 KNN/TFMLP、图像 preview 和新数据构建可放附录或探索部分，明确哪些结果没有在本轮完整重验。

文中“显著”只指对应实验已有的95%区间；没有跨独立数据域的总体显著性证明，也没有把所有配置比较做成多重检验校正。测试集上挑出的最佳配置只用于诊断，不能同时作为无偏部署收益。

## 历史草稿

本目录其余五份 MD 是早期方案或局部分析，保留供追溯：`ours_goal.md`、`ours_work.md`、`router_gain_analysis.md`、`router_failure_analysis.md`、`acc_vs_audc_geneval2.md`。其中不同模型步数、成本单位、动作集合和旧划分不能自动套入当前实验。

用户已暂缓标签补齐和 Hope；当前写作可以基于已核对的 CPU 结果收拢，不预写尚未完成实验的结论。本地核验与复现见 [操作说明](../docs/local_reproduction.md)。
