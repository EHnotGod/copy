# new_exp 实验目录

[返回项目入口](../项目导航.md)。整理日期：2026-09-11。

本目录按研究问题组织合并后的主记录；24份相关记录已合为9份，旧路径映射见 [核对记录](../修改日志.md)。链接到 MD 表示已有叙述记录，链接到 JSON 表示已有结构化结果；不据此推断集群实时状态。具体数字与有效口径以实验主记录及产物为准。旧UniGen存在400条pool/test重合，§18–22旧Ridge存在行序错误，详见 [数据核对](../修改日志.md#数据与对齐核对)。

> **修复后的结果**：优先阅读各实验页首的“修复后的当前结果”，对应数据在 `results/prompt_disjoint_v1/`。标签补齐与 Hope 已于 2026-09-12 暂缓。[写作入口](../writing/README.md)与[本地复现](../docs/local_reproduction.md)已补齐。完成范围见 [修复清单](../修改日志.md#corrected-results)。

## 1. 预测能力与路由基础：§1–10

| 实验 | 研究问题 / 记录 |
|---|---|
| §1 | [分数分布](exp1_distribution.md) |
| §2 | [R²(all/mid) 与 delta-R²](exp2_delta_r2.md) |
| §2b | [Ridge-only 预测对照](exp2b_ridge_only.md) |
| §2c | [Ridge-only 路由质量](exp2c_ridge_rq.md) |
| §3 | [预测学习曲线](exp3_lc.md) |
| §4 | [路由质量学习曲线](exp4_routing.md) |
| §5 | [生成 seed 数敏感性](exp5_seed_sens.md) |
| §6 | [Mid-pool / All-pool](exp6_mid_pool.md) |
| §7 | [类别内打乱消融](exp7_shuffle.md) |
| §8 | [预测指标与路由收益](exp8_metrics.md) |
| §9 | [Train/Val 划分](exp9_val_split.md) |
| §10 | [DP 路由与目标变体](exp10_dp_routing.md) |

[ALL_EXPERIMENTS](ALL_EXPERIMENTS.md) 已合并原 notables 的seed/R²补充，保留原图；范围仍为 **§1–8历史汇总**。

§5–8 另有同号脚本，不能只靠编号判断主题：

| 脚本 | 主题 |
|---|---|
| [run_exp5_routing_variants.py](run_exp5_routing_variants.py) | 路由变体，与 seed 数实验不同 |
| [run_exp6_dp_variants.py](run_exp6_dp_variants.py) | DP 变体，与 mid-pool 实验不同 |
| [run_exp7_softrq.py](run_exp7_softrq.py) | Soft-RQ，与 shuffle 实验不同 |
| [run_exp8_variants.py](run_exp8_variants.py) | 方法变体，与指标相关性实验不同 |

## 2. 统一预算与训练方法探索：§11–17

| 实验 | 记录 / 产物 | 说明 |
|---|---|---|
| §11 统一预算 | [summary](exp11/summary.md)、[GM](exp11/gm.md)、[AM](exp11/am.md)、[UniGen](exp11/unigen.md) | 单模型内动作路由，统一 budget=10；运行入口 [run_exp11_unified_budget.py](run_exp11_unified_budget.py) |
| §11 Soft-DP 脚本 | [run_exp11_softdp.py](run_exp11_softdp.py) | 同号的另一探索入口，不与统一预算表混用 |
| §12 | [loss variants JSON](results/exp12_loss_variants.json)、[脚本](run_exp12_loss_variants.py) | 有结果，未见独立同名 MD |
| §13 | [target variants JSON](results/exp13_target_variants.json)、[脚本](run_exp13_target_variants.py) | 有结果，未见独立同名 MD |
| §14 | [routing tricks JSON](results/exp14_routing_tricks.json)、[脚本](run_exp14_routing_tricks.py) | 有结果，未见独立同名 MD |
| §15 | [calibration JSON](results/exp15_calib.json)、[脚本](run_exp15_calib.py) | 有结果，未见独立同名 MD |
| §16 | [new ideas JSON](results/exp16_new_ideas.json)、[脚本](run_exp16_new_ideas.py) | 有结果，未见独立同名 MD |
| §17 | [DP targets JSON](results/exp17_dp_targets.json)、[脚本](run_exp17_dp_targets.py) | 有结果，未见独立同名 MD |

## 3. 跨模型动作池：§18–24

**原 exp11_9cfg 与 §18–22 五个分记录已合并。** 五项旧 Ridge 均确认行序错误；§18–22 Ridge 均已有修正后的 CPU 重跑，旧分数折叠保留。

| 实验 | 记录 | 阅读说明 |
|---|---|---|
| §18–22 主记录 | [五项实验与核对](exp18_22_9cfg.md) | 历史9动作组；各分节保留原协议与明细 |
| §18 | [9-cfg 路由](exp18_22_9cfg.md#exp18) | 完整 Ridge 曲线已重跑，UniGen 最大 train1800 |
| §19 | [9-cfg seed 敏感性](exp18_22_9cfg.md#exp19) | Ridge 已修复重跑，详见主记录页首 |
| §20 | [9-cfg mid-pool](exp18_22_9cfg.md#exp20) | Ridge 已修复重跑，详见主记录页首 |
| §21 | [9-cfg 类别内打乱](exp18_22_9cfg.md#exp21) | Ridge 已修复重跑，详见主记录页首 |
| §22 | [9-cfg 划分比例](exp18_22_9cfg.md#exp22) | Ridge 已修复重跑，详见主记录页首 |
| §23 | [A9 → A10，加入蒸馏动作](exp23_10action_distill.md) | 包含旧 §18 显式对齐复核 |
| §24 | [从 Qwen50 逐步扩动作池](exp24_action_growth.md) | 动作库增长与收益来源 |

## 4. 跨域与结果图辅助路由：§25–30、§41

| 实验 | 结果主记录 | 设计 / 承接 |
|---|---|---|
| §25 | [跨数据集泛化](exp25_cross_dataset_generalization.md) | [设计](exp25_cross_dataset_generalization.md#exp25-design)；五 benchmark 扩展见 §38–40 |
| §26 | [Probe 图评价 embedding](exp26_probe_image_router.md) | [设计与成本口径](exp26_probe_image_router.md#exp26-design) |
| §27 | [QEval 图像因果对照](exp27_qeval_causal_controls.md) | 区分图像新增信息与其他因素 |
| §28 | [Conditional probe](exp28_conditional_probe.md) | 条件触发探索 |
| §29 | [低分辨率 QEval](exp29_qeval_lowres.md) | 信息与计算量权衡 |
| §30 | [Qwen3.5-2B 图文 Router](exp30_qwen35_2b_probe_router.md) | 信息增益与完整计费净收益分别报告 |
| §41 | [普通 Router 不确定度触发 2B preview](exp41_conditional_2b_preview.md) | GM、3 splits 的条件预览结果；不能外推为五域已验证 |

## 5. 动作选择、入库与检索：§31–37

| 实验 | 记录 | 说明 |
|---|---|---|
| §31 | [动作库全子集选择](exp31_action_subset_selection.md) | Validation 选择与 held-out test |
| §32 | [动作入库预筛选](exp32_action_prescreen.md) | 历史 GM/AM 回溯，本轮未重跑；跨域现行结果见 §37 |
| §33 | [Text-only RAG preview](exp33_rag_preview.md) | 探索分支 |
| §34 | [Result-aware RAG preview](exp34_result_aware_rag_preview.md) | 探索分支 |
| §35 | [§35–36 合并记录](exp36_action_selection_crossbench.md)、[§35 JSON](results/exp35_action_subset_crossbench.json) | 没有单独 `exp35_*.md`，不是编号缺失 |
| §36 | [跨 benchmark 动作库选择](exp36_action_selection_crossbench.md) | A10 与公共 A7；脚本 [run_exp36_common_a7_crossbench.py](run_exp36_common_a7_crossbench.py) |
| §37 | [新动作快速入库](exp37_action_admission_crossbench.md) | 含小样本门槛、成本摊销和顺序筛选 |

## 6. 公共 Router 泛化：§38–40、§50–56

| 实验 | 记录 / 产物 | 实验边界 |
|---|---|---|
| §38 | [五 benchmark 5×5 泛化](exp38_router_crossbench_generalization.md) | A7，单 source → target；GM↔AM 是同 prompt 的评分聚合变化 |
| §39 | [Pooled LOBO](exp39_pooled_router_lobo.md) | A7，目标 benchmark 完全留出 |
| §40 | [Nested-LOBO 混合网格](exp40_nested_router_mix_grid.md) | source-only 选择与外层验证 |
| §50 | [A7 Joint JSON](results/exp50_joint_supervised_router.json)、[共享脚本](run_exp50_joint_supervised_router.py) | 结果说明并入 §51 的 A7 敏感性补充 |
| §51 | [Flux3 Joint](exp51_joint_supervised_router_flux3.md) | 所有已知 benchmark 都参与监督；用 §50 脚本的 `--action-space flux3` |
| §52 | [Flux3 LOBO](exp52_flux3_pooled_lobo.md) | 完整留出目标域；与 Joint 不同 |
| §53 | [公共 Router 数据条件消融](exp53_flux3_public_router_data_ablation.md) | source 覆盖、平衡、静态先验、弱 residual |
| §54 | [数据曲线 JSON](results/exp54_flux3_pooled_router_data_curve.json)、[脚本](run_exp54_flux3_pooled_router_data_curve.py) | 每源 100/200/300；叙述在 §53“消融二” |
| §55 | [Flux4 数据量曲线](exp55_56_flux4_pooled_router.md#corrected-results) | 增加 4B 动作后，未见域误路由 |
| §56 | [Flux4 每源 1,000 条](exp55_56_flux4_pooled_router.md#corrected-results) | 已补固定 test、LP1000 的100/200/300/1000曲线；旧§55/56仍分开保留 |

## 7. 样本效率、稳定性与扩量：§42–49

| 实验 | 主记录 | 配套 / 阅读说明 |
|---|---|---|
| §42 | [Router Factory 样本效率](exp42_router_factory_sample_efficiency.md) | 五 benchmark 公共段到 400，dense/sparse 与 source reuse |
| §43 | [大 benchmark 长尾曲线](exp43_largebench_router_tail.md) | GM/AM/UniGen 扩到 2400；调整 safety 协议 |
| §44 | [训练量 × safety 稳定性](exp44_router_stability_surface.md) | GM，独立 raw400 safety / raw400 test |
| §45 主实验 | [GM 2400→4000](exp45_geneval2_4000_training_curve.md) | 判断相对 LP 的收益与稳定 gate |
| §45 质量图 | [GM 200→4000 统一质量曲线](exp45_geneval2_4000_training_curve.md#exp45-quality) | 固定 raw400 重评估；不是旧表直接拼接 |
| §45 准备 | [数据准备与复现](exp45_geneval2_4000_training_curve.md#exp45-preparation) | 复现过程、标签补齐与实际 pipeline 参数；最终结论看主实验 |
| §46 | [AM 200→4000](exp46_geneval2_am_4000_training_curve.md) | 从已有 score_list 恢复扩展 AM；无需把“没有独立 AM 文件”理解成没有可用分数 |
| §47 | [WISE/CoreBench Flux-only 扩量](exp47_49_flux_only_extension.md#exp47) | 历史 raw 重划数据方案，与 construction-v2 新文本不同 |
| §48 | [WISE/CoreBench/UniGen 扩展汇总](exp47_49_flux_only_extension.md#exp48) | UniGen 运行脚本为 [run_exp48_flux_only_unigen_extension.py](run_exp48_flux_only_unigen_extension.py) |
| §49 | [WISE 扩到 1200](exp47_49_flux_only_extension.md#exp49) | 同样区分重划 raw 与生成新的独立 construction |

## 8. 数据与过程记录

| 记录 | 用途 | 有效状态去哪里核对 |
|---|---|---|
| [五 benchmark 基础数据盘点](base_model_data_inventory_5bench.md) | 模型 × 动作 × split × seed 的存量清单 | 历史图片存量，AM可恢复标签与UniGen交集已复核 |
| [construction-v2 主记录](construction_v2_record.md) | CoreBench/WISE 文本构建、审核、冻结、图片阶段 | 正式状态表及 [CoreBench manifest](results/construction_v2/final/corebench_v2_manifest.json)、[WISE manifest](results/construction_v2/final/wise_v2_manifest.json) |
| [WISE v2 计划](construction_v2_record.md#wise-design) | 预览、配额、作者/审核隔离与早期执行过程 | 设计已并入construction主记录，早期执行快照集中归档 |
| [已有 construction 审计](results/construction_audit/README.md) | 旧池审计 | 与正式 v2 选择结果分别保存 |
| [WISE 20 条预览](results/construction_v2/wise_preview20/wise_preview20.md) | 风格与内容审阅样本 | 预览不自动等于正式入选集 |

## 9. 比较实验前先统一口径

- **动作空间**：早期单模型 3 档、A9、A10、公共 A7、Flux3、Flux4 不能只按动作数量类比。`exp_815` 的 7 档不含 TTS，后期公共 A7 含 Qwen/Flux TTS，二者不是同一个库。
- **数据域与评分**：GM/AM 使用相同 GenEval2 prompt 和图片的不同聚合分，不是两个完全独立 prompt 域；文中“五 benchmark”沿用原实验命名。
- **预算与指标**：quality@固定预算、AUDC、R²、ACC、Router−LP 是不同指标；历史 CATImage 用 TFLOPs，后期常用归一化 budget=10，不能直接比较数字。
- **测试支持**：GenEval2 raw800 与 raw400、WISE/CoreBench 全量 raw 与重划小 test 要分别注明。§56 的扩量改变了小 benchmark 测试支持。
- **模型训练方式**：Joint 使用所有已知域监督；LOBO 留出目标域；source/pooled LP 与 target-seen LP 使用的标签信息也不同。
- **数据完成程度**：文本完成、某模型某 seed 出图完成、评分完成、整个动作库标签交集齐全是不同状态。WISE v2 的 binary judge 与历史 WISE 分数要分开。
- **主张强度**：正点估计、置信区间显著、跨 split 稳定、跨 benchmark 泛化逐层区分；图像信息增益另需核对 probe 与 evaluator 是否完整计费。

新增实验时把主记录登记到相应主题中，具体数字留在该主记录及其 JSON；不要继续向旧 `ALL_EXPERIMENTS*` 追加全量复制表。
