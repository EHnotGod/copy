# §43 三个大 Benchmark 的 Router 长尾样本效率曲线

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

完整曲线已在 CPU 重跑，当前分数与 gate 以 [JSON](results/prompt_disjoint_v1/exp43_largebench_router_tail.json) 为准。使用同版本 §38 全量参照和 §40 source300 预测，重新计算目标模型、LP、source 校准与置信区间。

训练候选容量按数据集区分：GM/AM 最大 k2400，UniGen 最大 k2000（fit1800 + safety200）。各点训练样本嵌套，固定 safety 保留区与测试集；不再把 UniGen2000 点标为2400。

下表为 dense / target_only 的末端点；k 包含 safety 标签。


| 评分集 | k | Router−LP | 95% CI | gate通过率 |
|---|---:|---:|---|---:|
| GM | 2400 | +0.015508 | [+0.001247, +0.030912] | 0% |
| AM | 2400 | +0.018556 | [+0.011337, +0.026313] | 100% |
| UniGen | 2000 | +0.008670 | [-0.005382, +0.022432] | 0% |

![修复后的长尾曲线](results/prompt_disjoint_v1/exp43_largebench_router_tail.png)
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 结论

Exp42 的五 benchmark 公共段只到400；本实验把 GM、AM、UniGen 补到
`k={400,600,800,1200,1800,2400}`。结论非常明确：

1. **三个大 benchmark 都需要远多于400条数据才能恢复 full-data Router。** GM、AM、
   UniGen 首次达到90% full-data learned gain 都在 `k=2400` dense；400条分别只有负收益、
   约53%和约19%。
2. **“稳定正收益”和“恢复90%收益”是不同停止点。** AM 在 dense `k=1800` 已稳定通过，
   UniGen 在 `k=1200` 已稳定通过；但二者都要到2400才恢复90%。如果追求尽快上线，不必等到
   90%。
3. **GM 是最难的。** dense `k=2400` 才转为显著正并恢复约99%；但200条 deployment-like
   safety 仍不足以让95%下界稳定大于0，规划值约需375–550条 safety。GM 应继续 LP 回退，
   不能仅因 full test 点估计为正就上线。
4. **成本稀疏训练不能作为三个大 benchmark 的最终替代。** UniGen sparse `k=2400` 能稳定
   获得 `+.0364`，成本仅dense的28.9%，但只恢复55%；AM只恢复71%且通过率60%；GM仍无
   显著收益。
5. **source prior 应随目标数据增多而衰减。** 25% calibrated source prior 在低/中样本常有
   帮助，但 `k=2400` 时对 AM、UniGen 分别降低约 `.0008/.0013`；它应是冷启动 regularizer，
   不是永久固定权重。

## 协议修正

- 三个 pool 均有2400条，`k` 是 fit prompt 与 safety prompt 的总数。
- safety 大小为 `min(200,max(50,round(k/10)))`；因此 `k=2400` 为2200 fit + 200 safety。
- safety 从 **raw deployment distribution** 固定切出最多200条，最终 test 保持固定且独立：
  GM/AM 600条、UniGen 400条。
- 早期试跑发现 generated pool safety 与 raw deployment 分布不一致，尤其 UniGen 在
  `k=2400` 时 pool gate 通过率为0%，raw gate 为100%。因此 pool safety 只保留为 shift
  diagnostic，不能作为上线 gate。
- 5 construction split seeds；公共 A7；预算10；每个动作5 image seeds；Router、LP、source
  calibration、bootstrap gate 均与 Exp42 相同。
- full-data reference 在同一个固定 final-test support 上重算：GM `+.01563`、AM `+.01970`、
  UniGen `+.06666`。

共完成180个 target models、360次 method evaluation；全部有限，最大策略成本10，预算检查
通过。

## Dense 曲线

下表给出 `target-only / source25-calibrated` 的 raw `Router−LP`。括号为 target-only Router
通过 deployment safety gate 的 split 比率。

| k | dense成本 | GM | AM | UniGen |
|---:|---:|---:|---:|---:|
| 400 | 231,310 | −.0056 / −.0026 (0%) | +.0104 / +.0119 (0%) | +.0128 / +.0138 (0%) |
| 600 | 346,965 | −.0026 / −.0010 (20%) | +.0127 / +.0130 (0%) | +.0202 / +.0205 (0%) |
| 800 | 462,620 | −.0035 / −.0016 (0%) | +.0110 / +.0131 (20%) | +.0238 / +.0253 (40%) |
| 1200 | 693,930 | +.0028 / +.0085 (0%) | +.0140 / +.0149 (40%) | +.0338 / +.0335 (**80%**) |
| 1800 | 1,040,895 | +.0049 / +.0075 (0%) | +.0166 / +.0152 (**100%**) | +.0480 / +.0499 (**100%**) |
| 2400 | 1,387,860 | **+.0155 / +.0184** (0%) | **+.0186 / +.0178** (**100%**) | **+.0626 / +.0613** (**100%**) |

`k=2400` 的 target-only raw 95% CI：

- GM `[+.0012,+.0309]`；
- AM `[+.0113,+.0263]`；
- UniGen `[+.0487,+.0775]`。

三者 full test 均显著为正。GM 的 gate 仍未通过，是因为 paired prompt 方差大，而不是均值仍为
负：5个 split 的200条 safety CI 下界都略小于0。

## 成本稀疏曲线

使用 `source25-calibrated`：

| k | sparse成本 | 相对dense节省 | GM raw | AM raw | UniGen raw |
|---:|---:|---:|---:|---:|---:|
| 400 | 74,377 | 67.8% | −.0070 | +.0065 | +.0087 |
| 600 | 104,874 | 69.8% | −.0058 | +.0094 | +.0112 |
| 800 | 139,548 | 69.8% | −.0083 | +.0100 | +.0123 |
| 1200 | 209,524 | 69.8% | +.0003 | +.0119 | +.0198 |
| 1800 | 314,500 | 69.8% | −.0020 | +.0113 | +.0278 |
| 2400 | 401,136 | 71.1% | +.0031 | +.0140 | **+.0364** |

在 `k=2400`：

- GM sparse CI `[−.0088,+.0154]`，不可用；
- AM sparse CI `[+.0076,+.0203]`，恢复71%，gate通过率60%；
- UniGen sparse CI `[+.0254,+.0480]`，恢复55%，gate通过率100%。

所以 sparse 的定位是低成本可用版本或动作筛选阶段。若目标是恢复完整 Router 收益，三个大
benchmark 都仍需 dense 标签。

## 最低停止点

| Benchmark | 首次稳定通过 | 对应成本 | 首次恢复90% raw gain | 对应成本 | 建议 |
|---|---:|---:|---:|---:|---|
| GM | 未达到 | — | dense 2400 | 1,387,860 | full fit 后继续扩大 deployment safety，期间保持 LP |
| AM | dense 1800 | 1,040,895 | dense 2400 | 1,387,860 | 追求尽快上线停1800；追求完整收益到2400 |
| UniGen | dense 1200 | 693,930 | dense 2400 | 1,387,860 | 1200即可安全上线；随后按边际收益决定是否扩到2400 |

若只要求“稳定正收益”而不要求90%恢复，UniGen 还可选择 sparse 2400，成本401,136、收益
`+.0364`；它比 dense 1200 少42.2%成本，但收益略高且训练数据更多、昂贵动作标签更少。

## 对 Router Factory 的修正

1. 公共冷启动配方仍可从100–400条 probe 开始，但大 benchmark 不能把400当训练完成。
2. `source blend` 应设为随目标样本量衰减的超参，例如从25%逐步退到0；不能固定永久使用。
3. safety 必须来自线上目标分布。由生成池构造训练数据时，需要额外保留 raw/traffic-like
   validation；训练分布内的 holdout 无法替代 deployment gate。
4. 停止策略应有两个档位：
   - `deploy-ready`：gate稳定通过即可上线；
   - `quality-complete`：达到 full reference 的90%才停止继续扩训练数据。
5. GM 类高方差 benchmark 需要把 safety 成本单独规划；当前估计 dense target-only 约需550
   条，使用25% source prior约需375条，明显大于统一200条上限。

## 产物

- 脚本：`run_exp43_largebench_router_tail.py`
- 结果：[results/exp43_largebench_router_tail.json](results/exp43_largebench_router_tail.json)
- 曲线图：[results/exp43_largebench_router_tail.png](results/exp43_largebench_router_tail.png)
- 180个 prediction caches 已保存。

</details>
