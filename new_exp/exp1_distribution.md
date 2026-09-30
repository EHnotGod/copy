# 实验一：分数分布

**核心命题：** 极端样本（0分 / 满分堆积）决定后续 R² 解读方式。

## 实验设置

| 项目 | 说明 |
|---|---|
| 数据集 | GenEval2 GM / GenEval2 AM / UniGenBench |
| 生成模型 | Qwen-Image-2512（qwen）/ FLUX.2-Klein-9B（flux）/ Boogu-Image（boogu）|
| 配置档位 | 10steps / 50steps / tts（teacher-forcing sampling）|
| 测试集 | geneval2: 800 条（raw_perseed）；unigen: 600 条（raw600_edit_v2）|
| 极端阈值 | 低分：y < 0.05；高分：y > 0.95 |

极端阈值 0.05 / 0.95 与后续 §2 R²_mid 的截断（y ∈ (0.10, 0.90)）略有不同：
分布统计用更宽松的 0.05/0.95 捕捉两端绝对堆积；§2 用更保守的 0.10/0.90 确保去掉
训练信号最强的极端样本后再评估中间区间预测力。

## GenEval2 GM（800 test samples）

![GenEval2 GM 分布](figs/exp1_dist_gm.png)

| model | 10steps mean | frac<0.05 | frac>0.95 | 50steps mean | frac<0.05 | frac>0.95 | tts mean | frac<0.05 | frac>0.95 |
|---|---|---|---|---|---|---|---|---|---|
| qwen | 0.242 | 28.6% | 3.1% | 0.248 | 26.5% | 2.6% | 0.448 | 4.6% | 7.8% |
| flux | 0.316 | 17.9% | 6.0% | 0.332 | 20.8% | 8.5% | 0.372 | 14.0% | 8.0% |
| boogu | 0.355 | 9.4% | 4.6% | 0.395 | 11.2% | 9.5% | 0.426 | 7.5% | 9.0% |

## GenEval2 AM（800 test samples）

![GenEval2 AM 分布](figs/exp1_dist_am.png)

| model | 10steps mean | frac<0.05 | frac>0.95 | 50steps mean | frac<0.05 | frac>0.95 | tts mean | frac<0.05 | frac>0.95 |
|---|---|---|---|---|---|---|---|---|---|
| qwen | 0.722 | 0.0% | 4.6% | 0.730 | 0.0% | 4.5% | 0.815 | 0.0% | 12.4% |
| flux | 0.747 | 0.0% | 9.5% | 0.768 | 0.0% | 11.0% | 0.784 | 0.0% | 11.9% |
| boogu | 0.786 | 0.0% | 7.6% | 0.813 | 0.0% | 14.4% | 0.812 | 0.0% | 13.2% |

## UniGenBench（600 test samples）

![UniGenBench 分布](figs/exp1_dist_unigen.png)

| model | 10steps mean | frac<0.05 | frac>0.95 | 50steps mean | frac<0.05 | frac>0.95 | tts mean | frac<0.05 | frac>0.95 |
|---|---|---|---|---|---|---|---|---|---|
| qwen | 0.747 | 3.2% | 28.7% | 0.752 | 2.5% | 30.7% | 0.886 | 0.8% | 47.5% |
| flux | 0.815 | 2.7% | 40.8% | 0.827 | 2.2% | 43.7% | 0.869 | 0.8% | 47.3% |
| boogu | 0.848 | 1.0% | 42.3% | 0.889 | 1.0% | 57.7% | 0.915 | 0.8% | 59.7% |

## 规律小结

- **GenEval2 GM**：几何均值的压制作用使低分堆积严重（qwen 10/50steps frac<0.05 超过 25%）；
  高分几乎没有堆积（frac>0.95 普遍 < 10%）。
- **GenEval2 AM**：算术均值跨维度补偿，两端堆积均极少（frac<0.05 全部 0%）；
  分数集中在 0.5–0.9 的中间区间。
- **UniGenBench**：任务相对简单，高分堆积严重（boogu tts frac>0.95 超过 55%）；
  低分几乎不存在。
- geneval2 与 unigen 呈镜像关系：前者 0 分多、满分少；后者满分多、0 分极少。
  这一差异决定后续 §2 中 R²(all) 的来源方向。
