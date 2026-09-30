# §7 类别内打乱消融实验

## 实验动机
predictor 路由超过 50steps 基线的原因有两种可能：
（1）学到了 prompt 级别的难度排序信号（真正有价值）；
（2）学到了训练集的类别级别分布偏置（可能虚高）。
通过类别内打乱（intra-shuffle）破坏 prompt 级信号但保留类别级分布，
与全局打乱（full-shuffle）和正常训练（real）对比，定位信号来源。

## 实验设置
- n=2200，score mode only，3 seeds，所有预测器
- **Real**：正常训练，predictor 能学到 per-prompt 难度
- **Intra-shuffle**：类别内随机打乱质量标签（position/count/other），破坏 prompt 级信号，保留类别级分布（GM/AM 有；UniGen 无结构类别跳过）
- **Full-shuffle**：全局随机打乱，彻底破坏所有信号
- GM/AM 类别划分：关键词匹配 instruction（position / count / other）

## 实验结论

## 数值汇总

| Dataset | Model | Oracle | 50steps | Ridge-Real | Ridge-Full | KNN-Real | KNN-Full | TF-Real | TF-Full |
|---------|-------|--------|---------|------------|------------|----------|----------|---------|--------|
| gm | qwen | 0.4058 | 0.2482 | 0.3269 | 0.2951 | 0.3225 | 0.2948 | 0.3207 | 0.2951 |
| gm | flux | 0.4335 | 0.3321 | 0.3449 | 0.3368 | 0.3435 | 0.3419 | 0.3433 | 0.3402 |
| gm | boogu | 0.4921 | 0.3947 | 0.4022 | 0.3777 | 0.3911 | 0.3787 | 0.4002 | 0.3874 |
| am | qwen | 0.7953 | 0.7300 | 0.7596 | 0.7444 | 0.7610 | 0.7466 | 0.7551 | 0.7462 |
| am | flux | 0.8077 | 0.7680 | 0.7666 | 0.7644 | 0.7655 | 0.7657 | 0.7660 | 0.7678 |
| am | boogu | 0.8416 | 0.8130 | 0.8085 | 0.8002 | 0.8124 | 0.8093 | 0.8040 | 0.8130 |
| unigen | qwen | 0.8782 | 0.7518 | 0.8399 | 0.7873 | 0.8120 | 0.7846 | 0.8091 | 0.7866 |
| unigen | flux | 0.9012 | 0.8275 | 0.8680 | 0.8348 | 0.8459 | 0.8364 | 0.8466 | 0.8399 |
| unigen | boogu | 0.9430 | 0.8887 | 0.9106 | 0.8817 | 0.8890 | 0.8858 | 0.8912 | 0.8875 |

## 图表

### GenEval-M (GM)
![exp7 shuffle gm](figs/exp7_shuffle_gm.png)

### AM-EditBench (AM)
![exp7 shuffle am](figs/exp7_shuffle_am.png)

### UniGenBench
![exp7 shuffle unigen](figs/exp7_shuffle_unigen.png)

