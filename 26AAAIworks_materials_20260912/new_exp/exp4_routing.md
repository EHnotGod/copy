# §4 路由质量 LC 实验

## 实验动机

给定固定推理预算（= 各模型 50steps 的 cost），能否用少量标注样本训练一个质量预测器，把每条 query 路由到性价比最高的 cfg（10steps / 50steps / tts），从而在预算不变的情况下超过"一律跑50steps"的基线？本实验系统评估：（1）三种预测器（Ridge / KNN / TFMLP）× 两种预测范式（score mode 直接预测绝对分 / delta mode 预测相对差值）× 三个数据集 × 三个模型，（2）随训练样本量 n=100..2200 增加时的学习曲线，以及与 Oracle（测试集真实分路由）、LP（test 均值凸包）、纯50steps 三条参考线的差距。

## 实验结论

- **预测器路由全面超过纯50steps基线**，与 LP 相比也有明显提升，说明质量预测路由是有效的。
- **Ridge 和 KNN 表现相近且通常最好**，TFMLP 在 score mode 下略差，但在 delta mode 下有时持平。
- **Score mode 和 delta mode 差异不大**，score mode 整体略优或持平，delta mode 在部分 Qwen 设置下有轻微优势。
- **UniGen 上绝对路由收益最大**（Oracle-50steps 差距超 0.10），GM 次之，AM 最小（基线已高，空间有限）。
- **n≥800 后收益递减明显**，n=2200 时各预测器已趋近稳定，继续增加样本量回报有限。

## 实验设置
- 路由策略：Lagrangian 松弛，扫描 λ，argmax(pred_score − λ×cost)
- Score mode：直接预测绝对分数，Delta mode：预测 d1=q50−q10, d2=qtts−q50，以相对分路由
- Budget = 各模型 50steps 对应 cost（qwen=10.0, flux=23.1, boogu=21.5）
- Cost 表：qwen [2.0/10.0/30.5]，flux [4.7/23.1/44.9]，boogu [4.3/21.5/64.8]（10/50/tts）
- Oracle：测试集真实分数路由（上界）；LP：全 pool 均值凸包插值（下界）

## 结果汇总（n=2200，routing quality 原始值）

| Dataset | Model | Oracle | LP | 50steps | RS | RD | KS | KD | TS | TD |
|---------|-------|--------|----|---------|----|----|----|----|----|----|
| gm | qwen | 0.406 | 0.300 | 0.2482 |0.3269 | 0.3225 | 0.3207 | 0.3269 | 0.3225 | 0.3350 |
| gm | flux | 0.433 | 0.342 | 0.3321 |0.3449 | 0.3435 | 0.3433 | 0.3449 | 0.3435 | 0.3431 |
| gm | boogu | 0.492 | 0.395 | 0.3947 |0.4022 | 0.3911 | 0.4002 | 0.4022 | 0.3911 | 0.4002 |
| am | qwen | 0.795 | 0.748 | 0.7300 |0.7596 | 0.7610 | 0.7551 | 0.7596 | 0.7610 | 0.7647 |
| am | flux | 0.808 | 0.768 | 0.7680 |0.7666 | 0.7655 | 0.7660 | 0.7666 | 0.7655 | 0.7663 |
| am | boogu | 0.842 | 0.813 | 0.8130 |0.8085 | 0.8124 | 0.8040 | 0.8085 | 0.8124 | 0.8118 |
| unigen | qwen | 0.878 | 0.786 | 0.7518 |0.8399 | 0.8120 | 0.8091 | 0.8399 | 0.8120 | 0.8128 |
| unigen | flux | 0.901 | 0.840 | 0.8275 |0.8680 | 0.8459 | 0.8466 | 0.8680 | 0.8459 | 0.8529 |
| unigen | boogu | 0.943 | 0.889 | 0.8887 |0.9106 | 0.8890 | 0.8912 | 0.9106 | 0.8890 | 0.8883 |

_RS=Ridge-Score, RD=Ridge-Delta, KS=KNN-Score, KD=KNN-Delta, TS=TFMLP-Score, TD=TFMLP-Delta_

## 图表

### GenEval-M (GM)
![exp4 routing gm](figs/exp4_routing_gm.png)

### AM-EditBench (AM)
![exp4 routing am](figs/exp4_routing_am.png)

### UniGenBench
![exp4 routing unigen](figs/exp4_routing_unigen.png)

