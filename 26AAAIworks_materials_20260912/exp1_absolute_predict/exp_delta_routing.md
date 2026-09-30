# 实验：delta predictor 路由 vs 绝对分 predictor 路由

**问题：** 把训练目标从"预测绝对分"换成"预测 cfg 间相对质量差（delta）"，路由效果会不会更好？

**背景：** `compare_metrics.py` 的结果显示，预测绝对分的 R²（R2_all / R2_mid）与路由 recovery 几乎不相关（rho=-0.017 / -0.400），但预测 cfg 间 delta 的能力（delta-R2 / delta-Spearman）与 recovery 强相关（rho=+0.917 / +0.933）。据此猜想：改用预测 delta 作为路由信号，可能提升路由质量。

---

## 方法

**abs predictor（对照）：** 对每个 cfg 独立训练一个 Ridge，预测绝对分，接 Lagrangian routing。等同于 exp6 的做法。

**delta predictor（实验）：** 对每对有序 cfg 组合 (i, j)，训练 Ridge 预测 `y_i - y_j`。路由时构造伪分数：
```
score_i = sum_{j≠i} pred(y_i - y_j)
```
再接同样的 Lagrangian routing。

其余设置完全对齐 exp6：pool 2400 条，每个 seed 独立随机切 2200/200（训练/val），3 seeds 均值，Lagrangian lambda sweep。

脚本：`run_exp_delta_routing.py`，结果：[results_11/exp_delta_routing.json](results_11/exp_delta_routing.json)。

---

## 结果

| 组 | oracle | LP | abs routing | delta routing | rec_abs | rec_delta | delta−abs |
|---|---|---|---|---|---|---|---|
| 组 | oracle | LP | abs | delta | rec_abs | rec_delta | delta−abs |
|---|---|---|---|---|---|---|---|
| qwen / GM     | 0.4476 | 0.3321 | 0.3616 | 0.3617 | +25.6% | +25.7% | +0.0001 |
| qwen / AM     | 0.8136 | 0.7645 | 0.7786 | 0.7785 | +28.7% | +28.5% | -0.0001 |
| qwen / UniGen | 0.9037 | 0.8208 | 0.8613 | 0.8614 | +48.9% | +48.9% | +0.0000 |
| flux / GM     | 0.4331 | 0.3426 | 0.3444 | 0.3445 | +2.0%  | +2.1%  | +0.0001 |
| flux / AM     | 0.8074 | 0.7621 | 0.7648 | 0.7647 | +6.0%  | +5.8%  | -0.0001 |
| flux / UniGen | 0.9012 | 0.8439 | 0.8695 | 0.8694 | +44.6% | +44.5% | -0.0001 |
| boogu / GM    | 0.5019 | 0.3923 | 0.4109 | 0.4110 | +17.0% | +17.0% | +0.0001 |
| boogu / AM    | 0.8458 | 0.8130 | 0.8090 | 0.8090 | **-12.3%** | **-12.1%** | +0.0001 |
| boogu / UniGen| 0.9432 | 0.8887 | 0.9132 | 0.9133 | +45.0% | +45.0% | +0.0000 |

---

## 结论

**delta predictor 和 abs predictor 路由效果几乎完全一样。** 9 组里 delta−abs 全在 ±0.0001 以内，3-seed std 约 0.001~0.003，差值远在噪声范围内。boogu_am 仍为负（-12.3% vs -12.1%）。

**为什么没有差别：** Ridge 是线性模型。预测 `y_i - y_j` 等价于先分别预测 `y_i`、`y_j` 再相减——两种方式在线性框架下学出来的决策边界是一样的，路由结果自然相同。

**delta-R² 与 recovery 的强相关（rho=0.917）描述的是哪些组"路由容易做"，不是说换了 delta 目标就能把那些组做得更好。** 信号强弱是数据本身的性质，不是目标函数的问题。

**如果要在此方向继续试：** 换非线性模型（MLP / GBT），才可能让"直接预测 delta"和"分别预测再相减"之间产生实质差距。但从 §2 的 R²_mid 来看，中间区间的信号本身就很弱，换模型可能也有限。
