# §44 GM Router 训练量 × Safety 数据量稳定性面

## 协议

- 复用 §43 dense target-only Router；训练规模 `k={400,600,800,1200,1800,2400}`。
- 官方 raw800 固定拆成400条 nested safety reserve和400条最终测试；二者完全不交叉。
- safety `n={50,100,150,200,250,300,350,400}`；5个 construction seeds。
- 稳定上线：最终测试 paired CI 下界>0、5/5 split点估计>0、且至少4/5 safety gate通过。
- sustained 阈值要求该点及其后所有更大 safety n 都满足，避免偶然穿越。

## 结果

表中为 `certified splits / 5`；`*` 表示达到稳定上线规则。

| train k | test gain [95% CI] | safety 50 | safety 100 | safety 150 | safety 200 | safety 250 | safety 300 | safety 350 | safety 400 | sustained safety n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | -0.0137 [-0.0267,-0.0010] | 1/5 | 2/5 | 0/5 | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 | — |
| 600 | -0.0057 [-0.0200,+0.0080] | 0/5 | 2/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | — |
| 800 | -0.0042 [-0.0181,+0.0097] | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | — |
| 1200 | -0.0018 [-0.0167,+0.0128] | 0/5 | 0/5 | 0/5 | 0/5 | 1/5 | 1/5 | 1/5 | 1/5 | — |
| 1800 | +0.0017 [-0.0131,+0.0172] | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | — |
| 2400 | +0.0138 [-0.0033,+0.0309] | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 1/5 | 0/5 | — |

## 结论

- 当前 `train≤2400, safety≤400` 网格仍没有满足预注册稳定上线规则的组合。
- 最终测试固定为400条，因此不同 safety n 的 test gain 完全可比；变化只来自 gate 证据量。
- 本实验回答验证样本量；训练量超过2400的部分将在新增1600条A7标签完成后接入同一协议。

![Exp44 stability surface](results/exp44_router_stability_surface.png)

## 产物

- 脚本：`run_exp44_router_stability_surface.py`
- 结果：[results/exp44_router_stability_surface.json](results/exp44_router_stability_surface.json)
- 图：[results/exp44_router_stability_surface.png](results/exp44_router_stability_surface.png)

## 承接实验

- §45 使用新增1600条互斥 construction 数据，把训练曲线扩展到 `k=4000`：
  `exp45_geneval2_4000_training_curve.md`。
- 标签生成、评分和输入审计的时间线：`exp45_geneval2_4000_training_curve.md#exp45-preparation`。
