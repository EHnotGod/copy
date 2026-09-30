# 计算方式详细说明

## 1. 问题定义

**目标**：给定 N 个 prompt，总计算预算 = 50×N 步，每个 case 必须分配到三档之一：

| 档位 | key | 实际步数（cost） |
|------|-----|-----------------|
| 10步 | `10steps` | 10 |
| 50步 | `50steps`（内存别名 `40steps`） | 50 |
| tts  | `tts` | 100 |

**约束**：`10·p₁₀ + 50·p₅₀ + 100·p_tts = 50`，即平均 cost = 50。

---

## 2. 数据划分

### gen 训练模式（表2）

- **训练集**：gen 库中随机取 300 条（seed 0/1/2 各取一次，取平均）
- **验证集**：同次随机再取 100 条（与训练不重叠）
- **测试集**：raw 库全量

对于 coco：raw 库共 2000 条，按 `seed=20260723` shuffle 后取前 400 条入 train pool，后 1600 条作测试集。

### 全量训练模式（表3）

- **训练集**：geneval2/corebench/wise/unigen 各取 400 条 gen 数据 + coco 400 条（共 2000 条混合）
- **测试集**：各 bench 的 raw 库全量
- seed 0/1/2 同样各跑一次取均值

---

## 3. Predictor 实现

### 3.1 KNN（CATImage 原版）

1. 用 frozen CLIP-ViT-L/14 text encoder 对每条 prompt 提取 77 token × 768 dim per-token embedding
2. 对有效 token 做 attention mask 均值池化，得到每条 prompt 的 CLIP pooled embedding（D 维）
3. StandardScaler 归一化
4. 在训练集上建 KNN（k = min(50, N_train//3)），对测试集每条找 k 个最近邻
5. **预测分数 `rs_knn[i]`**：k 个邻居在三档上真实分数的均值（shape: 3）

### 3.2 TFMLP（CATImage 原版）

**模型结构**（照搬论文 Appendix D）：

```
frozen CLIP text encoder
  → (N, 77, 768) per-token embedding + attention mask (N, 77)
  → Linear(768→512) + 可学习 position embedding (1, 77, 512)
  → 2层 TransformerEncoderLayer (d_model=512, nhead=8, ffn=1024, dropout=0.1, gelu)
  → per-token 2层 MLP head: Linear(512→512)+GELU+Linear(512→M)
  → attention mask 加权 average pooling over 77 tokens
  → (N, M) logits  →  Sigmoid  →  (N, M) 预测分数
```

其中 M=3（对应三档质量分）。

**训练**：
- 目标：把真实分数线性归一到 [0,1] 作为 BCE 的"概率目标"
  `y_norm = (y - y.min()) / (y.max() - y.min())`
- 损失：`BCEWithLogitsLoss`（sum over M heads，mean over batch）
- 优化器：Adam，lr=1e-3，weight_decay=1e-4
- 早停：val BCE 不改善 patience=30 epoch 后停，最多 max_epochs=300
- **CLIP 权重全程冻结**，只训 attn+MLP 部分

**推理**：`sigmoid(net(test_feat, test_mask))` → `rs_tf[i]` ∈ (0,1)³

> 注意：TFMLP 预测值是归一化空间内的分数，用于 λ-sweep 时和 COSTS 的相对大小关系不变，结果不影响路由决策。

### 3.3 RF（自研对比方法）

1. 用 Qwen3-Embedding-8B 对 prompt 编码，得到 4096 维 embedding
2. RandomForest（300 棵树）分别回归三档分数
3. **预测分数 `rs_rf[i]`**：RF 输出的三档预测值（shape: 3）

---

## 4. λ-sweep 路由框架（CATImage 核心）

### 4.1 理论背景

CATImage 的最优路由规则（Lagrangian 松弛）：

```
r*(x) = argmax_m [ γ̂(m)(x) - λ · c(m) ]
```

其中 γ̂(m)(x) 是对 case x 在第 m 档的预测质量分，c(m) 是该档 cost，λ 是拉格朗日乘子。

- λ 大 → 偏向低 cost 档（更多 10步）
- λ 小 → 偏向高 cost 档（更多 tts）
- λ 扫过 (−∞, +∞) → 遍历所有可能的 (avg_cost, avg_quality) 组合，形成 deferral curve

### 4.2 sweep_pts 实现

```python
lambdas = np.concatenate([np.linspace(-0.005, 0.05, 8000), [0.1, 0.5, 1, 100, -1]])
# 覆盖范围：分数 0~1，cost 10~100，转折 λ 量级约 1e-2；极端值 ±∞ 兜住两端

def sweep_pts(gt, rs, costs, lams):
    # gt:  (N, 3) 真实分数（用于计算实际 avg_quality）
    # rs:  (N, 3) predictor 预测分数（用于做路由决策）
    # costs: [10, 50, 100]
    n = rs.shape[0]; pts = []
    for lam in lams:
        ri = np.argmax(rs - lam * costs.reshape(1,-1), axis=1)  # 每条 case 选档，返回索引 {0,1,2}
        pts.append((
            float(np.mean(costs[ri])),           # 该 λ 下的平均 cost
            float(np.mean(gt[np.arange(n), ri])), # 该 λ 下的平均真实质量分
            lam
        ))
    pts.sort(key=lambda x: x[0])  # 按 avg_cost 排序
    return pts  # [(avg_cost, avg_quality, lam), ...]
```

**关键**：路由决策用 `rs`（predictor 预测），但质量评估用 `gt`（真实分数）。oracle 时 `rs == gt`。

### 4.3 @ B=50 分数计算

```python
def best_at_budget(pts, budget=50.0):
    # 找所有 avg_cost ≤ 50 的点，取其中 avg_cost 最大的（最充分利用预算）
    cands = [(c, q, lam) for c, q, lam in pts if c <= budget + 1e-6]
    best = max(cands, key=lambda x: x[0])
    return best[1], best[2]  # (avg_quality, λ*)
```

逻辑：在满足"平均不超预算"的前提下，尽量多花预算（λ* 尽量小），对应 deferral curve 上 avg_cost≤50 最右侧的点的 quality 值。

### 4.4 AUDC 计算

```python
def area_under_deferral_curve(costs, rewards, cmin=10, cmax=100):
    # 在固定横轴 [10, 100] 上做归一化梯形积分
    sweep = np.linspace(cmin, cmax, 1000)
    interp = np.interp(sweep, costs, rewards)  # 端点外用端点值
    return np.trapz(interp, sweep) / (cmax - cmin)
```

AUDC 是对所有预算值（10到100）的平均性能，不依赖某个固定预算点，反映 predictor 在整个预算区间上的路由质量。

---

## 5. Oracle 定义

Oracle 用**真实分数**代替 predictor 预测分数做 λ-sweep，代表已知每条 case 真实质量时的最优路由：

```python
oracle_pts = sweep_pts(y_test, y_test, COSTS, lambdas)  # rs == gt
oracle_q50, oracle_lam = best_at_budget(oracle_pts, budget=50.0)
oracle_route = np.argmax(y_test - oracle_lam * COSTS.reshape(1,-1), axis=1)  # 索引 {0,1,2}
```

Oracle 是理论上限——知道每条 case 的真实分数后能达到的最优质量分。

---

## 6. ACC 计算

ACC = predictor 路由与 oracle 路由的逐 case 吻合率：

```python
# predictor 侧：找自己的 λ*，得到路由索引
pred_pts = sweep_pts(y_test, rs_pred, COSTS, lambdas)
pred_q50, pred_lam = best_at_budget(pred_pts, budget=50.0)
route_pred = np.argmax(rs_pred - pred_lam * COSTS.reshape(1,-1), axis=1)  # {0,1,2}

# oracle 侧：同样用 λ-sweep（真实分数做 predictor）
oracle_route = np.argmax(y_test - oracle_lam * COSTS.reshape(1,-1), axis=1)  # {0,1,2}

ACC = np.mean(route_pred == oracle_route) * 100  # %
```

**注意**：
- oracle 和 predictor 各自独立用 λ-sweep 找自己的 λ*，再做 argmax；双方返回的都是档位索引 {0,1,2}，可以直接对比
- ACC 反映的是"predictor 的路由决策和最优路由的一致性"，不是分数本身的预测精度
- ACC 高不一定 @50 分高（路由一致但 predictor 预测的分与真值有偏差也算 ACC 高）