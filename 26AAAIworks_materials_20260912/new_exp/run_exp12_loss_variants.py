#!/usr/bin/env python3
"""
§12 Loss Variants for Routing（gm/qwen only）

对比 4 种训练 loss，用轻量 MLP（mean-pool CLIP → 2-layer FC）训练，秒级完成。
路由评估统一用 Lagrangian routing quality。

predictor: MLP-3（直接预测 3 个绝对分值，routing = pred_mat）
losses:
  mae        : MSELoss on 3 scores（baseline）
  listnet    : ListNet ranking loss（softmax cross-entropy on permutation prob）
  pairwise   : Pairwise margin loss（hinge on all pairs）
  wtmae      : Weighted MSE（按 max_delta 加权，delta 越大越重要）
"""
import os, sys, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE  = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(HERE, "..", "codes")
sys.path.insert(0, CODES)
import catimage_tf_mlp as TF

DATA  = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
CLIP_DIR = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/zhouwei/clip-vit-large-patch14"
CFGS  = ["10steps", "50steps", "tts"]
COSTS = np.array([2.0, 10.0, 30.5])
BUDGET = 10.0
SEEDS  = [0, 1, 2]
N_TRAIN = 2200
N_VAL   = 200

LAMBDAS = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000), np.linspace(0.05, 3.0, 700), [10., 1e6]]))


# ── 数据加载 ──────────────────────────────────────────────────
def load_scores(path):
    d = json.load(open(path))
    instrs, scores = [], {c: [] for c in CFGS}
    for x in d:
        sc = x["scores"]; vals = {}; ok = True
        for c in CFGS:
            v = sc.get(c)
            if v is None: ok = False; break
            if isinstance(v, dict): v = v.get("avg")
            if v is None: ok = False; break
            vals[c] = float(v)
        if not ok: continue
        instrs.append(x["instruction"])
        for c in CFGS: scores[c].append(vals[c])
    return instrs, {c: np.array(scores[c], np.float32) for c in CFGS}


# ── Lagrangian routing quality ────────────────────────────────
from collections import defaultdict
def route_quality(pred_mat, true_mat):
    pts = []
    for lam in LAMBDAS:
        adj = pred_mat - lam * COSTS[None, :]
        chosen = np.argmax(adj, axis=1)
        pts.append((float(np.mean(COSTS[chosen])),
                    float(np.mean(true_mat[np.arange(len(true_mat)), chosen]))))
    bucket = defaultdict(float)
    for c, q in pts:
        bucket[round(c, 6)] = max(bucket[round(c, 6)], q)
    xs = np.array(sorted(bucket.keys()))
    qs = np.array([bucket[x] for x in xs])
    if BUDGET <= xs[0]: return float(qs[0])
    if BUDGET >= xs[-1]: return float(qs[-1])
    idx = int(np.searchsorted(xs, BUDGET))
    c0, c1 = xs[idx-1], xs[idx]; q0, q1 = qs[idx-1], qs[idx]
    return float(q0 + (BUDGET - c0) / (c1 - c0) * (q1 - q0))


# ── 轻量 MLP ─────────────────────────────────────────────────
class LightMLP(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)


# ── Loss 函数 ─────────────────────────────────────────────────
def listnet_loss(pred, true):
    """ListNet: KL(softmax(true/T) || softmax(pred))，T=0.1 使 true 更尖锐"""
    T = 0.1
    p_true = F.softmax(true / T, dim=1)
    log_p_pred = F.log_softmax(pred, dim=1)
    return -(p_true * log_p_pred).sum(dim=1).mean()

def pairwise_loss(pred, true, margin=0.05):
    """Pairwise hinge：对所有 (i,j) pair，若 true_i > true_j 则 pred_i 应 > pred_j + margin"""
    loss = torch.tensor(0.0, device=pred.device)
    n_pairs = 0
    for i in range(3):
        for j in range(3):
            if i == j: continue
            delta_true = true[:, i] - true[:, j]   # 正 = i 应该比 j 好
            delta_pred = pred[:, i] - pred[:, j]
            # 只对 true delta 明显的 pair 施加约束（避免噪声 pair 干扰）
            mask = delta_true.abs() > 0.01
            if mask.sum() == 0: continue
            hinge = F.relu(margin - delta_true[mask] * delta_pred[mask])
            loss = loss + hinge.mean()
            n_pairs += 1
    return loss / max(n_pairs, 1)

def weighted_mse_loss(pred, true):
    """Weighted MSE：每个样本权重 = max(delta) + 0.1，delta 大的 query 更重要"""
    max_delta = true.max(dim=1).values - true.min(dim=1).values  # (B,)
    w = (max_delta + 0.1).detach()
    w = w / w.mean()   # normalize
    err = ((pred - true) ** 2).mean(dim=1)   # (B,)
    return (w * err).mean()


# ── 训练函数 ──────────────────────────────────────────────────
def train_mlp(Xtr, Ytr, Xva, Yva, Xte,
              loss_name="mae", seed=0, device="cuda:0",
              max_epochs=200, lr=3e-4, batch=128, patience=20):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    net = LightMLP(in_dim=Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    def to_t(a): return torch.from_numpy(np.asarray(a, np.float32)).to(dev)
    Xtr_t, Ytr_t = to_t(Xtr), to_t(Ytr)
    Xva_t, Yva_t = to_t(Xva), to_t(Yva)
    Xte_t = to_t(Xte)
    n = len(Xtr_t)

    loss_fn = {"mae": nn.MSELoss(),
               "listnet": listnet_loss,
               "pairwise": pairwise_loss,
               "wtmae": weighted_mse_loss}[loss_name]

    best_va, best_state, best_ep, bad = 1e9, None, -1, 0
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch):
            idx = perm[s:s+batch]
            opt.zero_grad()
            pred = net(Xtr_t[idx])
            loss = loss_fn(pred, Ytr_t[idx])
            loss.backward(); opt.step()
        # val early stop: 用与训练相同的 loss（排序/加权 loss 下 MSE 会立即升高，导致 ep=0 就停）
        net.eval()
        with torch.no_grad():
            va_pred = net(Xva_t)
            if loss_name == "mae":
                va_loss = F.mse_loss(va_pred, Yva_t).item()
            else:
                va_loss = loss_fn(va_pred, Yva_t).item()
        if va_loss < best_va - 1e-6:
            best_va, best_ep, bad = va_loss, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred_te = net(Xte_t).cpu().numpy()
    return pred_te, best_ep


# ── 主实验 ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    print("=== §12 Loss Variants: gm/qwen ===")
    p1 = f"{DATA}/geneval2/edit/geneval2_qwen_edit_gen_perseed.json"
    p2 = f"{DATA}/geneval2/edit/geneval2_qwen_edit_gen2000_perseed.json"
    i1, s1 = load_scores(p1); i2, s2 = load_scores(p2)
    pool_instrs = i1 + i2
    pool_scores = {c: np.concatenate([s1[c], s2[c]]) for c in CFGS}

    te_instrs, yte_cfg = load_scores(f"{DATA}/geneval2/edit/geneval2_qwen_edit_raw_perseed.json")
    n_te = len(te_instrs)
    true_mat = np.stack([yte_cfg[c] for c in CFGS], axis=1)
    oracle = route_quality(true_mat, true_mat)
    always50 = float(yte_cfg["50steps"].mean())
    print(f"  oracle={oracle:.4f}  50steps={always50:.4f}  test={n_te}")

    # CLIP features（mean pool）
    Fpool, Mpool = TF.clip_token_features(pool_instrs, "exp4_gm_qwen_pool", CLIP_DIR, dev)
    Fte,   Mte   = TF.clip_token_features(te_instrs,   "exp4_gm_qwen_test", CLIP_DIR, dev)

    def mean_pool(F, M):
        m = M[:, :, None]
        return ((F * m).sum(1) / np.clip(m.sum(1), 1e-6, None))

    Xpool = mean_pool(Fpool, Mpool)   # (N, 768)
    Xte_np = mean_pool(Fte, Mte)

    rng_fix = random.Random(42)
    all_idx = list(range(len(pool_instrs))); rng_fix.shuffle(all_idx)
    ext_val = np.array(all_idx[:N_VAL]); lc_pool = all_idx[N_VAL:]
    Xva = Xpool[ext_val]
    Yva = np.stack([pool_scores[c][ext_val] for c in CFGS], 1)

    LOSSES = ["mae", "listnet", "pairwise", "wtmae"]
    results = {l: [] for l in LOSSES}

    for seed in SEEDS:
        rng = random.Random(seed)
        sub = np.array(rng.sample(lc_pool, min(N_TRAIN, len(lc_pool))))
        Xtr = Xpool[sub]
        Ytr = np.stack([pool_scores[c][sub] for c in CFGS], 1)

        row = []
        for loss_name in LOSSES:
            pred, ep = train_mlp(Xtr, Ytr, Xva, Yva, Xte_np,
                                 loss_name=loss_name, seed=seed, device=dev)
            rq = route_quality(pred, true_mat)
            results[loss_name].append(rq)
            row.append(f"{loss_name}={rq:.4f}(ep{ep})")
        print(f"  seed={seed}  " + "  ".join(row), flush=True)

    print("\n=== 汇总（3 seeds 均值±std）===")
    print(f"  oracle={oracle:.4f}  50steps={always50:.4f}")
    for loss_name in LOSSES:
        vals = results[loss_name]
        print(f"  {loss_name:<10}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"vs_mae={np.mean(vals)-np.mean(results['mae']):+.4f}")

    out_path = os.path.join(HERE, "results", "exp12_loss_variants.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({
        "dataset": "gm", "model": "qwen",
        "oracle": oracle, "always_50steps": always50,
        **{f"{l}_mean": float(np.mean(results[l])) for l in LOSSES},
        **{f"{l}_std":  float(np.std(results[l]))  for l in LOSSES},
        "per_seed": {l: [float(x) for x in results[l]] for l in LOSSES},
    }, open(out_path, "w"), indent=2)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
