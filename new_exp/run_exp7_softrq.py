#!/usr/bin/env python3
"""
§7  Soft-RQ 直接优化（线性模型，无 TFMLP）

loss = -E_i[ Σ_j softmax_τ((Wx+b)[j] - λ*·cost_j) · true[i,j] ]

λ* 在训练集真实分上求解：令 avg_cost(argmax(true - λ·cost)) ≈ budget 的 λ
超参：τ ∈ {0.05,0.1,0.2,0.5,1.0}，L2 ∈ {0.01,0.05,0.1,0.5,1.0,5.0}
       联合以 val DP-RQ 选最优
对比：Ridge-Score（exp10 基线）
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from collections import defaultdict

HERE  = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(HERE, "..", "codes")
sys.path.insert(0, CODES)
from embed_lr_dp import load_cached_embedding
sys.path.insert(0, HERE)
from run_exp10_dp_routing import knapsack_dp_route

DATA  = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
CACHE = os.path.join(CODES, ".cache_embed")
EMB_MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/zhaobo45/CKPTS/Qwen3-Embedding-8B"

CFGS    = ["10steps", "50steps", "tts"]
COSTS   = {"qwen": np.array([2.0, 10.0, 30.5]),
           "flux": np.array([4.7, 23.1, 44.9])}
BUDGETS = {"qwen": 10.0, "flux": 23.1}
ALPHAS  = [0.01, 0.1, 1., 10., 100., 1000., 1e4]   # Ridge
TAUS    = [0.05, 0.1, 0.2, 0.5, 1.0]
L2S     = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
SEEDS   = [0, 1, 2]
N_POOL  = 2200
N_EXT_VAL = 200
N_EPOCHS  = 1500
LR        = 5e-3

LAMBDAS = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000),
    np.linspace(0.05, 0.3, 500),
    np.linspace(0.3, 3.0, 200),
    [10., 100., 1e6],
]))


# ── 数据 ──────────────────────────────────────────────────────
def load_scores(path):
    d = json.load(open(path)); instrs, scores = [], {c: [] for c in CFGS}
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

def get_emb(path):
    e = load_cached_embedding(CACHE, path, EMB_MODEL)
    if e is None: raise RuntimeError(f"cache miss: {path}")
    return e


# ── DP 路由质量 ────────────────────────────────────────────────
def dp_rq(pred_mat, true_mat, costs, budget):
    return knapsack_dp_route(pred_mat, true_mat, costs, budget)


# ── Ridge baseline ────────────────────────────────────────────
def fit_ridge(Xtr, Ytr, Xva, Yva, Xte):
    preds = []
    for j in range(3):
        best_sc, best_m = -1e18, None
        for a in ALPHAS:
            m = Ridge(alpha=a).fit(Xtr, Ytr[:, j])
            sc = r2_score(Yva[:, j], m.predict(Xva))
            if sc > best_sc: best_sc = sc; best_m = m
        preds.append(best_m.predict(Xte))
    return np.stack(preds, 1)


# ── λ* 求解 ───────────────────────────────────────────────────
def find_lam_star(true_mat, costs, budget):
    """训练集真实分上令 avg_cost ≈ budget 的 λ"""
    best_lam, best_dist = 0.0, float('inf')
    for lam in LAMBDAS:
        adj    = true_mat - lam * costs[None, :]
        chosen = np.argmax(adj, axis=1)
        avg_c  = float(np.mean(costs[chosen]))
        dist   = abs(avg_c - budget)
        if dist < best_dist:
            best_dist = dist; best_lam = lam
    return float(best_lam)


# ── Soft-RQ 训练 ──────────────────────────────────────────────
def train_softrq(Xtr_t, Ytr_t, lam_star, costs_t, tau, l2,
                 n_epochs=N_EPOCHS, lr=LR, device="cpu"):
    in_dim = Xtr_t.shape[1]
    W = torch.zeros(3, in_dim, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)

    for ep in range(n_epochs):
        pred = Xtr_t @ W.T + b          # (N, 3)
        adj  = pred - lam_star * costs_t[None, :]
        wt   = F.softmax(adj / tau, dim=1)
        loss = -(wt * Ytr_t).sum(dim=1).mean() + 0.5 * l2 * (W ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()

    return W.detach(), b.detach()


def predict_softrq(W, b, Xte_t):
    with torch.no_grad():
        return (Xte_t @ W.T + b).cpu().numpy()


# ── 单次 seed ─────────────────────────────────────────────────
def run_one(model, seed, emb_pool, pool_scores, emb_te, true_mat,
            costs, budget, ext_val, lc_pool):
    rng = random.Random(seed)
    sub = np.array(rng.sample(lc_pool, min(N_POOL, len(lc_pool))))

    sc = StandardScaler().fit(emb_pool[sub])
    Xtr = sc.transform(emb_pool[sub]).astype(np.float32)
    Xva = sc.transform(emb_pool[ext_val]).astype(np.float32)
    Xte = sc.transform(emb_te).astype(np.float32)

    Ytr = np.stack([pool_scores[c][sub]     for c in CFGS], 1).astype(np.float32)
    Yva = np.stack([pool_scores[c][ext_val] for c in CFGS], 1).astype(np.float32)
    true_va = Yva

    # Ridge baseline
    ridge_pred = fit_ridge(Xtr, Ytr, Xva, Yva, Xte)
    rq_ridge   = dp_rq(ridge_pred, true_mat, costs, budget)

    # λ* from training set
    lam_star = find_lam_star(Ytr, costs, budget)

    # tensors
    device   = "cpu"
    costs_t  = torch.tensor(costs, dtype=torch.float32, device=device)
    Xtr_t    = torch.tensor(Xtr, device=device)
    Ytr_t    = torch.tensor(Ytr, device=device)
    Xva_t    = torch.tensor(Xva, device=device)
    Xte_t    = torch.tensor(Xte, device=device)
    true_va_t= torch.tensor(true_va, device=device)

    # grid search (τ, L2) by val DP-RQ
    best_rq_va, best_hp, best_Wb = -1e18, None, None
    for tau in TAUS:
        for l2 in L2S:
            W, b = train_softrq(Xtr_t, Ytr_t, lam_star, costs_t,
                                 tau=tau, l2=l2, device=device)
            pred_va = predict_softrq(W, b, Xva_t)
            rq_va   = dp_rq(pred_va, true_va, costs, budget)
            if rq_va > best_rq_va:
                best_rq_va = rq_va; best_hp = (tau, l2); best_Wb = (W, b)

    W, b = best_Wb
    pred_te = predict_softrq(W, b, Xte_t)
    rq_softrq = dp_rq(pred_te, true_mat, costs, budget)

    return {"ridge": rq_ridge, "softrq": rq_softrq,
            "best_tau": best_hp[0], "best_l2": best_hp[1],
            "lam_star": lam_star}


# ── 单模型主流程 ──────────────────────────────────────────────
def run_model(model):
    p1 = f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen_perseed.json"
    p2 = f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen2000_perseed.json"
    tp = f"{DATA}/geneval2/edit/geneval2_{model}_edit_raw_perseed.json"

    i1, s1 = load_scores(p1); i2, s2 = load_scores(p2)
    pool_scores = {c: np.concatenate([s1[c], s2[c]]) for c in CFGS}
    n_pool = len(i1) + len(i2)

    _, yte  = load_scores(tp)
    n_te    = len(yte[CFGS[0]])
    true_mat= np.stack([yte[c] for c in CFGS], 1)

    costs  = COSTS[model]; budget = BUDGETS[model]
    oracle   = dp_rq(true_mat, true_mat, costs, budget)
    always50 = float(np.mean(yte["50steps"]))

    emb_pool = np.concatenate([get_emb(p1), get_emb(p2)], 0)
    emb_te   = get_emb(tp)[:n_te]

    rng_fix = random.Random(42)
    all_idx = list(range(n_pool)); rng_fix.shuffle(all_idx)
    ext_val = np.array(all_idx[:N_EXT_VAL])
    lc_pool = all_idx[N_EXT_VAL:]

    ridge_rqs, softrq_rqs, hps = [], [], []
    for seed in SEEDS:
        r = run_one(model, seed, emb_pool, pool_scores, emb_te, true_mat,
                    costs, budget, ext_val, lc_pool)
        ridge_rqs.append(r["ridge"]); softrq_rqs.append(r["softrq"])
        hps.append((r["best_tau"], r["best_l2"]))
        print(f"  gm/{model} seed={seed}  ridge={r['ridge']:.4f}  "
              f"softrq={r['softrq']:.4f}  "
              f"τ={r['best_tau']}  L2={r['best_l2']}  λ*={r['lam_star']:.4f}",
              flush=True)

    ridge_m  = float(np.mean(ridge_rqs))
    softrq_m = float(np.mean(softrq_rqs))
    print(f"\n  gm/{model}  oracle={oracle:.4f}  50steps={always50:.4f}")
    print(f"    Ridge-Score : {ridge_m:.4f}  gain={ridge_m-always50:+.4f}")
    print(f"    Soft-RQ     : {softrq_m:.4f}  gain={softrq_m-always50:+.4f}  "
          f"vs_ridge={softrq_m-ridge_m:+.4f}\n")

    return {
        "model": model, "oracle": oracle, "always_50steps": always50,
        "ridge_mean": ridge_m, "ridge_seeds": ridge_rqs,
        "softrq_mean": softrq_m, "softrq_seeds": softrq_rqs,
        "best_hps": hps,
    }


def main():
    print("=== §7 Soft-RQ vs Ridge-Score（gm/qwen + gm/flux）===\n")
    results = []
    for model in ["qwen", "flux"]:
        results.append(run_model(model))
    out = os.path.join(HERE, "results", "exp7_softrq.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
