#!/usr/bin/env python3
"""
§8  路由/损失变体（gm/qwen + gm/flux）

B: 样本重加权 Ridge   — 训练时给决策边界附近样本高权重
C: Multi-λ Soft-RQ   — 在 λ 区间内平均 soft-RQ loss，修复 §7 单点 λ* 退化
D: Pairwise SVM      — 对 10vs50 / 50vstts 各训练一个线性 SVM，decision_function 做路由
E: Isotonic 校准     — Ridge 预测后对每个 cfg 做 isotonic 单调回归校准

所有方法均用 DP 路由（exp10），val DP-RQ 选超参。
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC
from sklearn.isotonic import IsotonicRegression
import warnings; warnings.filterwarnings("ignore")

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
COSTS   = {"qwen": np.array([2.0,  10.0, 30.5]),
           "flux": np.array([4.7,  23.1, 44.9])}
BUDGETS = {"qwen": 10.0, "flux": 23.1}
RIDGE_ALPHAS = [0.01, 0.1, 1., 10., 100., 1000., 1e4]
SEEDS   = [0, 1, 2]
N_POOL  = 2200
N_EXT_VAL = 200

LAMBDAS = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000),
    np.linspace(0.05, 0.3, 500),
    np.linspace(0.3, 3.0, 200),
    [10., 100., 1e6],
]))


# ── 数据 / DP-RQ ──────────────────────────────────────────────
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

def dp_rq(pred_mat, true_mat, costs, budget):
    return knapsack_dp_route(pred_mat, true_mat, costs, budget)


# ── λ* ────────────────────────────────────────────────────────
def find_lam_star(Ytr, costs, budget):
    best_lam, best_dist = 0.0, float('inf')
    for lam in LAMBDAS:
        chosen = np.argmax(Ytr - lam * costs[None, :], axis=1)
        dist = abs(float(np.mean(costs[chosen])) - budget)
        if dist < best_dist: best_dist = dist; best_lam = lam
    return float(best_lam)


# ── Ridge utils ───────────────────────────────────────────────
def fit_ridge_j(Xtr, ytr, Xva, yva, sample_weight=None):
    best_sc, best_m = -1e18, None
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, ytr, sample_weight=sample_weight)
        sc = r2_score(yva, m.predict(Xva))
        if sc > best_sc: best_sc = sc; best_m = m
    return best_m

def predict_mat(models, X):
    return np.stack([m.predict(X) for m in models], 1)


# ── B: 加权 Ridge ─────────────────────────────────────────────
def run_B(Xtr, Ytr, Xva, Yva, Xte, true_va, costs, budget, lam_star):
    adj = Ytr - lam_star * costs[None, :]
    s   = np.sort(adj, axis=1)
    margin = s[:, -1] - s[:, -2]        # best - second-best under λ*

    best_rq_va, best_pred_te, best_hp = -1e18, None, None
    for eps_pct in [5, 15]:
        for alpha_w in [0.5, 1.0, 2.0]:
            eps_w = np.percentile(margin, eps_pct)
            w = 1.0 / (margin + eps_w) ** alpha_w
            w /= w.mean()

            ms_va = [fit_ridge_j(Xtr, Ytr[:, j], Xva, Yva[:, j], w) for j in range(3)]
            rq_va = dp_rq(predict_mat(ms_va, Xva), true_va, costs, budget)
            if rq_va > best_rq_va:
                best_rq_va = rq_va; best_hp = (eps_pct, alpha_w)
                # refit same α already chosen above, predict test
                ms_te = [fit_ridge_j(Xtr, Ytr[:, j], Xva, Yva[:, j], w) for j in range(3)]
                best_pred_te = predict_mat(ms_te, Xte)

    return dp_rq(best_pred_te, None, costs, budget), best_hp

def run_B_full(Xtr, Ytr, Xva, Yva, Xte, true_va, true_te, costs, budget, lam_star):
    adj = Ytr - lam_star * costs[None, :]
    s   = np.sort(adj, axis=1)
    margin = s[:, -1] - s[:, -2]

    best_rq_va, best_pred_te, best_hp = -1e18, None, None
    for eps_pct in [5, 15]:
        for alpha_w in [0.5, 1.0, 2.0]:
            eps_w = np.percentile(margin, eps_pct)
            w = 1.0 / (margin + eps_w) ** alpha_w
            w /= w.mean()
            ms = [fit_ridge_j(Xtr, Ytr[:, j], Xva, Yva[:, j], w) for j in range(3)]
            rq_va = dp_rq(predict_mat(ms, Xva), true_va, costs, budget)
            if rq_va > best_rq_va:
                best_rq_va = rq_va; best_hp = (eps_pct, alpha_w)
                best_pred_te = predict_mat(ms, Xte)

    return dp_rq(best_pred_te, true_te, costs, budget), best_hp


# ── C: Multi-λ Soft-RQ ───────────────────────────────────────
def train_multilam(Xtr_t, Ytr_t, lam_star, costs_t, lam_max,
                   tau, l2, K=15, epochs=800, lr=5e-3):
    lams = torch.linspace(0, lam_max, K, dtype=torch.float32)
    W = torch.zeros(3, Xtr_t.shape[1], requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(epochs):
        pred = Xtr_t @ W.T + b
        loss = torch.tensor(0.)
        for lam in lams:
            adj = pred - lam * costs_t[None, :]
            wt  = F.softmax(adj / tau, dim=1)
            loss = loss - (wt * Ytr_t).sum(1).mean()
        loss = loss / K + 0.5 * l2 * (W ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (Xtr_t @ W.T + b).numpy(), W.detach(), b.detach()

def run_C_full(Xtr, Ytr, Xva, Yva, Xte, true_va, true_te, costs, budget, lam_star):
    lam_max = float(max(0.03, 5 * lam_star))
    costs_t  = torch.tensor(costs, dtype=torch.float32)
    Xtr_t    = torch.tensor(Xtr, dtype=torch.float32)
    Ytr_t    = torch.tensor(Ytr, dtype=torch.float32)
    Xva_t    = torch.tensor(Xva, dtype=torch.float32)
    Xte_t    = torch.tensor(Xte, dtype=torch.float32)
    true_va_t= torch.tensor(Yva, dtype=torch.float32)

    best_rq_va, best_pred_te, best_hp = -1e18, None, None
    for tau in [0.1, 0.2, 0.5]:
        for l2 in [0.1, 1.0, 5.0]:
            _, W, b = train_multilam(Xtr_t, Ytr_t, lam_star, costs_t,
                                      lam_max, tau, l2)
            with torch.no_grad():
                pred_va = (Xva_t @ W.T + b).numpy()
                pred_te = (Xte_t @ W.T + b).numpy()
            rq_va = dp_rq(pred_va, Yva, costs, budget)
            if rq_va > best_rq_va:
                best_rq_va = rq_va; best_hp = (tau, l2)
                best_pred_te = pred_te

    return dp_rq(best_pred_te, true_te, costs, budget), best_hp


# ── D: Pairwise SVM ──────────────────────────────────────────
def run_D_full(Xtr, Ytr, Xva, Yva, Xte, true_va, true_te, costs, budget, lam_star):
    # 两个二元决策的标签
    thr1 = lam_star * (costs[1] - costs[0])   # 10 vs 50
    thr2 = lam_star * (costs[2] - costs[1])   # 50 vs tts
    y1 = np.where(Ytr[:, 1] - Ytr[:, 0] > thr1, 1, -1)
    y2 = np.where(Ytr[:, 2] - Ytr[:, 1] > thr2, 1, -1)

    best_rq_va, best_pred_te, best_hp = -1e18, None, None
    for C_val in [0.001, 0.01, 0.1, 1.0, 10.0]:
        try:
            svm1 = LinearSVC(C=C_val, max_iter=3000, dual=True).fit(Xtr, y1)
            svm2 = LinearSVC(C=C_val, max_iter=3000, dual=True).fit(Xtr, y2)
        except Exception:
            continue
        df1_va = svm1.decision_function(Xva)
        df2_va = svm2.decision_function(Xva)
        pred_va = np.stack([-df1_va, np.zeros(len(Xva)), df2_va], 1)
        rq_va = dp_rq(pred_va, Yva, costs, budget)
        if rq_va > best_rq_va:
            best_rq_va = rq_va; best_hp = C_val
            df1_te = svm1.decision_function(Xte)
            df2_te = svm2.decision_function(Xte)
            best_pred_te = np.stack([-df1_te, np.zeros(len(Xte)), df2_te], 1)

    return dp_rq(best_pred_te, true_te, costs, budget), best_hp


# ── E: Isotonic 校准 ──────────────────────────────────────────
def run_E_full(ridge_pred_va, ridge_pred_te, Yva, true_te, costs, budget):
    calib = np.empty_like(ridge_pred_te)
    for j in range(3):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(ridge_pred_va[:, j], Yva[:, j])
        calib[:, j] = iso.predict(ridge_pred_te[:, j])
    return dp_rq(calib, true_te, costs, budget)


# ── 单次 seed ─────────────────────────────────────────────────
def run_one(model, seed, emb_pool, pool_scores, emb_te, true_te,
            costs, budget, ext_val, lc_pool):
    rng = random.Random(seed)
    sub = np.array(rng.sample(lc_pool, min(N_POOL, len(lc_pool))))

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(emb_pool[sub])
    Xtr = sc.transform(emb_pool[sub]).astype(np.float32)
    Xva = sc.transform(emb_pool[ext_val]).astype(np.float32)
    Xte = sc.transform(emb_te).astype(np.float32)

    Ytr = np.stack([pool_scores[c][sub]     for c in CFGS], 1).astype(np.float32)
    Yva = np.stack([pool_scores[c][ext_val] for c in CFGS], 1).astype(np.float32)

    lam_star = find_lam_star(Ytr, costs, budget)

    # Ridge baseline（也供 E 用）
    ms_ridge = [fit_ridge_j(Xtr, Ytr[:, j], Xva, Yva[:, j]) for j in range(3)]
    ridge_pred_va = predict_mat(ms_ridge, Xva)
    ridge_pred_te = predict_mat(ms_ridge, Xte)
    rq_ridge = dp_rq(ridge_pred_te, true_te, costs, budget)

    # B
    rq_B, hp_B = run_B_full(Xtr, Ytr, Xva, Yva, Xte, Yva, true_te, costs, budget, lam_star)
    # C
    rq_C, hp_C = run_C_full(Xtr, Ytr, Xva, Yva, Xte, Yva, true_te, costs, budget, lam_star)
    # D
    rq_D, hp_D = run_D_full(Xtr, Ytr, Xva, Yva, Xte, Yva, true_te, costs, budget, lam_star)
    # E
    rq_E = run_E_full(ridge_pred_va, ridge_pred_te, Yva, true_te, costs, budget)

    return dict(ridge=rq_ridge, B=rq_B, C=rq_C, D=rq_D, E=rq_E,
                lam_star=lam_star, hp_B=hp_B, hp_C=hp_C, hp_D=hp_D)


# ── 单模型主流程 ──────────────────────────────────────────────
def run_model(model):
    p1 = f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen_perseed.json"
    p2 = f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen2000_perseed.json"
    tp = f"{DATA}/geneval2/edit/geneval2_{model}_edit_raw_perseed.json"

    i1, s1 = load_scores(p1); i2, s2 = load_scores(p2)
    pool_scores = {c: np.concatenate([s1[c], s2[c]]) for c in CFGS}
    n_pool = len(i1) + len(i2)

    _, yte   = load_scores(tp)
    n_te     = len(yte[CFGS[0]])
    true_te  = np.stack([yte[c] for c in CFGS], 1)

    costs  = COSTS[model]; budget = BUDGETS[model]
    oracle   = dp_rq(true_te, true_te, costs, budget)
    always50 = float(np.mean(yte["50steps"]))

    emb_pool = np.concatenate([get_emb(p1), get_emb(p2)], 0)
    emb_te   = get_emb(tp)[:n_te]

    rng_fix = random.Random(42)
    all_idx = list(range(n_pool)); rng_fix.shuffle(all_idx)
    ext_val = np.array(all_idx[:N_EXT_VAL])
    lc_pool = all_idx[N_EXT_VAL:]

    keys = ["ridge", "B", "C", "D", "E"]
    acc  = {k: [] for k in keys}
    all_hps = []

    for seed in SEEDS:
        r = run_one(model, seed, emb_pool, pool_scores, emb_te, true_te,
                    costs, budget, ext_val, lc_pool)
        for k in keys: acc[k].append(r[k])
        all_hps.append({"lam_star": r["lam_star"], "hp_B": r["hp_B"],
                         "hp_C": r["hp_C"], "hp_D": r["hp_D"]})
        row = "  ".join(f"{k}={r[k]:.4f}" for k in keys)
        print(f"  gm/{model} seed={seed}  {row}  λ*={r['lam_star']:.4f}", flush=True)

    print(f"\n  gm/{model}  oracle={oracle:.4f}  50steps={always50:.4f}")
    baseline = float(np.mean(acc["ridge"]))
    for k in keys:
        m = float(np.mean(acc[k]))
        tag = f"vs_ridge={m-baseline:+.4f}" if k != "ridge" else ""
        print(f"    {k:<7}: {m:.4f}  gain50={m-always50:+.4f}  {tag}")
    print()

    return {
        "model": model, "oracle": oracle, "always_50steps": always50,
        **{f"{k}_mean": float(np.mean(acc[k])) for k in keys},
        **{f"{k}_seeds": [float(x) for x in acc[k]] for k in keys},
        "hps": all_hps,
    }


def main():
    print("=== §8 路由/损失变体（gm/qwen + gm/flux）===\n")
    results = []
    for model in ["qwen", "flux"]:
        results.append(run_model(model))
    out = os.path.join(HERE, "results", "exp8_variants.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
