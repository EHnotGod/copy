#!/usr/bin/env python3
"""
§11  统一 Budget=10 路由实验

Budget 固定为 10（= qwen 50steps cost），所有 9 个 dataset/model 组合统一使用。
复刻 exp4~exp8 的全部方法：
  - Ridge-Score / Ridge-Delta
  - KNN-Score / KNN-Delta
  - TFMLP-Score / TFMLP-Delta
  - Soft-RQ (§7)
  - Multi-λ Soft-RQ (§8C)
  - Weighted Ridge (§8B)
  - Pairwise SVM (§8D)
  - Isotonic 校准 (§8E)

LC 曲线：n = 100 / 300 / 600 / 1000 / 1500 / 2200
路由：Lagrangian（主） + DP oracle 对比
参考：oracle_lag / oracle_dp / lp / always_50steps

用法：
  python run_exp11_unified_budget.py --shard gm/qwen --gpu 0
  python run_exp11_unified_budget.py --merge          # 汇总+写 md
"""
import os, sys, json, random, argparse, tempfile, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC
from sklearn.isotonic import IsotonicRegression
import sklearn.neighbors

HERE   = os.path.dirname(os.path.abspath(__file__))
CODES  = os.path.join(HERE, "..", "codes")
OUT    = os.path.join(HERE, "exp11")
sys.path.insert(0, CODES)
from embed_lr_dp import load_cached_embedding
import catimage_tf_mlp as TF
sys.path.insert(0, HERE)
from run_exp10_dp_routing import knapsack_dp_route

DATA  = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
BBASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/Results/geneval2_boogu/eval"
UBASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/Results"
CACHE = os.path.join(CODES, ".cache_embed")
CLIP_DIR = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/zhouwei/clip-vit-large-patch14"
EMB_MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/zhaobo45/CKPTS/Qwen3-Embedding-8B"

CFGS      = ["10steps", "50steps", "tts"]
DATASETS  = ["gm", "am", "unigen"]
MODELS    = ["qwen", "flux", "boogu"]
LC_SIZES  = [100, 300, 600, 1000, 1500, 2200]
SEEDS     = [0, 1, 2]
N_EXT_VAL = 200
BUDGET    = 10.0   # 统一 budget

COSTS = {
    "qwen":  np.array([2.0,  10.0, 30.5]),
    "flux":  np.array([4.7,  23.1, 44.9]),
    "boogu": np.array([4.3,  21.5, 64.8]),
}

RIDGE_ALPHAS = [0.01, 0.1, 1., 10., 100., 1000., 1e4]
LAMBDAS = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000),
    np.linspace(0.05, 0.3, 500),
    np.linspace(0.3, 3.0, 200),
    [10., 100., 1e6],
]))


# ── 路径 ──────────────────────────────────────────────────────
def train_paths(dataset, model):
    if dataset == "gm":
        if model == "boogu": return [f"{BBASE}/gen2400_perseed.json"]
        return [f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen_perseed.json",
                f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen2000_perseed.json"]
    if dataset == "am":
        if model == "boogu": return [f"{BBASE}/gen2400_am_v2_perseed.json"]
        return [f"{DATA}/geneval2/edit_am/geneval2_{model}_edit_gen_perseed.json",
                f"{DATA}/geneval2/edit_am/geneval2_{model}_edit_gen2000_perseed.json"]
    if model == "boogu": return [f"{UBASE}/unigenbench_boogu/eval/gen2400_tts_edit_v2.json"]
    return [f"{UBASE}/unigen_{model}/eval/gen2400_edit_v2.json"]

def test_path(dataset, model):
    if dataset == "gm":
        if model == "boogu": return f"{BBASE}/raw800_perseed.json"
        return f"{DATA}/geneval2/edit/geneval2_{model}_edit_raw_perseed.json"
    if dataset == "am":
        if model == "boogu": return f"{BBASE}/raw800_am_perseed.json"
        return f"{DATA}/geneval2/edit_am/geneval2_{model}_edit_raw_perseed.json"
    if model == "boogu": return f"{UBASE}/unigenbench_boogu/eval/raw600_tts_edit_v2.json"
    return f"{UBASE}/unigen_{model}/eval/raw600_edit_v2.json"


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

def merge_pool(dataset, model):
    instrs, scores = [], {c: [] for c in CFGS}
    for p in train_paths(dataset, model):
        ii, ss = load_scores(p)
        instrs += ii
        for c in CFGS: scores[c].extend(ss[c].tolist())
    return instrs, {c: np.array(scores[c], np.float32) for c in CFGS}

def merge_emb(dataset, model):
    return np.concatenate([get_emb(p) for p in train_paths(dataset, model)], 0)

def mean_pool(F, M):
    m = M[:, :, None]
    return (F * m).sum(1) / np.clip(m.sum(1), 1e-6, None)


# ── 路由 ──────────────────────────────────────────────────────
def lag_rq(pred_mat, true_mat, costs, budget):
    pts = []
    for lam in LAMBDAS:
        adj    = pred_mat - lam * costs[None, :]
        chosen = np.argmax(adj, axis=1)
        avg_c  = float(np.mean(costs[chosen]))
        avg_q  = float(np.mean(true_mat[np.arange(len(true_mat)), chosen]))
        pts.append((avg_c, avg_q))
    bucket = defaultdict(float)
    for c, q in pts:
        bucket[round(c, 6)] = max(bucket[round(c, 6)], q)
    xs = np.array(sorted(bucket.keys()))
    qs = np.array([bucket[x] for x in xs])
    if budget <= xs[0]:  return float(qs[0])
    if budget >= xs[-1]: return float(qs[-1])
    idx = int(np.searchsorted(xs, budget))
    c0, c1 = xs[idx-1], xs[idx]
    q0, q1 = qs[idx-1], qs[idx]
    if c1 - c0 < 1e-9: return float(max(q0, q1))
    return float(q0 + (budget - c0) / (c1 - c0) * (q1 - q0))

def dp_rq(pred_mat, true_mat, costs, budget):
    return knapsack_dp_route(pred_mat, true_mat, costs, budget)

def lp_quality(test_means, costs, budget):
    pts = sorted(zip(costs.tolist(), test_means.tolist()))
    best = -np.inf
    for i in range(len(pts)):
        for j in range(len(pts)):
            ci, qi = pts[i]; cj, qj = pts[j]
            if i == j:
                if abs(ci - budget) < 1e-9: best = max(best, qi)
                continue
            lo, hi = (i, j) if ci < cj else (j, i)
            cl, ql = pts[lo]; ch, qh = pts[hi]
            if cl <= budget <= ch:
                w = (budget - cl) / (ch - cl + 1e-12)
                best = max(best, ql + w * (qh - ql))
    return float(best)


# ── Ridge ─────────────────────────────────────────────────────
def fit_ridge(Xtr, ytr, Xva, yva):
    best_sc, best_m = -1e18, None
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        sc = r2_score(yva, m.predict(Xva))
        if sc > best_sc: best_sc = sc; best_m = m
    return best_m

def fit_ridge_w(Xtr, ytr, Xva, yva, w):
    best_sc, best_m = -1e18, None
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, ytr, sample_weight=w)
        sc = r2_score(yva, m.predict(Xva))
        if sc > best_sc: best_sc = sc; best_m = m
    return best_m


# ── λ* ────────────────────────────────────────────────────────
def find_lam_star(Ytr, costs, budget):
    best_lam, best_dist = 0.0, float('inf')
    for lam in LAMBDAS:
        chosen = np.argmax(Ytr - lam * costs[None, :], axis=1)
        dist = abs(float(np.mean(costs[chosen])) - budget)
        if dist < best_dist: best_dist = dist; best_lam = lam
    return float(best_lam)


# ── Soft-RQ (§7 fixed λ*) ─────────────────────────────────────
def train_softrq(Xtr, Ytr, lam_star, costs, tau, l2, epochs=1000, lr=5e-3, device="cpu"):
    import torch, torch.nn.functional as F
    dev = torch.device(device)
    costs_t = torch.tensor(costs, dtype=torch.float32, device=dev)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    W = torch.zeros(3, Xt.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(epochs):
        pred = Xt @ W.T + b
        adj  = pred - lam_star * costs_t
        wt   = F.softmax(adj / tau, dim=1)
        loss = -(wt * Yt).sum(1).mean() + 0.5 * l2 * (W**2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach()

def pred_linear(W, b, X):
    import torch
    dev = W.device
    with torch.no_grad():
        return (torch.tensor(X, dtype=torch.float32, device=dev) @ W.T + b).cpu().numpy()

def run_softrq(Xtr, Ytr, Xva, Yva, Xte, lam_star, costs, budget, device="cpu"):
    best_rq, best_pred = -1e18, None
    for tau in [0.1, 0.2, 0.5]:
        for l2 in [0.1, 1.0, 5.0]:
            W, b = train_softrq(Xtr, Ytr, lam_star, costs, tau, l2, device=device)
            rq_va = dp_rq(pred_linear(W, b, Xva), Yva, costs, budget)
            if rq_va > best_rq:
                best_rq = rq_va; best_pred = pred_linear(W, b, Xte)
    return best_pred

# ── Multi-λ Soft-RQ (§8C) ─────────────────────────────────────
def run_multilam(Xtr, Ytr, Xva, Yva, Xte, lam_star, costs, budget, device="cpu"):
    import torch, torch.nn.functional as F
    dev = torch.device(device)
    lam_max  = float(max(0.03, 5 * lam_star))
    costs_t  = torch.tensor(costs, dtype=torch.float32, device=dev)
    Xt  = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Yt  = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    Xvt = torch.tensor(Xva, dtype=torch.float32, device=dev)
    Xtet= torch.tensor(Xte, dtype=torch.float32, device=dev)
    lams = torch.linspace(0, lam_max, 15, device=dev)
    best_rq, best_pred = -1e18, None
    for tau in [0.1, 0.2, 0.5]:
        for l2 in [0.1, 1.0, 5.0]:
            W   = torch.zeros(3, Xt.shape[1], device=dev, requires_grad=True)
            b_t = torch.zeros(3, device=dev, requires_grad=True)
            opt = torch.optim.Adam([W, b_t], lr=5e-3)
            for _ in range(800):
                pred = Xt @ W.T + b_t
                loss = torch.tensor(0., device=dev)
                for lam in lams:
                    wt = F.softmax((pred - lam * costs_t) / tau, dim=1)
                    loss = loss - (wt * Yt).sum(1).mean()
                loss = loss / len(lams) + 0.5 * l2 * (W**2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pva = (Xvt  @ W.T + b_t).cpu().numpy()
                pte = (Xtet @ W.T + b_t).cpu().numpy()
            rq_va = dp_rq(pva, Yva, costs, budget)
            if rq_va > best_rq: best_rq = rq_va; best_pred = pte
    return best_pred

# ── Weighted Ridge (§8B) ──────────────────────────────────────
def run_weighted_ridge(Xtr, Ytr, Xva, Yva, Xte, lam_star, costs, budget):
    adj    = Ytr - lam_star * costs[None, :]
    s      = np.sort(adj, axis=1)
    margin = s[:, -1] - s[:, -2]
    best_rq, best_pred = -1e18, None
    for eps_pct in [5, 15]:
        for alpha_w in [0.5, 1.0, 2.0]:
            eps_w = np.percentile(margin, eps_pct)
            w = 1.0 / (margin + eps_w) ** alpha_w; w /= w.mean()
            ms = [fit_ridge_w(Xtr, Ytr[:, j], Xva, Yva[:, j], w) for j in range(3)]
            pva = np.stack([m.predict(Xva) for m in ms], 1)
            rq_va = dp_rq(pva, Yva, costs, budget)
            if rq_va > best_rq:
                best_rq = rq_va
                best_pred = np.stack([m.predict(Xte) for m in ms], 1)
    return best_pred

# ── Pairwise SVM (§8D) ────────────────────────────────────────
def run_svm(Xtr, Ytr, Xva, Yva, Xte, lam_star, costs, budget):
    thr1 = lam_star * (costs[1] - costs[0])
    thr2 = lam_star * (costs[2] - costs[1])
    y1 = np.where(Ytr[:, 1] - Ytr[:, 0] > thr1, 1, -1)
    y2 = np.where(Ytr[:, 2] - Ytr[:, 1] > thr2, 1, -1)
    best_rq, best_pred = -1e18, None
    for C_val in [0.001, 0.01, 0.1, 1.0]:
        try:
            s1 = LinearSVC(C=C_val, max_iter=2000, dual=True).fit(Xtr, y1)
            s2 = LinearSVC(C=C_val, max_iter=2000, dual=True).fit(Xtr, y2)
        except Exception: continue
        df1v = s1.decision_function(Xva); df2v = s2.decision_function(Xva)
        pva = np.stack([-df1v, np.zeros(len(Xva)), df2v], 1)
        rq_va = dp_rq(pva, Yva, costs, budget)
        if rq_va > best_rq:
            best_rq = rq_va
            df1t = s1.decision_function(Xte); df2t = s2.decision_function(Xte)
            best_pred = np.stack([-df1t, np.zeros(len(Xte)), df2t], 1)
    return best_pred

# ── Isotonic (§8E) ────────────────────────────────────────────
def run_isotonic(ridge_pva, ridge_pte, Yva):
    calib = np.empty_like(ridge_pte)
    for j in range(3):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(ridge_pva[:, j], Yva[:, j])
        calib[:, j] = iso.predict(ridge_pte[:, j])
    return calib


# ── 单 shard ──────────────────────────────────────────────────
def run_shard(dataset, model, gpu):
    print(f"\n=== §11  {dataset}/{model}  budget={BUDGET}  gpu={gpu} ===", flush=True)
    dev   = f"cuda:{gpu}"
    costs = COSTS[model]

    tp = test_path(dataset, model)
    if not os.path.exists(tp):
        print(f"  [skip] {tp}"); return None

    pool_instrs, pool_scores = merge_pool(dataset, model)
    n_pool = len(pool_instrs)
    emb_pool = merge_emb(dataset, model)

    te_instrs, yte = load_scores(tp)
    n_te     = len(te_instrs)
    true_mat = np.stack([yte[c] for c in CFGS], 1)
    emb_te   = get_emb(tp)[:n_te]

    # CLIP features
    pfx = f"exp11_{dataset}_{model}_"
    Fpool, Mpool = TF.clip_token_features(pool_instrs, pfx+"pool", CLIP_DIR, dev)
    Fte,   Mte   = TF.clip_token_features(te_instrs,   pfx+"test", CLIP_DIR, dev)
    Cte = mean_pool(Fte, Mte)

    # 常数指标
    oracle_lag = lag_rq(true_mat, true_mat, costs, BUDGET)
    oracle_dp  = dp_rq(true_mat, true_mat, costs, BUDGET)
    lp         = lp_quality(np.array([yte[c].mean() for c in CFGS]), costs, BUDGET)
    always50   = float(yte["50steps"].mean())
    print(f"  oracle_lag={oracle_lag:.4f}  oracle_dp={oracle_dp:.4f}  "
          f"lp={lp:.4f}  50steps={always50:.4f}", flush=True)

    # ext_val 固定划分
    rng_fix = random.Random(42)
    all_idx = list(range(n_pool)); rng_fix.shuffle(all_idx)
    ext_val = np.array(all_idx[:N_EXT_VAL])
    lc_pool = all_idx[N_EXT_VAL:]
    y_va    = {c: pool_scores[c][ext_val] for c in CFGS}
    Yva_mat = np.stack([y_va[c] for c in CFGS], 1)
    Fva, Mva = Fpool[ext_val], Mpool[ext_val]

    METHODS = ["ridge_score", "ridge_delta",
               "knn_score",   "knn_delta",
               "tf_score",    "tf_delta",
               "softrq",      "multilam",
               "weighted_ridge", "svm",   "isotonic"]

    lc_data = []

    for n in LC_SIZES:
        print(f"  n={n:5d}", end="  ", flush=True)
        rows = {m: [] for m in METHODS}

        for seed in SEEDS:
            rng = random.Random(seed)
            sub = np.array(rng.sample(lc_pool, min(n, len(lc_pool))))
            y_tr  = {c: pool_scores[c][sub] for c in CFGS}
            Ytr_mat = np.stack([y_tr[c] for c in CFGS], 1)
            d_tr  = np.stack([y_tr["50steps"] - y_tr["10steps"],
                               y_tr["tts"]     - y_tr["50steps"]], 1)
            d_va  = np.stack([y_va["50steps"] - y_va["10steps"],
                               y_va["tts"]     - y_va["50steps"]], 1)

            lam_star = find_lam_star(Ytr_mat, costs, BUDGET)

            # ── Ridge ──────────────────────────────────────
            sc_r  = StandardScaler().fit(emb_pool[sub])
            Xtr_r = sc_r.transform(emb_pool[sub]).astype(np.float32)
            Xva_r = sc_r.transform(emb_pool[ext_val]).astype(np.float32)
            Xte_r = sc_r.transform(emb_te).astype(np.float32)

            ms_score = [fit_ridge(Xtr_r, Ytr_mat[:, j], Xva_r, Yva_mat[:, j]) for j in range(3)]
            ridge_pte = np.stack([m.predict(Xte_r) for m in ms_score], 1)
            ridge_pva = np.stack([m.predict(Xva_r) for m in ms_score], 1)

            d1_r = ridge_pte[:, 1] - ridge_pte[:, 0]
            d2_r = ridge_pte[:, 2] - ridge_pte[:, 1]
            ridge_delta_mat = np.stack([-d1_r, np.zeros(n_te), d2_r], 1)

            rows["ridge_score"].append(lag_rq(ridge_pte, true_mat, costs, BUDGET))
            rows["ridge_delta"].append(lag_rq(ridge_delta_mat, true_mat, costs, BUDGET))

            # ── KNN ────────────────────────────────────────
            k     = min(50, max(1, n // 3))
            sc_c  = StandardScaler().fit(mean_pool(Fpool[sub], Mpool[sub]))
            Ctr_s = sc_c.transform(mean_pool(Fpool[sub], Mpool[sub]))
            Cte_s = sc_c.transform(Cte)

            knn_s_mat = np.stack([
                sklearn.neighbors.KNeighborsRegressor(n_neighbors=k)
                    .fit(Ctr_s, y_tr[c]).predict(Cte_s)
                for c in CFGS], 1)

            nn = sklearn.neighbors.NearestNeighbors(n_neighbors=k).fit(Ctr_s)
            idx_nn = nn.kneighbors(Cte_s, return_distance=False)
            d_te_knn = np.stack([d_tr[idx_nn[i]].mean(0) for i in range(n_te)])
            knn_delta_mat = np.stack([-d_te_knn[:, 0], np.zeros(n_te), d_te_knn[:, 1]], 1)

            rows["knn_score"].append(lag_rq(knn_s_mat, true_mat, costs, BUDGET))
            rows["knn_delta"].append(lag_rq(knn_delta_mat, true_mat, costs, BUDGET))

            # ── TFMLP ──────────────────────────────────────
            tf_score = {}
            for c in CFGS:
                lo, hi = y_tr[c].min(), y_tr[c].max()
                rs = max(hi - lo, 1e-6)
                nf = lambda a, lo=lo, rs=rs: (a - lo) / rs
                pred_n, _, _ = TF.train_tf_mlp(
                    Fpool[sub], Mpool[sub], nf(y_tr[c]).reshape(-1, 1),
                    Fva, Mva, nf(y_va[c]).reshape(-1, 1),
                    Fte, Mte, M=1, seed=seed, device=dev)
                tf_score[c] = pred_n[:, 0] * rs + lo
            tf_score_mat = np.stack([tf_score[c] for c in CFGS], 1)

            lo_d, hi_d = d_tr.min(), d_tr.max()
            rd = max(hi_d - lo_d, 1e-6)
            nd = lambda a: (a - lo_d) / rd
            pred_dn, _, _ = TF.train_tf_mlp(
                Fpool[sub], Mpool[sub], nd(d_tr),
                Fva, Mva, nd(d_va),
                Fte, Mte, M=2, seed=seed, device=dev)
            tf_delta_raw = pred_dn * rd + lo_d
            tf_delta_mat = np.stack([-tf_delta_raw[:, 0], np.zeros(n_te), tf_delta_raw[:, 1]], 1)

            rows["tf_score"].append(lag_rq(tf_score_mat, true_mat, costs, BUDGET))
            rows["tf_delta"].append(lag_rq(tf_delta_mat, true_mat, costs, BUDGET))

            # ── Soft-RQ §7 ─────────────────────────────────
            if n >= 300:
                pte = run_softrq(Xtr_r, Ytr_mat, Xva_r, Yva_mat, Xte_r,
                                  lam_star, costs, BUDGET, device=dev)
                rows["softrq"].append(lag_rq(pte, true_mat, costs, BUDGET))
            else:
                rows["softrq"].append(float('nan'))

            # ── Multi-λ §8C ────────────────────────────────
            if n >= 300:
                pte = run_multilam(Xtr_r, Ytr_mat, Xva_r, Yva_mat, Xte_r,
                                    lam_star, costs, BUDGET, device=dev)
                rows["multilam"].append(lag_rq(pte, true_mat, costs, BUDGET))
            else:
                rows["multilam"].append(float('nan'))

            # ── Weighted Ridge §8B ─────────────────────────
            pte = run_weighted_ridge(Xtr_r, Ytr_mat, Xva_r, Yva_mat, Xte_r,
                                      lam_star, costs, BUDGET)
            rows["weighted_ridge"].append(lag_rq(pte, true_mat, costs, BUDGET))

            # ── Pairwise SVM §8D ───────────────────────────
            pte = run_svm(Xtr_r, Ytr_mat, Xva_r, Yva_mat, Xte_r,
                           lam_star, costs, BUDGET)
            if pte is not None:
                rows["svm"].append(lag_rq(pte, true_mat, costs, BUDGET))
            else:
                rows["svm"].append(float('nan'))

            # ── Isotonic §8E ───────────────────────────────
            calib = run_isotonic(ridge_pva, ridge_pte, Yva_mat)
            rows["isotonic"].append(lag_rq(calib, true_mat, costs, BUDGET))

        means = {m: float(np.nanmean(rows[m])) for m in METHODS}
        print("  ".join(f"{m}={means[m]:.4f}" for m in ["ridge_score","tf_delta","softrq","multilam"]),
              flush=True)
        lc_data.append({"n": n, **{m: means[m] for m in METHODS},
                         **{f"{m}_seeds": [float(x) for x in rows[m]] for m in METHODS}})

    result = {
        "dataset": dataset, "model": model,
        "oracle_lag": oracle_lag, "oracle_dp": oracle_dp,
        "lp": lp, "always_50steps": always50,
        "lc": lc_data,
    }
    out_path = os.path.join(OUT, f"{dataset}_{model}.json")
    fd, tmp = tempfile.mkstemp(dir=OUT, suffix=".json")
    os.close(fd)
    json.dump(result, open(tmp, "w"), indent=2)
    shutil.move(tmp, out_path)
    print(f"  [saved] {out_path}", flush=True)
    return result


# ── merge + md ────────────────────────────────────────────────
def merge_and_write():
    all_results = []
    for ds in DATASETS:
        for model in MODELS:
            p = os.path.join(OUT, f"{ds}_{model}.json")
            if os.path.exists(p):
                all_results.append(json.load(open(p)))

    METHODS = ["ridge_score", "ridge_delta", "knn_score", "knn_delta",
               "tf_score", "tf_delta", "softrq", "multilam",
               "weighted_ridge", "svm", "isotonic"]

    # 写 summary md (n=2200)
    lines = [
        "# §11 统一 Budget=10 路由实验",
        "",
        f"Budget 固定 = {BUDGET}（= qwen 50steps cost），所有 9 个 dataset/model 统一。",
        "路由方法：Lagrangian。方法缩写见下表。",
        "",
        "| 缩写 | 方法 |",
        "|---|---|",
        "| RS | Ridge-Score |",
        "| RD | Ridge-Delta |",
        "| KS | KNN-Score |",
        "| KD | KNN-Delta |",
        "| TS | TFMLP-Score |",
        "| TD | TFMLP-Delta |",
        "| SRQ | Soft-RQ (§7) |",
        "| MLam | Multi-λ Soft-RQ (§8C) |",
        "| WR | Weighted Ridge (§8B) |",
        "| SVM | Pairwise SVM (§8D) |",
        "| ISO | Isotonic (§8E) |",
        "",
        "## 结果（n=2200，3 seeds 均值）",
        "",
        "| Dataset/Model | Oracle | 50steps | RS | RD | KS | KD | TS | TD | SRQ | MLam | WR | SVM | ISO |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in all_results:
        tag  = f"{r['dataset']}/{r['model']}"
        lc2200 = next((x for x in r["lc"] if x["n"] == 2200), None)
        if lc2200 is None: continue
        vals = [f"{lc2200.get(m, float('nan')):.4f}" for m in METHODS]
        lines.append(f"| {tag} | {r['oracle_lag']:.4f} | {r['always_50steps']:.4f} | "
                     + " | ".join(vals) + " |")

    lines += ["", "## 学习曲线（Ridge-Score，n=100~2200）", ""]
    for r in all_results:
        tag = f"{r['dataset']}/{r['model']}"
        ns  = [x["n"] for x in r["lc"]]
        rss = [x.get("ridge_score", float('nan')) for x in r["lc"]]
        tds = [x.get("tf_delta",    float('nan')) for x in r["lc"]]
        lines.append(f"**{tag}** oracle={r['oracle_lag']:.4f} 50s={r['always_50steps']:.4f}")
        lines.append("n:  " + "  ".join(f"{n:>6}" for n in ns))
        lines.append("RS: " + "  ".join(f"{v:>6.4f}" for v in rss))
        lines.append("TD: " + "  ".join(f"{v:>6.4f}" for v in tds))
        lines.append("")

    md_path = os.path.join(OUT, "summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[md] {md_path}")

    # 每个 dataset 写一个单独 md
    ds_labels = {"gm": "GenEval2-GM", "am": "GenEval2-AM", "unigen": "UniGenBench"}
    for ds in DATASETS:
        rows = [r for r in all_results if r["dataset"] == ds]
        if not rows: continue
        dlines = [f"# §11 {ds_labels[ds]}  (budget={BUDGET})", ""]
        dlines += [
            "| Model | Oracle | 50steps | RS | RD | KS | KD | TS | TD | SRQ | MLam | WR | SVM | ISO |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lc2200 = next((x for x in r["lc"] if x["n"] == 2200), None)
            if lc2200 is None: continue
            vals = [f"{lc2200.get(m, float('nan')):.4f}" for m in METHODS]
            dlines.append(f"| {r['model']} | {r['oracle_lag']:.4f} | {r['always_50steps']:.4f} | "
                          + " | ".join(vals) + " |")
        dlines.append("")
        with open(os.path.join(OUT, f"{ds}.md"), "w") as f:
            f.write("\n".join(dlines) + "\n")
        print(f"[md] {OUT}/{ds}.md")


# ── 入口 ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="dataset/model")
    ap.add_argument("--gpu",   type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    if a.merge:
        merge_and_write(); return

    if a.shard:
        ds, model = a.shard.split("/")
        run_shard(ds, model, a.gpu)
        return

    # 顺序跑（调试）
    for ds in DATASETS:
        for model in MODELS:
            run_shard(ds, model, a.gpu)
    merge_and_write()


if __name__ == "__main__":
    main()
