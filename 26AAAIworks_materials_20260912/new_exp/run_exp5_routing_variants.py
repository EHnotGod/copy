#!/usr/bin/env python3
"""
§5  路由变体实验（只改路由，不改预测器）

固定：Ridge + Qwen3-Embedding-8B (4096d)，n=2200，gm/qwen + gm/flux
变体：
  score_r2      : 预测绝对分，Lagrangian 路由，α by val-R²（基线）
  score_rq      : 预测绝对分，Lagrangian 路由，α by val-RQ
  delta_dir_r2  : 直接预测 d1/d2，路由用 [-d1,0,d2]，α by val-R²
  delta_dir_rq  : 直接预测 d1/d2，路由用 [-d1,0,d2]，α by val-RQ
  ranknorm_r2   : 预测 q-mean_q，Lagrangian 路由，α by val-R²
  ranknorm_rq   : 预测 q-mean_q，Lagrangian 路由，α by val-RQ
"""
import os, sys, json
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from collections import defaultdict

HERE  = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(HERE, "..", "codes")
sys.path.insert(0, CODES)
from embed_lr_dp import load_cached_embedding

DATA  = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
CACHE = os.path.join(CODES, ".cache_embed")
EMB_MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/zhaobo45/CKPTS/Qwen3-Embedding-8B"

CFGS  = ["10steps", "50steps", "tts"]
COSTS = {"qwen": np.array([2.0, 10.0, 30.5]),
         "flux": np.array([4.7, 23.1, 44.9])}
BUDGETS = {"qwen": 10.0, "flux": 23.1}
ALPHAS = [0.01, 0.1, 1., 10., 100., 1000., 1e4]
SEEDS  = [0, 1, 2]
N_POOL = 2200
N_EXT_VAL = 200

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


# ── 路由质量 ──────────────────────────────────────────────────
def route_quality(pred_mat, true_mat, costs, budget):
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


# ── Ridge helpers ─────────────────────────────────────────────
def best_ridge_r2(Xtr, ytr, Xva, yva):
    """α by val R²"""
    best_sc, best_m = -1e18, None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        sc = r2_score(yva, m.predict(Xva))
        if sc > best_sc: best_sc = sc; best_m = m
    return best_m

def best_ridge_rq_scalar(Xtr, ytr_mat, Xva, yva_mat, Xte,
                         true_va, true_te, costs, budget,
                         pred_fn):
    """
    α by val RQ.
    pred_fn(model_list, Xte) → pred_mat: (N_te, 3)  [called once per α to get val pred too]
    ytr_mat: (N_tr, K), yva_mat: (N_va, K)  — targets for fitting
    pred_fn signature: pred_fn(models, X) where models is list of fitted Ridge
    """
    best_rq, best_models = -1e18, None
    for a in ALPHAS:
        models = [Ridge(alpha=a).fit(Xtr, ytr_mat[:, j])
                  for j in range(ytr_mat.shape[1])]
        va_pred = pred_fn(models, Xva)
        rq_va = route_quality(va_pred, true_va, costs, budget)
        if rq_va > best_rq:
            best_rq = rq_va; best_models = models
    te_pred = pred_fn(best_models, Xte)
    return te_pred


# ── 单次 seed 实验 ────────────────────────────────────────────
def run_one(model, seed, emb_pool, pool_scores, emb_te, true_mat, costs, budget, ext_val, lc_pool):
    rng = random.Random(seed)
    sub = np.array(rng.sample(lc_pool, min(N_POOL, len(lc_pool))))

    sc = StandardScaler().fit(emb_pool[sub])
    Xtr = sc.transform(emb_pool[sub])
    Xva = sc.transform(emb_pool[ext_val])
    Xte = sc.transform(emb_te)

    y_tr = {c: pool_scores[c][sub]     for c in CFGS}
    y_va = {c: pool_scores[c][ext_val] for c in CFGS}
    Ytr_abs = np.stack([y_tr[c] for c in CFGS], 1)   # (N_tr, 3)
    Yva_abs = np.stack([y_va[c] for c in CFGS], 1)
    true_va = np.stack([pool_scores[c][ext_val] for c in CFGS], 1)

    # delta targets: d1 = q50-q10, d2 = qtts-q50
    Ytr_d = np.stack([y_tr["50steps"] - y_tr["10steps"],
                      y_tr["tts"]     - y_tr["50steps"]], 1)
    Yva_d = np.stack([y_va["50steps"] - y_va["10steps"],
                      y_va["tts"]     - y_va["50steps"]], 1)

    # rank_norm targets: q - row_mean
    mu_tr = Ytr_abs.mean(1, keepdims=True)
    mu_va = Yva_abs.mean(1, keepdims=True)
    Ytr_rn = Ytr_abs - mu_tr
    Yva_rn = Yva_abs - mu_va

    results = {}

    # ── 1. score_r2 (baseline) ────────────────────────────────
    ms = [best_ridge_r2(Xtr, Ytr_abs[:, j], Xva, Yva_abs[:, j]) for j in range(3)]
    pred = np.stack([m.predict(Xte) for m in ms], 1)
    results["score_r2"] = route_quality(pred, true_mat, costs, budget)

    # ── 2. score_rq ───────────────────────────────────────────
    def pred_abs(models, X): return np.stack([m.predict(X) for m in models], 1)
    pred = best_ridge_rq_scalar(Xtr, Ytr_abs, Xva, Yva_abs, Xte,
                                true_va, true_mat, costs, budget, pred_abs)
    results["score_rq"] = route_quality(pred, true_mat, costs, budget)

    # ── 3. delta_dir_r2 ───────────────────────────────────────
    # 直接拟合 d1, d2；路由用 [-d1_pred, 0, d2_pred]
    md = [best_ridge_r2(Xtr, Ytr_d[:, j], Xva, Yva_d[:, j]) for j in range(2)]
    d_pred = np.stack([m.predict(Xte) for m in md], 1)
    delta_mat = np.stack([-d_pred[:, 0], np.zeros(len(emb_te)), d_pred[:, 1]], 1)
    results["delta_dir_r2"] = route_quality(delta_mat, true_mat, costs, budget)

    # ── 4. delta_dir_rq ───────────────────────────────────────
    def pred_delta(models, X):
        dp = np.stack([m.predict(X) for m in models], 1)  # (N, 2)
        return np.stack([-dp[:, 0], np.zeros(len(X)), dp[:, 1]], 1)
    pred_d = best_ridge_rq_scalar(Xtr, Ytr_d, Xva, Yva_d, Xte,
                                  true_va, true_mat, costs, budget, pred_delta)
    results["delta_dir_rq"] = route_quality(pred_d, true_mat, costs, budget)

    # ── 5. ranknorm_r2 ────────────────────────────────────────
    mrn = [best_ridge_r2(Xtr, Ytr_rn[:, j], Xva, Yva_rn[:, j]) for j in range(3)]
    pred = np.stack([m.predict(Xte) for m in mrn], 1)
    results["ranknorm_r2"] = route_quality(pred, true_mat, costs, budget)

    # ── 6. ranknorm_rq ────────────────────────────────────────
    def pred_rn(models, X): return np.stack([m.predict(X) for m in models], 1)
    pred = best_ridge_rq_scalar(Xtr, Ytr_rn, Xva, Yva_rn, Xte,
                                true_va, true_mat, costs, budget, pred_rn)
    results["ranknorm_rq"] = route_quality(pred, true_mat, costs, budget)

    return results


# ── 主逻辑 ────────────────────────────────────────────────────
def run_model(model):
    p1 = f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen_perseed.json"
    p2 = f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen2000_perseed.json"
    tp = f"{DATA}/geneval2/edit/geneval2_{model}_edit_raw_perseed.json"

    i1, s1 = load_scores(p1); i2, s2 = load_scores(p2)
    pool_scores = {c: np.concatenate([s1[c], s2[c]]) for c in CFGS}
    n_pool = len(i1) + len(i2)

    _, yte = load_scores(tp)
    n_te   = len(yte[CFGS[0]])
    true_mat = np.stack([yte[c] for c in CFGS], 1)

    costs  = COSTS[model]
    budget = BUDGETS[model]
    oracle   = route_quality(true_mat, true_mat, costs, budget)
    always50 = float(np.mean(yte["50steps"]))

    emb_pool = np.concatenate([get_emb(p1), get_emb(p2)], 0)
    emb_te   = get_emb(tp)[:n_te]

    rng_fix  = random.Random(42)
    all_idx  = list(range(n_pool)); rng_fix.shuffle(all_idx)
    ext_val  = np.array(all_idx[:N_EXT_VAL])
    lc_pool  = all_idx[N_EXT_VAL:]

    VARIANTS = ["score_r2", "score_rq",
                "delta_dir_r2", "delta_dir_rq",
                "ranknorm_r2", "ranknorm_rq"]
    acc = {v: [] for v in VARIANTS}

    for seed in SEEDS:
        r = run_one(model, seed, emb_pool, pool_scores, emb_te, true_mat,
                    costs, budget, ext_val, lc_pool)
        for v in VARIANTS:
            acc[v].append(r[v])
        row = "  ".join(f"{v}={r[v]:.4f}" for v in VARIANTS)
        print(f"  gm/{model} seed={seed}  {row}", flush=True)

    print(f"\n  gm/{model} MEAN  oracle={oracle:.4f}  50steps={always50:.4f}")
    for v in VARIANTS:
        m = float(np.mean(acc[v]))
        gain = m - always50
        print(f"    {v:<16}: {m:.4f}  gain={gain:+.4f}")
    print()

    return {
        "model": model, "oracle": oracle, "always_50steps": always50,
        **{v: float(np.mean(acc[v])) for v in VARIANTS},
        **{f"{v}_per_seed": [float(x) for x in acc[v]] for v in VARIANTS},
    }


def main():
    print("=== §5 路由变体（gm/qwen + gm/flux）===\n")
    results = []
    for model in ["qwen", "flux"]:
        r = run_model(model)
        results.append(r)

    out = os.path.join(HERE, "results", "exp5_routing_variants.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
