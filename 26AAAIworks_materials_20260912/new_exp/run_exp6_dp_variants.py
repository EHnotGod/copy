#!/usr/bin/env python3
"""
§6  DP 路由 × 预测目标变体（Ridge-only，gm/qwen + gm/flux）

路由方法：MCKP 背包 DP（exp10 的 knapsack_dp_route）
预测器：Ridge + Qwen3-Embedding-8B (4096d)
α 选择：val-R²（默认）或 val-DP-RQ

变体：
  score_r2      : 预测绝对分，α by val-R²（DP路由基线）
  score_rq      : 预测绝对分，α by val-DP-RQ
  delta_dir_r2  : 直接预测 d1=q50-q10 / d2=qtts-q50，路由用 [-d1,0,d2]，α by val-R²
  delta_dir_rq  : 同上，α by val-DP-RQ
  ranknorm_r2   : 预测 q-mean_q，α by val-R²
  ranknorm_rq   : 预测 q-mean_q，α by val-DP-RQ
"""
import os, sys, json, random
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

HERE  = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(HERE, "..", "codes")
sys.path.insert(0, CODES)
from embed_lr_dp import load_cached_embedding
sys.path.insert(0, HERE)
from run_exp10_dp_routing import knapsack_dp_route

DATA  = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
CACHE = os.path.join(CODES, ".cache_embed")
EMB_MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/zhaobo45/CKPTS/Qwen3-Embedding-8B"

CFGS  = ["10steps", "50steps", "tts"]
COSTS   = {"qwen": np.array([2.0, 10.0, 30.5]),
           "flux": np.array([4.7, 23.1, 44.9])}
BUDGETS = {"qwen": 10.0, "flux": 23.1}
ALPHAS  = [0.01, 0.1, 1., 10., 100., 1000., 1e4]
SEEDS   = [0, 1, 2]
N_POOL  = 2200
N_EXT_VAL = 200

VARIANTS = ["score_r2", "score_rq",
            "delta_dir_r2", "delta_dir_rq",
            "ranknorm_r2", "ranknorm_rq"]


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


# ── Ridge helpers ─────────────────────────────────────────────
def best_ridge_r2(Xtr, ytr, Xva, yva, Xte):
    best_sc, best_m = -1e18, None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        sc = r2_score(yva, m.predict(Xva))
        if sc > best_sc: best_sc = sc; best_m = m
    return best_m.predict(Xte)

def best_ridge_rq(Xtr, Ytr, Xva, Yva_true, Xte, true_va,
                  costs, budget, make_pred_mat):
    """
    α by val DP-RQ.
    Ytr: (N_tr, K) — 训练 targets
    make_pred_mat(models, X) → (N, 3) routing pred mat
    true_va: (N_va, 3) 真实分（评估 val RQ 用）
    """
    best_rq, best_models = -1e18, None
    for a in ALPHAS:
        models = [Ridge(alpha=a).fit(Xtr, Ytr[:, j]) for j in range(Ytr.shape[1])]
        va_pred = make_pred_mat(models, Xva)
        rq_va = dp_rq(va_pred, true_va, costs, budget)
        if rq_va > best_rq:
            best_rq = rq_va; best_models = models
    return make_pred_mat(best_models, Xte)


# ── 单次 seed ─────────────────────────────────────────────────
def run_one(model, seed, emb_pool, pool_scores, emb_te, true_mat,
            costs, budget, ext_val, lc_pool):
    rng = random.Random(seed)
    sub = np.array(rng.sample(lc_pool, min(N_POOL, len(lc_pool))))

    sc = StandardScaler().fit(emb_pool[sub])
    Xtr = sc.transform(emb_pool[sub])
    Xva = sc.transform(emb_pool[ext_val])
    Xte = sc.transform(emb_te)

    y_tr = {c: pool_scores[c][sub]     for c in CFGS}
    y_va = {c: pool_scores[c][ext_val] for c in CFGS}
    Ytr_abs = np.stack([y_tr[c] for c in CFGS], 1)
    Yva_abs = np.stack([y_va[c] for c in CFGS], 1)
    true_va = np.stack([pool_scores[c][ext_val] for c in CFGS], 1)

    # delta targets
    Ytr_d = np.stack([y_tr["50steps"] - y_tr["10steps"],
                      y_tr["tts"]     - y_tr["50steps"]], 1)
    Yva_d = np.stack([y_va["50steps"] - y_va["10steps"],
                      y_va["tts"]     - y_va["50steps"]], 1)

    # ranknorm targets
    mu_tr = Ytr_abs.mean(1, keepdims=True)
    mu_va = Yva_abs.mean(1, keepdims=True)
    Ytr_rn = Ytr_abs - mu_tr
    Yva_rn = Yva_abs - mu_va

    n_te = len(emb_te)
    res = {}

    # ── 1. score_r2 ───────────────────────────────────────────
    pred = np.stack([best_ridge_r2(Xtr, Ytr_abs[:, j], Xva, Yva_abs[:, j], Xte)
                     for j in range(3)], 1)
    res["score_r2"] = dp_rq(pred, true_mat, costs, budget)

    # ── 2. score_rq ───────────────────────────────────────────
    def mk_abs(models, X): return np.stack([m.predict(X) for m in models], 1)
    pred = best_ridge_rq(Xtr, Ytr_abs, Xva, Yva_abs, Xte, true_va,
                         costs, budget, mk_abs)
    res["score_rq"] = dp_rq(pred, true_mat, costs, budget)

    # ── 3. delta_dir_r2 ───────────────────────────────────────
    d_pred = np.stack([best_ridge_r2(Xtr, Ytr_d[:, j], Xva, Yva_d[:, j], Xte)
                       for j in range(2)], 1)
    delta_mat = np.stack([-d_pred[:, 0], np.zeros(n_te), d_pred[:, 1]], 1)
    res["delta_dir_r2"] = dp_rq(delta_mat, true_mat, costs, budget)

    # ── 4. delta_dir_rq ───────────────────────────────────────
    def mk_delta(models, X):
        dp_ = np.stack([m.predict(X) for m in models], 1)
        return np.stack([-dp_[:, 0], np.zeros(len(X)), dp_[:, 1]], 1)
    pred = best_ridge_rq(Xtr, Ytr_d, Xva, Yva_d, Xte, true_va,
                         costs, budget, mk_delta)
    res["delta_dir_rq"] = dp_rq(pred, true_mat, costs, budget)

    # ── 5. ranknorm_r2 ────────────────────────────────────────
    pred = np.stack([best_ridge_r2(Xtr, Ytr_rn[:, j], Xva, Yva_rn[:, j], Xte)
                     for j in range(3)], 1)
    res["ranknorm_r2"] = dp_rq(pred, true_mat, costs, budget)

    # ── 6. ranknorm_rq ────────────────────────────────────────
    def mk_rn(models, X): return np.stack([m.predict(X) for m in models], 1)
    pred = best_ridge_rq(Xtr, Ytr_rn, Xva, Yva_rn, Xte, true_va,
                         costs, budget, mk_rn)
    res["ranknorm_rq"] = dp_rq(pred, true_mat, costs, budget)

    return res


# ── 单模型主流程 ──────────────────────────────────────────────
def run_model(model):
    p1 = f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen_perseed.json"
    p2 = f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen2000_perseed.json"
    tp = f"{DATA}/geneval2/edit/geneval2_{model}_edit_raw_perseed.json"

    i1, s1 = load_scores(p1); i2, s2 = load_scores(p2)
    pool_scores = {c: np.concatenate([s1[c], s2[c]]) for c in CFGS}
    n_pool = len(i1) + len(i2)

    _, yte = load_scores(tp)
    n_te     = len(yte[CFGS[0]])
    true_mat = np.stack([yte[c] for c in CFGS], 1)

    costs  = COSTS[model]; budget = BUDGETS[model]
    oracle   = dp_rq(true_mat, true_mat, costs, budget)
    always50 = float(np.mean(yte["50steps"]))

    emb_pool = np.concatenate([get_emb(p1), get_emb(p2)], 0)
    emb_te   = get_emb(tp)[:n_te]

    rng_fix = random.Random(42)
    all_idx = list(range(n_pool)); rng_fix.shuffle(all_idx)
    ext_val = np.array(all_idx[:N_EXT_VAL])
    lc_pool = all_idx[N_EXT_VAL:]

    acc = {v: [] for v in VARIANTS}
    for seed in SEEDS:
        r = run_one(model, seed, emb_pool, pool_scores, emb_te, true_mat,
                    costs, budget, ext_val, lc_pool)
        for v in VARIANTS: acc[v].append(r[v])
        row = "  ".join(f"{v}={r[v]:.4f}" for v in VARIANTS)
        print(f"  gm/{model} seed={seed}  {row}", flush=True)

    print(f"\n  gm/{model}  oracle={oracle:.4f}  50steps={always50:.4f}")
    for v in VARIANTS:
        m = float(np.mean(acc[v]))
        print(f"    {v:<16}: {m:.4f}  gain={m-always50:+.4f}")
    print()

    return {
        "model": model, "oracle": oracle, "always_50steps": always50,
        **{v: float(np.mean(acc[v])) for v in VARIANTS},
        **{f"{v}_seeds": [float(x) for x in acc[v]] for v in VARIANTS},
    }


def main():
    print("=== §6 DP路由 × 预测目标（gm/qwen + gm/flux）===\n")
    results = []
    for model in ["qwen", "flux"]:
        results.append(run_model(model))
    out = os.path.join(HERE, "results", "exp6_dp_variants.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
